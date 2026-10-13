"""Query: SQL analytics that power every dashboard view.

All queries are written from the perspective of one player using the ``v_player_games``
view (one row per game participant), joined to ``moves`` (only that player's moves) and
``move_analysis`` (engine results). Filters are applied with bound parameters.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .database import query_df


@dataclass(frozen=True)
class Filters:
    color: str | None = None  # "white" | "black" | None
    date_from: str | None = None  # ISO date, inclusive
    date_to: str | None = None  # ISO date, inclusive
    time_classes: tuple[str, ...] = field(default_factory=tuple)
    openings: tuple[str, ...] = field(default_factory=tuple)


def player_cte(player: str, filters: Filters | None = None) -> tuple[str, list]:
    """CTEs ``pg`` (the player's games) and ``pm`` (the player's analysed moves)."""
    f = filters or Filters()
    where = ["p.username = ?"]
    params: list = [player]
    if f.color in ("white", "black"):
        where.append("vpg.color = ?")
        params.append(f.color)
    if f.date_from:
        where.append("date(vpg.played_at) >= date(?)")
        params.append(f.date_from)
    if f.date_to:
        where.append("date(vpg.played_at) <= date(?)")
        params.append(f.date_to)
    if f.time_classes:
        where.append(f"vpg.time_class IN ({','.join('?' * len(f.time_classes))})")
        params.extend(f.time_classes)
    if f.openings:
        where.append(f"vpg.opening_family IN ({','.join('?' * len(f.openings))})")
        params.extend(f.openings)
    sql = f"""
    WITH pg AS (
        SELECT vpg.*, opp.username AS opponent
        FROM v_player_games vpg
        JOIN players p   ON p.player_id = vpg.player_id
        JOIN players opp ON opp.player_id = vpg.opponent_id
        WHERE {' AND '.join(where)}
    ),
    pm AS (
        SELECT m.move_id, m.game_id, m.ply, m.move_number, m.phase, m.color, m.san, m.uci,
               m.fen_before, m.clock_before_s, m.material_balance,
               a.eval_before, a.eval_after, a.best_move_san, a.best_move_uci, a.cpl,
               a.classification, a.accuracy,
               CASE WHEN a.classification IN ('mistake', 'blunder') THEN 1 ELSE 0 END AS serious
        FROM pg
        JOIN moves m          ON m.game_id = pg.game_id AND m.color = pg.color
        JOIN move_analysis a  ON a.move_id = m.move_id
    ),
    ga AS (
        SELECT game_id, COUNT(*) AS n_moves, AVG(cpl) AS acpl, AVG(accuracy) AS accuracy,
               SUM(classification = 'inaccuracy') AS inaccuracies,
               SUM(classification = 'mistake') AS mistakes,
               SUM(classification = 'blunder') AS blunders
        FROM pm GROUP BY game_id
    )
    """
    return sql, params


def _run(conn: sqlite3.Connection, player: str, filters: Filters | None, body: str,
         extra_params: list | None = None) -> pd.DataFrame:
    cte, params = player_cte(player, filters)
    return query_df(conn, cte + body, params + (extra_params or []))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def overview(conn: sqlite3.Connection, player: str, filters: Filters | None = None) -> dict:
    df = _run(
        conn, player, filters,
        """
        SELECT
            (SELECT COUNT(*) FROM pg) AS games,
            (SELECT COUNT(*) FROM pg WHERE score IS NOT NULL) AS decided_games,
            (SELECT COUNT(*) FROM ga) AS analyzed_games,
            (SELECT SUM(score = 1.0) FROM pg) AS wins,
            (SELECT SUM(score = 0.5) FROM pg) AS draws,
            (SELECT SUM(score = 0.0) FROM pg) AS losses,
            (SELECT AVG(score) FROM pg) AS avg_score,
            (SELECT COUNT(*) FROM pm) AS moves,
            (SELECT AVG(cpl) FROM pm) AS acpl,
            (SELECT AVG(accuracy) FROM pm) AS accuracy,
            (SELECT AVG(classification = 'inaccuracy') FROM pm) AS inaccuracy_rate,
            (SELECT AVG(classification = 'mistake') FROM pm) AS mistake_rate,
            (SELECT AVG(classification = 'blunder') FROM pm) AS blunder_rate,
            (SELECT AVG(serious) FROM pm) AS serious_rate
        """,
    )
    row = df.iloc[0].to_dict()
    for k in ("games", "decided_games", "analyzed_games", "wins", "draws", "losses", "moves"):
        row[k] = int(row[k] or 0)
    return row


def by_color(conn, player, filters=None) -> pd.DataFrame:
    return _run(
        conn, player, filters,
        """
        SELECT pg.color,
               COUNT(*) AS games,
               AVG(pg.score = 1.0) AS win_rate,
               AVG(pg.score = 0.5) AS draw_rate,
               AVG(pg.score = 0.0) AS loss_rate,
               AVG(pg.score) AS avg_score,
               SUM(ga.n_moves) AS moves,
               SUM(ga.acpl * ga.n_moves) / SUM(ga.n_moves) AS acpl,
               1.0 * SUM(ga.blunders) / SUM(ga.n_moves) AS blunder_rate,
               1.0 * (SUM(ga.mistakes) + SUM(ga.blunders)) / SUM(ga.n_moves) AS serious_rate
        FROM pg LEFT JOIN ga ON ga.game_id = pg.game_id
        WHERE pg.score IS NOT NULL
        GROUP BY pg.color
        ORDER BY pg.color DESC
        """,
    )


def games_table(conn, player, filters=None) -> pd.DataFrame:
    df = _run(
        conn, player, filters,
        """
        SELECT pg.game_id, pg.played_at, pg.color, pg.opponent, pg.player_elo, pg.opponent_elo,
               pg.result, pg.score, pg.time_control, pg.time_class, pg.eco, pg.opening,
               pg.opening_family, pg.termination, pg.num_plies, pg.site,
               ga.n_moves, ga.acpl, ga.accuracy, ga.inaccuracies, ga.mistakes, ga.blunders
        FROM pg LEFT JOIN ga ON ga.game_id = pg.game_id
        ORDER BY pg.played_at, pg.game_id
        """,
    )
    df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
    df["outcome"] = df["score"].map({1.0: "Win", 0.5: "Draw", 0.0: "Loss"}).fillna("Unfinished")
    return df


# ---------------------------------------------------------------------------
# Openings, phases, time controls
# ---------------------------------------------------------------------------


def opening_stats(conn, player, filters=None) -> pd.DataFrame:
    df = _run(
        conn, player, filters,
        """
        SELECT pg.opening_family AS opening,
               COUNT(*) AS games,
               SUM(pg.score = 1.0) AS wins,
               SUM(pg.score = 0.5) AS draws,
               SUM(pg.score = 0.0) AS losses,
               AVG(pg.score) AS avg_score,
               SUM(pg.color = 'white') AS as_white,
               SUM(pg.color = 'black') AS as_black,
               SUM(ga.n_moves) AS moves,
               SUM(ga.acpl * ga.n_moves) / NULLIF(SUM(ga.n_moves), 0) AS acpl,
               1.0 * (SUM(ga.mistakes) + SUM(ga.blunders)) / NULLIF(SUM(ga.n_moves), 0)
                   AS serious_rate,
               op.opening_moves,
               op.opening_inaccuracy_rate
        FROM pg
        LEFT JOIN ga ON ga.game_id = pg.game_id
        LEFT JOIN (
            SELECT pg2.opening_family,
                   COUNT(*) AS opening_moves,
                   AVG(pm.classification != 'good') AS opening_inaccuracy_rate
            FROM pm JOIN pg AS pg2 ON pg2.game_id = pm.game_id
            WHERE pm.phase = 'opening'
            GROUP BY pg2.opening_family
        ) op ON op.opening_family = pg.opening_family
        GROUP BY pg.opening_family
        ORDER BY games DESC, opening
        """,
    )
    df["win_rate"] = df["wins"] / df["games"]
    df["small_sample"] = df["games"] < config.MIN_GAMES_FOR_OPENING_CLAIM
    return df


def phase_stats(conn, player, filters=None, split_by_color: bool = False) -> pd.DataFrame:
    group = "pm.phase, pg.color" if split_by_color else "pm.phase"
    select_color = "pg.color," if split_by_color else ""
    df = _run(
        conn, player, filters,
        f"""
        SELECT pm.phase, {select_color}
               COUNT(*) AS moves,
               AVG(pm.cpl) AS acpl,
               AVG(pm.accuracy) AS accuracy,
               AVG(pm.classification = 'inaccuracy') AS inaccuracy_rate,
               AVG(pm.classification = 'mistake') AS mistake_rate,
               AVG(pm.classification = 'blunder') AS blunder_rate,
               AVG(pm.serious) AS serious_rate,
               SUM(pm.serious) AS serious_errors
        FROM pm JOIN pg ON pg.game_id = pm.game_id
        GROUP BY {group}
        ORDER BY CASE pm.phase WHEN 'opening' THEN 1 WHEN 'middlegame' THEN 2 ELSE 3 END
        """,
    )
    return df


def time_class_stats(conn, player, filters=None) -> pd.DataFrame:
    return _run(
        conn, player, filters,
        """
        SELECT pg.time_class,
               COUNT(*) AS games,
               AVG(pg.score) AS avg_score,
               AVG(pg.score = 1.0) AS win_rate,
               SUM(ga.n_moves) AS moves,
               SUM(ga.acpl * ga.n_moves) / NULLIF(SUM(ga.n_moves), 0) AS acpl,
               1.0 * (SUM(ga.mistakes) + SUM(ga.blunders)) / NULLIF(SUM(ga.n_moves), 0)
                   AS serious_rate
        FROM pg LEFT JOIN ga ON ga.game_id = pg.game_id
        WHERE pg.score IS NOT NULL
        GROUP BY pg.time_class
        ORDER BY games DESC
        """,
    )


def move_number_profile(conn, player, filters=None, bucket: int = 10) -> pd.DataFrame:
    df = _run(
        conn, player, filters,
        """
        SELECT ((pm.move_number - 1) / ?) AS bucket,
               COUNT(*) AS moves,
               AVG(pm.cpl) AS acpl,
               AVG(pm.serious) AS serious_rate,
               AVG(pm.classification = 'blunder') AS blunder_rate
        FROM pm
        GROUP BY bucket ORDER BY bucket
        """,
        [bucket],
    )
    df["label"] = df["bucket"].map(lambda b: f"{b * bucket + 1}-{(b + 1) * bucket}")
    return df


def eval_context_stats(conn, player, filters=None, threshold: int = 200) -> pd.DataFrame:
    """Error rates when the player was clearly better, roughly equal, or clearly worse."""
    return _run(
        conn, player, filters,
        """
        SELECT CASE WHEN pm.eval_before >= ? THEN 'winning'
                    WHEN pm.eval_before <= -? THEN 'losing'
                    ELSE 'balanced' END AS context,
               COUNT(*) AS moves,
               AVG(pm.cpl) AS acpl,
               AVG(pm.serious) AS serious_rate,
               AVG(pm.classification = 'blunder') AS blunder_rate
        FROM pm
        GROUP BY context
        """,
        [threshold, threshold],
    )


def time_pressure_stats(conn, player, filters=None, fraction: float = 0.1) -> pd.DataFrame:
    """Error rate with less than ``fraction`` of the base clock left (needs %clk data)."""
    return _run(
        conn, player, filters,
        """
        , clocked AS (
            SELECT pm.*,
                   CASE WHEN instr(pg.time_control, '+') > 0
                        THEN CAST(substr(pg.time_control, 1, instr(pg.time_control, '+') - 1) AS REAL)
                        ELSE CAST(pg.time_control AS REAL) END AS base_s
            FROM pm JOIN pg ON pg.game_id = pm.game_id
            WHERE pm.clock_before_s IS NOT NULL AND pg.time_control GLOB '[0-9]*'
        )
        SELECT CASE WHEN clock_before_s < ? * base_s THEN 'low_clock' ELSE 'normal' END AS clock,
               COUNT(*) AS moves,
               AVG(serious) AS serious_rate,
               AVG(classification = 'blunder') AS blunder_rate
        FROM clocked
        WHERE base_s > 0 AND ply > 2
        GROUP BY clock
        """,
        [fraction],
    )


def trend(conn, player, filters=None) -> pd.DataFrame:
    """Per-period aggregates; granularity adapts to the date span (day / week / month)."""
    span = _run(
        conn, player, filters,
        "SELECT julianday(MAX(played_at)) - julianday(MIN(played_at)) AS days FROM pg",
    ).iloc[0]["days"]
    days = 0 if span is None or (isinstance(span, float) and math.isnan(span)) else span
    fmt = "%Y-%m" if days > 120 else ("%Y-W%W" if days > 21 else "%Y-%m-%d")
    df = _run(
        conn, player, filters,
        """
        SELECT strftime(?, pg.played_at) AS period,
               MIN(date(pg.played_at)) AS period_start,
               COUNT(*) AS games,
               AVG(pg.score) AS avg_score,
               AVG(pg.player_elo) AS avg_rating,
               SUM(ga.n_moves) AS moves,
               SUM(ga.acpl * ga.n_moves) / NULLIF(SUM(ga.n_moves), 0) AS acpl,
               1.0 * SUM(ga.blunders) / NULLIF(SUM(ga.n_moves), 0) AS blunder_rate,
               1.0 * SUM(ga.mistakes) / NULLIF(SUM(ga.n_moves), 0) AS mistake_rate
        FROM pg LEFT JOIN ga ON ga.game_id = pg.game_id
        WHERE pg.played_at IS NOT NULL
        GROUP BY period ORDER BY period
        """,
        [fmt],
    )
    df.attrs["granularity"] = {"%Y-%m": "month", "%Y-W%W": "week"}.get(fmt, "day")
    return df


def move_rows(conn, player, filters=None) -> pd.DataFrame:
    """Every analysed move of the player (for distributions and exports)."""
    return _run(
        conn, player, filters,
        """
        SELECT pm.game_id, pg.played_at, pg.color, pg.opponent, pg.opening_family,
               pg.time_class, pm.ply, pm.move_number, pm.phase, pm.san AS move_played,
               pm.best_move_san AS engine_best, pm.eval_before, pm.eval_after, pm.cpl,
               pm.classification, pm.accuracy, pm.fen_before
        FROM pm JOIN pg ON pg.game_id = pm.game_id
        ORDER BY pg.played_at, pm.game_id, pm.ply
        """,
    )


def mistakes(conn, player, filters=None, categories=("blunder", "mistake"),
             extra_where: str = "", extra_params: list | None = None, limit: int = 200) -> pd.DataFrame:
    """Critical mistakes with full game context, worst first.

    ``extra_where`` is a trusted, code-defined SQL fragment (never user text).
    """
    cats = list(categories)
    return _run(
        conn, player, filters,
        f"""
        SELECT pm.move_id, pm.game_id, pg.played_at, pg.color, pg.opponent, pg.player_elo,
               pg.opponent_elo, pg.result, pg.score, pg.opening, pg.opening_family,
               pg.time_control, pg.time_class, pg.site, pm.ply, pm.move_number, pm.phase,
               pm.san, pm.uci, pm.best_move_san, pm.best_move_uci, pm.fen_before,
               pm.eval_before, pm.eval_after, pm.cpl, pm.classification
        FROM pm JOIN pg ON pg.game_id = pm.game_id
        WHERE pm.classification IN ({','.join('?' * len(cats))}) {extra_where}
        ORDER BY pm.cpl DESC, pg.played_at DESC
        LIMIT ?
        """,
        cats + (extra_params or []) + [limit],
    )


def game_analysis(conn: sqlite3.Connection, game_id: int) -> pd.DataFrame:
    """Both players' analysed moves for one game (for the evaluation curve)."""
    return query_df(
        conn,
        """SELECT m.ply, m.move_number, m.color, m.san, m.phase, a.eval_before, a.eval_after,
                  a.cpl, a.classification, a.best_move_san
           FROM moves m JOIN move_analysis a ON a.move_id = m.move_id
           WHERE m.game_id = ? ORDER BY m.ply""",
        [game_id],
    )


# ---------------------------------------------------------------------------
# Filter options
# ---------------------------------------------------------------------------


def filter_options(conn, player) -> dict:
    df = _run(
        conn, player, None,
        """SELECT DISTINCT time_class, opening_family, date(played_at) AS d FROM pg""",
    )
    dates = pd.to_datetime(df["d"], errors="coerce").dropna()
    openings = (
        df["opening_family"].value_counts().index.tolist() if not df.empty else []
    )
    return {
        "time_classes": sorted(df["time_class"].dropna().unique().tolist()),
        "openings": openings,
        "date_min": dates.min().date() if not dates.empty else None,
        "date_max": dates.max().date() if not dates.empty else None,
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (robust for small samples)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(x1: float, n1: int, x2: float, n2: int) -> float:
    """z statistic for H0: p1 == p2 (pooled). Positive when group 1 has the higher rate."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (p1 - p2) / se
