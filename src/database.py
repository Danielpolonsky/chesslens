"""SQLite storage: normalized schema, inserts, duplicate handling and the eval cache."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import pandas as pd

from .engine_analysis import MoveEval, PositionEval
from .ingest import ParsedGame

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS games (
    game_id         INTEGER PRIMARY KEY,
    game_hash       TEXT NOT NULL UNIQUE,
    source          TEXT NOT NULL,
    site            TEXT,
    white_id        INTEGER NOT NULL REFERENCES players(player_id),
    black_id        INTEGER NOT NULL REFERENCES players(player_id),
    white_elo       INTEGER,
    black_elo       INTEGER,
    result          TEXT NOT NULL CHECK (result IN ('1-0', '0-1', '1/2-1/2', '*')),
    played_at       TEXT,
    time_control    TEXT,
    time_class      TEXT NOT NULL DEFAULT 'unknown',
    eco             TEXT,
    opening         TEXT,
    opening_family  TEXT NOT NULL DEFAULT 'Unknown',
    termination     TEXT,
    num_plies       INTEGER NOT NULL CHECK (num_plies > 0),
    pgn             TEXT,
    imported_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS moves (
    move_id             INTEGER PRIMARY KEY,
    game_id             INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ply                 INTEGER NOT NULL,
    move_number         INTEGER NOT NULL,
    color               TEXT NOT NULL CHECK (color IN ('white', 'black')),
    san                 TEXT NOT NULL,
    uci                 TEXT NOT NULL,
    fen_before          TEXT NOT NULL,
    phase               TEXT NOT NULL CHECK (phase IN ('opening', 'middlegame', 'endgame')),
    material_balance    INTEGER,
    total_material      INTEGER,
    legal_moves         INTEGER,
    in_check            INTEGER,
    own_vulnerable      INTEGER,
    opp_vulnerable      INTEGER,
    captures_available  INTEGER,
    checks_available    INTEGER,
    clock_before_s      REAL,
    UNIQUE (game_id, ply)
);

-- Engine evaluations keyed by position; doubles as the analysis cache.
CREATE TABLE IF NOT EXISTS position_evals (
    position_key    TEXT NOT NULL,
    depth           INTEGER NOT NULL,
    score_cp        INTEGER NOT NULL,
    mate            INTEGER,
    best_move_uci   TEXT,
    PRIMARY KEY (position_key, depth)
);

CREATE TABLE IF NOT EXISTS move_analysis (
    move_id         INTEGER PRIMARY KEY REFERENCES moves(move_id) ON DELETE CASCADE,
    eval_before     INTEGER NOT NULL,
    eval_after      INTEGER NOT NULL,
    best_move_uci   TEXT,
    best_move_san   TEXT,
    cpl             INTEGER NOT NULL CHECK (cpl >= 0),
    classification  TEXT NOT NULL
                    CHECK (classification IN ('good', 'inaccuracy', 'mistake', 'blunder')),
    win_prob_before REAL,
    win_prob_after  REAL,
    accuracy        REAL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    game_id     INTEGER PRIMARY KEY REFERENCES games(game_id) ON DELETE CASCADE,
    depth       INTEGER NOT NULL,
    engine      TEXT,
    analyzed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_games_white ON games(white_id);
CREATE INDEX IF NOT EXISTS idx_games_black ON games(black_id);
CREATE INDEX IF NOT EXISTS idx_games_played_at ON games(played_at);
CREATE INDEX IF NOT EXISTS idx_moves_game_color ON moves(game_id, color);

-- One row per (game, participant): lets every analytic query be written from the
-- perspective of a single player.
CREATE VIEW IF NOT EXISTS v_player_games AS
SELECT g.game_id, g.white_id AS player_id, g.black_id AS opponent_id, 'white' AS color,
       g.white_elo AS player_elo, g.black_elo AS opponent_elo,
       CASE g.result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 WHEN '1/2-1/2' THEN 0.5 END AS score,
       g.result, g.played_at, g.time_control, g.time_class, g.eco, g.opening, g.opening_family,
       g.termination, g.num_plies, g.site
FROM games g
UNION ALL
SELECT g.game_id, g.black_id, g.white_id, 'black',
       g.black_elo, g.white_elo,
       CASE g.result WHEN '0-1' THEN 1.0 WHEN '1-0' THEN 0.0 WHEN '1/2-1/2' THEN 0.5 END,
       g.result, g.played_at, g.time_control, g.time_class, g.eco, g.opening, g.opening_family,
       g.termination, g.num_plies, g.site
FROM games g;
"""

MOVE_FEATURE_COLUMNS = (
    "material_balance",
    "total_material",
    "legal_moves",
    "in_check",
    "own_vulnerable",
    "opp_vulnerable",
    "captures_available",
    "checks_available",
)


@dataclass
class InsertSummary:
    inserted: int = 0
    duplicates: int = 0
    game_ids: list[int] = field(default_factory=list)


def connect(path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        uri = f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def query_df(conn: sqlite3.Connection, sql: str, params: Sequence | dict = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def get_or_create_player(conn: sqlite3.Connection, username: str) -> int:
    conn.execute("INSERT OR IGNORE INTO players(username) VALUES (?)", (username,))
    row = conn.execute("SELECT player_id FROM players WHERE username = ?", (username,)).fetchone()
    return int(row[0])


def find_player_id(conn: sqlite3.Connection, username: str) -> int | None:
    row = conn.execute("SELECT player_id FROM players WHERE username = ?", (username,)).fetchone()
    return int(row[0]) if row else None


def insert_games(conn: sqlite3.Connection, games: Iterable[ParsedGame]) -> InsertSummary:
    """Insert games and their moves; games already present (same hash) are skipped."""
    summary = InsertSummary()
    with conn:
        for g in games:
            game_hash = g.game_hash
            existing = conn.execute(
                "SELECT game_id FROM games WHERE game_hash = ?", (game_hash,)
            ).fetchone()
            if existing:
                summary.duplicates += 1
                summary.game_ids.append(int(existing[0]))
                continue
            white_id = get_or_create_player(conn, g.white)
            black_id = get_or_create_player(conn, g.black)
            cur = conn.execute(
                """INSERT INTO games (game_hash, source, site, white_id, black_id, white_elo,
                       black_elo, result, played_at, time_control, time_class, eco, opening,
                       opening_family, termination, num_plies, pgn)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_hash, g.source, g.site, white_id, black_id, g.white_elo, g.black_elo,
                    g.result, g.played_at, g.time_control, g.time_class, g.eco, g.opening,
                    g.opening_family, g.termination, g.num_plies, g.pgn,
                ),
            )
            game_id = int(cur.lastrowid)
            conn.executemany(
                f"""INSERT INTO moves (game_id, ply, move_number, color, san, uci, fen_before,
                        phase, {", ".join(MOVE_FEATURE_COLUMNS)}, clock_before_s)
                    VALUES ({", ".join("?" * (9 + len(MOVE_FEATURE_COLUMNS)))})""",
                [
                    (
                        game_id, m.ply, m.move_number, m.color, m.san, m.uci, m.fen_before,
                        m.features["phase"], *(m.features[c] for c in MOVE_FEATURE_COLUMNS),
                        m.clock_before_s,
                    )
                    for m in g.moves
                ],
            )
            summary.inserted += 1
            summary.game_ids.append(game_id)
    return summary


def game_moves(conn: sqlite3.Connection, game_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT move_id, ply, fen_before, uci FROM moves WHERE game_id = ? ORDER BY ply",
        (game_id,),
    ).fetchall()


def save_game_analysis(
    conn: sqlite3.Connection, game_id: int, move_ids: list[int], evals: list[MoveEval],
    depth: int, engine: str,
) -> None:
    with conn:
        conn.executemany(
            """INSERT OR REPLACE INTO move_analysis (move_id, eval_before, eval_after,
                   best_move_uci, best_move_san, cpl, classification, win_prob_before,
                   win_prob_after, accuracy)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                (mid, e.eval_before, e.eval_after, e.best_move_uci, e.best_move_san, e.cpl,
                 e.classification, e.win_prob_before, e.win_prob_after, e.accuracy)
                for mid, e in zip(move_ids, evals)
            ],
        )
        conn.execute(
            "INSERT OR REPLACE INTO analysis_runs (game_id, depth, engine) VALUES (?,?,?)",
            (game_id, depth, engine),
        )


def cache_get(conn: sqlite3.Connection, key: str, depth: int) -> PositionEval | None:
    """Return a cached eval searched at ``depth`` or deeper."""
    row = conn.execute(
        """SELECT score_cp, mate, best_move_uci FROM position_evals
           WHERE position_key = ? AND depth >= ? ORDER BY depth DESC LIMIT 1""",
        (key, depth),
    ).fetchone()
    if row is None:
        return None
    return PositionEval(score_cp=row[0], mate=row[1], best_move_uci=row[2])


def cache_put(conn: sqlite3.Connection, key: str, depth: int, ev: PositionEval) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO position_evals VALUES (?,?,?,?,?)",
        (key, depth, ev.score_cp, ev.mate, ev.best_move_uci),
    )


def list_players(conn: sqlite3.Connection) -> pd.DataFrame:
    return query_df(
        conn,
        """SELECT p.username, COUNT(*) AS games,
                  SUM(CASE WHEN ar.game_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed
           FROM v_player_games vpg
           JOIN players p ON p.player_id = vpg.player_id
           LEFT JOIN analysis_runs ar ON ar.game_id = vpg.game_id
           GROUP BY p.player_id
           ORDER BY games DESC, p.username""",
    )


def unanalyzed_games(conn: sqlite3.Connection, username: str, limit: int) -> list[int]:
    """Most recent games of ``username`` that have not been analysed yet."""
    rows = conn.execute(
        """SELECT vpg.game_id
           FROM v_player_games vpg
           JOIN players p ON p.player_id = vpg.player_id
           LEFT JOIN analysis_runs ar ON ar.game_id = vpg.game_id
           WHERE p.username = ? AND ar.game_id IS NULL
           ORDER BY vpg.played_at DESC, vpg.game_id DESC
           LIMIT ?""",
        (username, limit),
    ).fetchall()
    return [int(r[0]) for r in rows]


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = ["players", "games", "moves", "position_evals", "move_analysis", "analysis_runs"]
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables}
