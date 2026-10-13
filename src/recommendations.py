"""Rule-based coaching recommendations grounded in the player's own statistics.

Each rule compares a slice of the player's moves/games (a phase, a colour, an opening,
...) with the rest of their data and scores the gap with a z statistic, so larger and
better-supported weaknesses rank first. Rules require minimum sample sizes and every
recommendation carries its supporting numbers and concrete example mistakes.

ChessLens does not detect specific tactical motifs (forks, pins, ...); advice is phrased
around phases, colours, openings and contexts that are actually measured.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from . import analytics as an
from . import config

MIN_ANALYZED_GAMES = 3
MIN_CONTEXT_MOVES = 25
MIN_COLOR_GAMES = 5
MIN_OPENING_GAMES = 5
SCORE_SD = 0.45  # approx. per-game standard deviation of a 1 / 0.5 / 0 score

PHASE_ADVICE = {
    "opening": (
        "Tighten your opening play: replay the first 12 moves of your recent games against "
        "the engine line and write down where you left known theory. Build a short "
        "repertoire file for the openings you play most."
    ),
    "middlegame": (
        "Prioritize middlegame calculation and threat-recognition exercises. Before each "
        "move, list the opponent's checks, captures and threats, and solve 10-15 mixed "
        "tactics puzzles per session."
    ),
    "endgame": (
        "Study fundamental endgames (king activity, rook endgames, pawn races) and practise "
        "converting simplified positions against an engine or training partner."
    ),
}


@dataclass
class Recommendation:
    area: str
    title: str
    evidence: str
    advice: str
    z: float
    sample_size: int
    examples: list[dict] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if self.z >= 2.5:
            return "Strong evidence"
        if self.z >= 1.5:
            return "Moderate evidence"
        return "Tentative - small or modest gap"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


EXAMPLES_PER_REC = 2


def _examples(conn, player, filters, where: str = "", params: list | None = None, n: int = 6) -> list[dict]:
    df = an.mistakes(conn, player, filters, extra_where=where, extra_params=params, limit=n)
    out = []
    for r in df.itertuples():
        date = str(r.played_at)[:10] if r.played_at else "unknown date"
        move = f"{r.move_number}{'.' if r.color == 'white' else '...'}{r.san}"
        out.append(
            {
                "game_id": int(r.game_id),
                "move_id": int(r.move_id),
                "label": (
                    f"{move} vs {r.opponent} ({date}) - engine preferred "
                    f"{r.best_move_san or '?'}, eval {r.eval_before / 100:+.1f} -> "
                    f"{r.eval_after / 100:+.1f} ({r.classification}, {int(r.cpl)} cp lost)"
                ),
            }
        )
    return out


def _phase_rules(conn, player, filters) -> list[Recommendation]:
    df = an.phase_stats(conn, player, filters)
    if df.empty:
        return []
    total_moves = int(df["moves"].sum())
    total_serious = float(df["serious_errors"].sum())
    recs = []
    for r in df.itertuples():
        n_rest = total_moves - int(r.moves)
        if r.moves < config.MIN_MOVES_FOR_PHASE_CLAIM or n_rest < config.MIN_MOVES_FOR_PHASE_CLAIM:
            continue
        x_rest = total_serious - float(r.serious_errors)
        z = an.two_proportion_z(float(r.serious_errors), int(r.moves), x_rest, n_rest)
        if r.serious_rate <= x_rest / n_rest:
            continue
        others = df[df["phase"] != r.phase]
        best_other = others.loc[others["serious_rate"].idxmin()]
        evidence = (
            f"Your {r.phase} serious-error rate (mistakes + blunders) is {_pct(r.serious_rate)} "
            f"across {int(r.moves)} analyzed {r.phase} moves, compared with "
            f"{_pct(best_other.serious_rate)} in the {best_other.phase} "
            f"({int(best_other.moves)} moves). Blunders alone: {_pct(r.blunder_rate)} of "
            f"{r.phase} moves; average centipawn loss {r.acpl:.0f}."
        )
        recs.append(
            Recommendation(
                area=f"phase:{r.phase}",
                title=f"Reduce {r.phase} errors",
                evidence=evidence,
                advice=PHASE_ADVICE[r.phase],
                z=z,
                sample_size=int(r.moves),
                examples=_examples(conn, player, filters, "AND pm.phase = ?", [r.phase]),
            )
        )
    return recs


def _color_rule(conn, player, filters) -> list[Recommendation]:
    if filters and filters.color:
        return []
    df = an.by_color(conn, player, filters).set_index("color")
    if not {"white", "black"} <= set(df.index):
        return []
    w, b = df.loc["white"], df.loc["black"]
    if min(w.games, b.games) < MIN_COLOR_GAMES:
        return []
    weak, strong = ("black", "white") if b.avg_score < w.avg_score else ("white", "black")
    ws, ss = df.loc[weak], df.loc[strong]
    se = SCORE_SD * (1 / ws.games + 1 / ss.games) ** 0.5
    z_score = (ss.avg_score - ws.avg_score) / se
    z_err = 0.0
    if pd.notna(ws.moves) and pd.notna(ss.moves) and ws.moves and ss.moves:
        z_err = an.two_proportion_z(
            ws.serious_rate * ws.moves, int(ws.moves), ss.serious_rate * ss.moves, int(ss.moves)
        )
    z = max(z_score, z_err)
    if z <= 0:
        return []
    evidence = (
        f"As {weak.title()} you scored {_pct(ws.avg_score)} over {int(ws.games)} games versus "
        f"{_pct(ss.avg_score)} over {int(ss.games)} games as {strong.title()}."
    )
    if pd.notna(ws.serious_rate) and pd.notna(ss.serious_rate):
        evidence += (
            f" Serious-error rate: {_pct(ws.serious_rate)} as {weak.title()} vs "
            f"{_pct(ss.serious_rate)} as {strong.title()}."
        )
    advice = (
        f"Give your {weak.title()} repertoire extra attention: pick one reliable system, "
        f"study two or three model games in it, and review your recent {weak.title()} losses "
        "to find the move where the evaluation first turned against you."
    )
    return [
        Recommendation(
            area=f"color:{weak}", title=f"Strengthen your games as {weak.title()}",
            evidence=evidence, advice=advice, z=z, sample_size=int(ws.games),
            examples=_examples(conn, player, filters, "AND pg.color = ?", [weak]),
        )
    ]


def _opening_rule(conn, player, filters, overall_score: float) -> list[Recommendation]:
    df = an.opening_stats(conn, player, filters)
    df = df[(df["games"] >= MIN_OPENING_GAMES) & (df["opening"] != "Unknown")]
    if df.empty or pd.isna(overall_score):
        return []
    df = df.assign(z=(overall_score - df["avg_score"]) / (SCORE_SD / df["games"] ** 0.5))
    worst = df.sort_values("z", ascending=False).iloc[0]
    if worst.z <= 0:
        return []
    evidence = (
        f"In the {worst.opening} you scored {_pct(worst.avg_score)} across {int(worst.games)} "
        f"games ({int(worst.wins)}W / {int(worst.draws)}D / {int(worst.losses)}L), compared "
        f"with {_pct(overall_score)} overall."
    )
    if pd.notna(worst.acpl):
        evidence += f" Average centipawn loss in these games: {worst.acpl:.0f}."
    if pd.notna(worst.opening_inaccuracy_rate):
        evidence += (
            f" {_pct(worst.opening_inaccuracy_rate)} of your opening-phase moves in this "
            "opening were inaccuracies or worse."
        )
    return [
        Recommendation(
            area=f"opening:{worst.opening}", title=f"Repair your {worst.opening}",
            evidence=evidence,
            advice=(
                f"Review your {worst.opening} games move by move, identify the first point where "
                "you left the engine's preferred line, and study the typical plans and pawn "
                "structures of that opening. If results stay poor, consider a simpler alternative."
            ),
            z=float(worst.z), sample_size=int(worst.games),
            examples=_examples(
                conn, player, filters, "AND pg.opening_family = ?", [worst.opening]
            ),
        )
    ]


def _context_rules(conn, player, filters) -> list[Recommendation]:
    recs = []
    ctx = an.eval_context_stats(conn, player, filters).set_index("context")
    if "winning" in ctx.index:
        win = ctx.loc["winning"]
        rest = ctx.drop(index="winning")
        n_rest = int(rest["moves"].sum())
        if win.moves >= MIN_CONTEXT_MOVES and n_rest >= MIN_CONTEXT_MOVES:
            x_rest = float((rest["serious_rate"] * rest["moves"]).sum())
            z = an.two_proportion_z(win.serious_rate * win.moves, int(win.moves), x_rest, n_rest)
            if z > 0:
                recs.append(
                    Recommendation(
                        area="context:winning", title="Convert winning positions more reliably",
                        evidence=(
                            f"When you were clearly better (engine eval at least +2.0), "
                            f"{_pct(win.serious_rate)} of your {int(win.moves)} moves were mistakes "
                            f"or blunders, versus {_pct(x_rest / n_rest)} in other positions."
                        ),
                        advice=(
                            "Practise converting advantages: when ahead, trade pieces (not pawns), "
                            "remove your opponent's counterplay before attacking, and do a final "
                            "blunder check on every capture and check available to your opponent."
                        ),
                        z=z, sample_size=int(win.moves),
                        examples=_examples(conn, player, filters, "AND pm.eval_before >= 200"),
                    )
                )

    clock = an.time_pressure_stats(conn, player, filters).set_index("clock")
    if {"low_clock", "normal"} <= set(clock.index):
        lo, no = clock.loc["low_clock"], clock.loc["normal"]
        if lo.moves >= 15 and no.moves >= MIN_CONTEXT_MOVES:
            z = an.two_proportion_z(lo.serious_rate * lo.moves, int(lo.moves),
                                    no.serious_rate * no.moves, int(no.moves))
            if z > 0:
                recs.append(
                    Recommendation(
                        area="context:time", title="Manage your clock",
                        evidence=(
                            f"With under 10% of your starting time left, {_pct(lo.serious_rate)} of "
                            f"your {int(lo.moves)} moves were mistakes or blunders, versus "
                            f"{_pct(no.serious_rate)} with more time ({int(no.moves)} moves)."
                        ),
                        advice=(
                            "Budget time per phase (e.g. spend less in the opening you know), "
                            "and practise faster decision-making in simplified positions."
                        ),
                        z=z, sample_size=int(lo.moves),
                    )
                )

    late = an.move_number_profile(conn, player, filters, bucket=30)
    if len(late) >= 2:
        early, later = late.iloc[0], late.iloc[1:]
        n_late = int(later["moves"].sum())
        if early.moves >= MIN_CONTEXT_MOVES and n_late >= MIN_CONTEXT_MOVES:
            x_late = float((later["serious_rate"] * later["moves"]).sum())
            z = an.two_proportion_z(x_late, n_late, early.serious_rate * early.moves, int(early.moves))
            if z > 0:
                recs.append(
                    Recommendation(
                        area="context:late", title="Stay sharp late in the game",
                        evidence=(
                            f"After move 30 your serious-error rate is {_pct(x_late / n_late)} "
                            f"({n_late} moves) compared with {_pct(early.serious_rate)} in moves "
                            f"1-30 ({int(early.moves)} moves)."
                        ),
                        advice=(
                            "Long games expose concentration and time-management issues. Play some "
                            "longer time controls, and in training games keep a short "
                            "checks-captures-threats routine going all the way to the end."
                        ),
                        z=z, sample_size=n_late,
                        examples=_examples(conn, player, filters, "AND pm.move_number > 30"),
                    )
                )
    return recs


def _time_class_rule(conn, player, filters, overall_score: float) -> list[Recommendation]:
    if filters and len(filters.time_classes) == 1:
        return []
    df = an.time_class_stats(conn, player, filters)
    df = df[(df["games"] >= MIN_COLOR_GAMES) & (df["time_class"] != "unknown")]
    if len(df) < 2 or pd.isna(overall_score):
        return []
    df = df.assign(z=(overall_score - df["avg_score"]) / (SCORE_SD / df["games"] ** 0.5))
    worst = df.sort_values("z", ascending=False).iloc[0]
    if worst.z <= 0:
        return []
    return [
        Recommendation(
            area=f"time:{worst.time_class}", title=f"Improve your {worst.time_class} results",
            evidence=(
                f"You scored {_pct(worst.avg_score)} in {int(worst.games)} {worst.time_class} games "
                f"compared with {_pct(overall_score)} across all time controls"
                + (f"; serious-error rate {_pct(worst.serious_rate)}." if pd.notna(worst.serious_rate) else ".")
            ),
            advice=(
                f"In {worst.time_class} games, rely on familiar structures and avoid sharp lines you "
                "do not know. If you are training for slower events, reduce your share of faster games."
            ),
            z=float(worst.z), sample_size=int(worst.games),
            examples=_examples(conn, player, filters, "AND pg.time_class = ?", [worst.time_class]),
        )
    ]


def generate(conn: sqlite3.Connection, player: str, filters: an.Filters | None = None,
             top_n: int = 3) -> tuple[list[Recommendation], str | None]:
    """Return the top recommendations and an optional note about data sufficiency."""
    ov = an.overview(conn, player, filters)
    if ov["analyzed_games"] < MIN_ANALYZED_GAMES:
        return [], (
            f"At least {MIN_ANALYZED_GAMES} engine-analyzed games are needed for recommendations "
            f"(currently {ov['analyzed_games']}). Analyze more games on the Import page."
        )
    overall_score = ov["avg_score"] if ov["avg_score"] is not None else float("nan")
    candidates: list[Recommendation] = []
    candidates += _phase_rules(conn, player, filters)
    candidates += _color_rule(conn, player, filters)
    candidates += _opening_rule(conn, player, filters, overall_score)
    candidates += _context_rules(conn, player, filters)
    candidates += _time_class_rule(conn, player, filters, overall_score)
    candidates.sort(key=lambda r: r.z, reverse=True)

    chosen: list[Recommendation] = []
    seen_kinds: set[str] = set()
    for rec in candidates:
        kind = rec.area.split(":")[0]
        if kind in seen_kinds and kind != "context":
            continue
        chosen.append(rec)
        seen_kinds.add(kind)
        if len(chosen) == top_n:
            break
    used: set[int] = set()
    for rec in chosen:
        fresh = [e for e in rec.examples if e["move_id"] not in used][:EXAMPLES_PER_REC]
        used.update(e["move_id"] for e in fresh)
        rec.examples = fresh
    note = None
    if len(chosen) < top_n:
        note = (
            "Fewer than three weaknesses stood out clearly in this data - more analyzed games "
            "will make the recommendations more specific."
        )
    if ov["analyzed_games"] < 15:
        note = (note + " " if note else "") + (
            f"Based on {ov['analyzed_games']} analyzed games; treat these as early signals."
        )
    return chosen, note
