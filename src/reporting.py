"""Exports: CSV downloads and a Markdown coaching summary."""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import config
from .recommendations import Recommendation


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _num(x, fmt: str = "{:.0f}") -> str:
    return "n/a" if x is None or pd.isna(x) else fmt.format(x)


def coaching_summary(
    player: str,
    overview: dict,
    phases: pd.DataFrame,
    openings: pd.DataFrame,
    recs: list[Recommendation],
    note: str | None,
    dataset_label: str,
) -> str:
    lines = [
        f"# ChessLens coaching summary - {player}",
        "",
        f"_Generated {date.today().isoformat()} from {dataset_label}. Built for Polonsky Academy._",
        "",
        "## Snapshot",
        "",
        f"- Games: {overview['games']} (engine-analyzed: {overview['analyzed_games']})",
        f"- Results: {overview['wins']}W / {overview['draws']}D / {overview['losses']}L "
        f"(score {_pct(overview['avg_score'])})",
        f"- Analyzed moves by this player: {overview['moves']}",
        f"- Average centipawn loss: {_num(overview['acpl'])}",
        f"- Inaccuracy / mistake / blunder rate per move: {_pct(overview['inaccuracy_rate'])} / "
        f"{_pct(overview['mistake_rate'])} / {_pct(overview['blunder_rate'])}",
        "",
        "## Top practice priorities",
        "",
    ]
    if not recs:
        lines.append(note or "Not enough analyzed data for recommendations yet.")
    for i, r in enumerate(recs, 1):
        lines += [
            f"### {i}. {r.title}",
            "",
            f"**Evidence ({r.confidence.lower()}):** {r.evidence}",
            "",
            f"**Practice:** {r.advice}",
            "",
        ]
        if r.examples:
            lines.append("**Examples from your games:**")
            lines += [f"- {e['label']}" for e in r.examples]
            lines.append("")
    if recs and note:
        lines += [f"_Note: {note}_", ""]

    if not phases.empty:
        lines += ["## Game-phase breakdown", "", "| Phase | Moves | ACPL | Mistake rate | Blunder rate |",
                  "|---|---:|---:|---:|---:|"]
        for r in phases.itertuples():
            lines.append(f"| {r.phase.title()} | {int(r.moves)} | {_num(r.acpl)} | "
                         f"{_pct(r.mistake_rate)} | {_pct(r.blunder_rate)} |")
        lines.append("")

    top = openings.sort_values("games", ascending=False).head(8) if not openings.empty else openings
    if not top.empty:
        lines += ["## Most-played openings", "", "| Opening | Games | Score | ACPL |", "|---|---:|---:|---:|"]
        for r in top.itertuples():
            lines.append(f"| {r.opening} | {int(r.games)} | {_pct(r.avg_score)} | {_num(r.acpl)} |")
        lines.append("")

    lines += [
        "## Method notes",
        "",
        f"- Moves are evaluated with Stockfish; centipawn loss is measured from the mover's perspective "
        f"with evaluations clamped to +/-{config.EVAL_CAP_CP} cp (mates count as the cap).",
        f"- Categories are project-defined: inaccuracy {config.INACCURACY_CP}-{config.MISTAKE_CP - 1} cp, "
        f"mistake {config.MISTAKE_CP}-{config.BLUNDER_CP - 1} cp, blunder {config.BLUNDER_CP}+ cp "
        "(not official Chess.com or Lichess classifications).",
        "- Recommendations are rule-based comparisons of your own error rates and scores; small samples "
        "are flagged and should be read as tentative.",
        "",
    ]
    return "\n".join(lines)
