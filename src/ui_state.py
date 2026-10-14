"""Pure helpers for UI state (kept out of ``app.py`` so they can be unit-tested)."""

from __future__ import annotations

from datetime import date

from .analytics import Filters


def pick_move(move_ids: list[int], requested: int | None, current: int | None) -> int | None:
    """Which move the explorer should show: an explicit request (e.g. a coaching example)
    wins, then the move already on screen, then the first in the list."""
    if not move_ids:
        return None
    if requested in move_ids:
        return requested
    if current in move_ids:
        return current
    return move_ids[0]


def step_move(move_ids: list[int], current: int | None, delta: int) -> int | None:
    if not move_ids:
        return None
    i = move_ids.index(current) if current in move_ids else 0
    return move_ids[max(0, min(len(move_ids) - 1, i + delta))]


def describe_filters(f: Filters, date_bounds: tuple[date | None, date | None] = (None, None)) -> list[str]:
    """Human-readable list of the filters that actually narrow the data."""
    parts = []
    if f.color:
        parts.append(f"as {f.color.title()}")
    if f.time_classes:
        parts.append(", ".join(t.title() for t in f.time_classes))
    if f.openings:
        parts.append(f.openings[0] if len(f.openings) == 1 else f"{len(f.openings)} openings")
    lo, hi = date_bounds
    narrowed = (f.date_from and lo and f.date_from > lo.isoformat()) or (
        f.date_to and hi and f.date_to < hi.isoformat())
    if narrowed:
        parts.append(f"{f.date_from} to {f.date_to}")
    return parts


def sample_unit(area: str) -> str:
    """Unit of ``Recommendation.sample_size`` for each rule kind."""
    return "moves" if area.split(":")[0] in ("phase", "context") else "games"
