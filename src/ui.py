"""Streamlit presentation helpers: theme CSS, stat cards, empty states, chessboards."""

from __future__ import annotations

import html
from typing import Iterable

import chess
import chess.svg
import streamlit as st

NAVY = "#1F3B5B"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
ORANGE = "#F4A261"
RED = "#E76F51"
SLATE = "#64748B"
LIGHT = "#CBD5E1"

CLASS_COLORS = {"good": TEAL, "inaccuracy": AMBER, "mistake": ORANGE, "blunder": RED}
OUTCOME_COLORS = {"Win": TEAL, "Draw": "#94A3B8", "Loss": RED, "Unfinished": LIGHT}
COLOR_COLORS = {"white": "#B8C4D3", "black": NAVY}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; color: #0F172A; }
.cl-brand { display:flex; align-items:baseline; gap:.6rem; margin-bottom:.15rem; }
.cl-brand .logo { font-size: 2.1rem; font-weight: 700; color: #1F3B5B; letter-spacing:-0.02em; }
.cl-brand .logo span { color: #2A9D8F; }
.cl-brand .tag { color:#64748B; font-size:.95rem; }
.cl-page-title { font-size: 1.55rem; font-weight: 650; color:#0F172A; margin: .2rem 0 .1rem 0; }
.cl-page-sub { color:#64748B; font-size:.93rem; margin-bottom: 1.1rem; }
.cl-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .8rem;
  margin-bottom: 1rem; }
.cl-card { background:#FFFFFF; border:1px solid #E5E7EB; border-radius: 12px; padding: .95rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.cl-card .label { color:#64748B; font-size:.78rem; font-weight:600; text-transform:uppercase;
  letter-spacing:.04em; }
.cl-card .value { color:#0F172A; font-size:1.65rem; font-weight:700; margin-top:.15rem; line-height:1.2; }
.cl-card .sub { color:#64748B; font-size:.82rem; margin-top:.2rem; }
.cl-card.accent { border-left: 4px solid #2A9D8F; }
.cl-card.warn { border-left: 4px solid #E76F51; }
.cl-hero { background: linear-gradient(135deg, #1F3B5B 0%, #24496F 60%, #2A9D8F 140%);
  color: #F8FAFC; border-radius: 16px; padding: 2.1rem 2.2rem; margin-bottom: 1.2rem; }
.cl-hero h1 { color:#FFFFFF; font-size: 2.3rem; margin: 0 0 .4rem 0; font-weight:700; }
.cl-hero p { color:#DCE6F0; font-size: 1.04rem; max-width: 760px; margin:0; }
.cl-feature { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:1rem 1.1rem;
  height:100%; }
.cl-feature h4 { margin:0 0 .3rem 0; font-size:1rem; color:#1F3B5B; }
.cl-feature p { margin:0; color:#475569; font-size:.9rem; }
.cl-empty { border:1px dashed #CBD5E1; border-radius:12px; padding:2rem; text-align:center;
  color:#64748B; background:#FAFBFC; }
.cl-empty h4 { color:#334155; margin:0 0 .3rem 0; }
.cl-rec { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:1.1rem 1.3rem;
  margin-bottom:.9rem; border-left: 5px solid #2A9D8F; }
.cl-rec h3 { margin: 0 0 .35rem 0; font-size:1.15rem; }
.cl-rec .evidence { color:#334155; font-size:.95rem; margin:.35rem 0; }
.cl-rec .advice { color:#0F172A; font-size:.95rem; background:#F1F5F9; border-radius:8px;
  padding:.6rem .8rem; margin-top:.5rem; }
.cl-badge { display:inline-block; font-size:.72rem; font-weight:600; padding:.15rem .55rem;
  border-radius:999px; background:#E6F4F1; color:#1E7268; margin-left:.4rem; vertical-align:middle; }
.cl-badge.mod { background:#FDF3DC; color:#8A6514; }
.cl-badge.weak { background:#EEF2F7; color:#475569; }
.cl-pill { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; color:#FFFFFF; }
.cl-board { display:flex; justify-content:center; }
.cl-board svg { max-width: 100%; height: auto; border-radius: 6px; }
.cl-kv { display:grid; grid-template-columns: 150px 1fr; row-gap:.35rem; font-size:.93rem; }
.cl-kv .k { color:#64748B; } .cl-kv .v { color:#0F172A; font-weight:500; }
.cl-footer { color:#94A3B8; font-size:.8rem; margin-top:1.2rem; }
section[data-testid="stSidebar"] .cl-side-brand { font-weight:700; font-size:1.25rem; color:#1F3B5B; }
section[data-testid="stSidebar"] .cl-side-brand span { color:#2A9D8F; }
</style>
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="cl-page-title">{esc(title)}</div>'
        + (f'<div class="cl-page-sub">{esc(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def stat_cards(cards: Iterable[tuple], cols: int | None = None) -> None:
    """``cards``: iterable of (label, value, sub[, style]) where style is '', 'accent' or 'warn'."""
    parts = []
    for card in cards:
        label, value, sub = card[0], card[1], card[2]
        style = card[3] if len(card) > 3 else ""
        parts.append(
            f'<div class="cl-card {esc(style)}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
        )
    grid_style = f' style="grid-template-columns: repeat({cols}, 1fr);"' if cols else ""
    st.markdown(f'<div class="cl-grid"{grid_style}>{"".join(parts)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="cl-empty"><h4>{esc(title)}</h4><div>{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def class_pill(classification: str) -> str:
    color = CLASS_COLORS.get(classification, SLATE)
    return f'<span class="cl-pill" style="background:{color}">{esc(classification.title())}</span>'


def confidence_badge(label: str) -> str:
    cls = "" if label.startswith("Strong") else ("mod" if label.startswith("Moderate") else "weak")
    return f'<span class="cl-badge {cls}">{esc(label)}</span>'


def board_svg(fen: str, played_uci: str | None, best_uci: str | None, orientation: str = "white",
              size: int = 440) -> str:
    board = chess.Board(fen)
    arrows = []
    if best_uci and best_uci != played_uci:
        mv = chess.Move.from_uci(best_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color="#2A9D8Fcc"))
    if played_uci:
        mv = chess.Move.from_uci(played_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color="#E76F51d0"))
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board,
        orientation=chess.WHITE if orientation == "white" else chess.BLACK,
        arrows=arrows,
        check=check,
        size=size,
        coordinates=True,
        colors={
            "square light": "#EEF2F6",
            "square dark": "#7F97B2",
            "margin": "#1F3B5B",
            "coord": "#E2E8F0",
        },
    )


def render_board(svg: str) -> None:
    st.markdown(f'<div class="cl-board">{svg}</div>', unsafe_allow_html=True)


def fmt_pct(x, digits: int = 1) -> str:
    try:
        if x is None or x != x:
            return "-"
        return f"{100 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_num(x, digits: int = 0) -> str:
    try:
        if x is None or x != x:
            return "-"
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_eval(cp) -> str:
    if cp is None or cp != cp:
        return "-"
    cp = int(cp)
    if cp >= 1000:
        return "+10.0 (winning / mate)"
    if cp <= -1000:
        return "-10.0 (losing / mate)"
    return f"{cp / 100:+.2f}"
