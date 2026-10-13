"""Streamlit presentation helpers: theme CSS, stat cards, empty states, chessboards."""

from __future__ import annotations

import html
from typing import Iterable

import chess
import chess.svg
import streamlit as st

INK = "#171717"
MUTED = "#737373"
GREEN = "#3DA673"
AMBER = "#F5C451"
ORANGE = "#F0915A"
RED = "#E5605F"
SLATE = "#A3A3A3"
LIGHT = "#E5E5E5"

CLASS_COLORS = {"good": "#D4D4D4", "inaccuracy": AMBER, "mistake": ORANGE, "blunder": RED}
OUTCOME_COLORS = {"Win": GREEN, "Draw": "#C7C7C7", "Loss": RED, "Unfinished": LIGHT}
COLOR_COLORS = {"white": LIGHT, "black": INK}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}
.block-container { padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1180px; }
h1, h2, h3, h4, h5 { color: #171717; letter-spacing: -0.011em; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
h4 { font-size: 1.02rem !important; font-weight: 600 !important; margin-top: .6rem; }
hr { margin: 1rem 0 !important; }

/* Page header */
.cl-page-title { font-size: 1.6rem; font-weight: 600; color: #171717; letter-spacing: -0.02em;
  margin: 0 0 .15rem 0; }
.cl-page-sub { color: #737373; font-size: .95rem; margin-bottom: 1.4rem; }
.cl-section { font-size: 1.02rem; font-weight: 600; color: #171717; margin: 1.6rem 0 .6rem 0; }

/* Stat cards */
.cl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem;
  margin-bottom: .9rem; }
.cl-card { background: #FFFFFF; border: 1px solid #EBEBEB; border-radius: 12px; padding: .9rem 1rem; }
.cl-card .label { color: #737373; font-size: .86rem; font-weight: 500; display: flex;
  align-items: center; gap: .4rem; }
.cl-card .value { color: #171717; font-size: 1.55rem; font-weight: 600; margin-top: .3rem;
  line-height: 1.2; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.cl-card .sub { color: #A3A3A3; font-size: .82rem; margin-top: .25rem; }
.cl-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.cl-dot.accent { background: #2F9E6B; }
.cl-dot.warn { background: #E5484D; }

/* Status badges and move-class pills */
.cl-badge { display: inline-flex; align-items: center; gap: .3rem; font-size: .76rem; font-weight: 500;
  padding: .1rem .5rem; border-radius: 6px; border: 1px solid #ABEFC6; background: #ECFDF3;
  color: #067647; vertical-align: middle; line-height: 1.5; white-space: nowrap; }
.cl-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.cl-badge.mod { border-color: #FEDF89; background: #FFFAEB; color: #B54708; }
.cl-badge.weak { border-color: #E5E5E5; background: #FAFAFA; color: #525252; }
.cl-badge.red { border-color: #FECDCA; background: #FEF3F2; color: #B42318; }
.cl-badge.orange { border-color: #F9DBAF; background: #FEF6EE; color: #B93815; }
.cl-badge.amber { border-color: #FEDF89; background: #FFFAEB; color: #B54708; }

/* Home */
.cl-greeting { font-size: 1.6rem; font-weight: 600; color: #171717; letter-spacing: -0.02em; margin: 0; }
.cl-lead { color: #737373; font-size: .98rem; margin: .3rem 0 1.6rem 0; max-width: 720px; }
.cl-preview { border: 1px solid #EBEBEB; border-radius: 12px; padding: 10px; background: #FAFAFA; }
.cl-preview svg { width: 100%; height: auto; display: block; border-radius: 6px; }
.cl-meta-title { display: flex; align-items: center; gap: .5rem; font-weight: 600; font-size: 1rem;
  color: #171717; }
.cl-meta { color: #737373; font-size: .9rem; margin-top: .3rem; }
.cl-meta b { color: #171717; font-weight: 500; }
.cl-label { color: #737373; font-size: .86rem; margin: 1rem 0 .15rem 0; }
.cl-value { color: #171717; font-size: .92rem; font-weight: 500; }
.cl-steps { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid #EBEBEB;
  border-radius: 12px; overflow: hidden; margin: .4rem 0 0 0; }
.cl-step { padding: .85rem 1rem; border-right: 1px solid #EBEBEB; }
.cl-step:last-child { border-right: none; }
.cl-step .n { color: #A3A3A3; font-size: .78rem; font-weight: 500; }
.cl-step .t { color: #171717; font-size: .92rem; font-weight: 600; margin: .15rem 0 .2rem 0; }
.cl-step .d { color: #737373; font-size: .84rem; line-height: 1.45; }

/* Recommendations */
.cl-rec-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem; }
.cl-rec-num { width: 24px; height: 24px; border-radius: 6px; background: #F5F5F5; color: #525252;
  font-size: .8rem; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; }
.cl-rec-title { font-size: 1.02rem; font-weight: 600; color: #171717; }
.cl-rec .evidence { color: #525252; font-size: .92rem; line-height: 1.55; }
.cl-rec .advice { color: #171717; font-size: .92rem; background: #FAFAFA; border: 1px solid #F0F0F0;
  border-radius: 8px; padding: .6rem .8rem; margin-top: .7rem; line-height: 1.55; }
.cl-rec .examples { color: #525252; font-size: .88rem; margin-top: .7rem; }
.cl-rec .examples ul { margin: .25rem 0 0 0; padding-left: 1.1rem; }

/* Empty state, board, key/value list */
.cl-empty { border: 1px solid #EBEBEB; border-radius: 12px; padding: 2.2rem; text-align: center;
  color: #737373; background: #FAFAFA; font-size: .92rem; }
.cl-empty h4 { color: #171717; margin: 0 0 .3rem 0; font-size: 1rem; }
.cl-board { display: flex; justify-content: center; border: 1px solid #EBEBEB; border-radius: 12px;
  padding: 10px; background: #FFFFFF; }
.cl-board svg { max-width: 100%; height: auto; border-radius: 4px; }
.cl-kv { display: grid; grid-template-columns: 150px 1fr; font-size: .9rem; border: 1px solid #EBEBEB;
  border-radius: 12px; overflow: hidden; }
.cl-kv > div { padding: .5rem .85rem; border-bottom: 1px solid #F2F2F2; }
.cl-kv > div:nth-last-child(-n+2) { border-bottom: none; }
.cl-kv .k { color: #737373; background: #FAFAFA; }
.cl-kv .v { color: #171717; font-weight: 500; }
.cl-footer { color: #A3A3A3; font-size: .8rem; margin-top: 1.4rem; line-height: 1.5; }

/* Charts sit in hairline cards, like the tables */
[data-testid="stPlotlyChart"] { border: 1px solid #EBEBEB; border-radius: 12px; overflow: hidden;
  background: #FFFFFF; box-sizing: border-box; }

/* Sidebar */
section[data-testid="stSidebar"] .cl-side-brand { display: flex; align-items: center; gap: .55rem;
  font-weight: 600; font-size: 1.05rem; color: #171717; letter-spacing: -0.01em; }
.cl-logo { width: 26px; height: 26px; border-radius: 7px; background: #171717; color: #FFFFFF;
  display: inline-flex; align-items: center; justify-content: center; font-size: 16px; line-height: 1; }
section[data-testid="stSidebar"] h5 { font-size: .8rem !important; font-weight: 500 !important;
  color: #737373 !important; margin: .8rem 0 0 0; }
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


def section(title: str) -> None:
    st.markdown(f'<div class="cl-section">{esc(title)}</div>', unsafe_allow_html=True)


def stat_cards(cards: Iterable[tuple], cols: int | None = None) -> None:
    """``cards``: iterable of (label, value, sub[, style]) where style is '', 'accent' or 'warn'."""
    parts = []
    for card in cards:
        label, value, sub = card[0], card[1], card[2]
        style = card[3] if len(card) > 3 else ""
        dot = f'<span class="cl-dot {esc(style)}"></span>' if style else ""
        parts.append(
            f'<div class="cl-card"><div class="label">{dot}{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
        )
    grid_style = f' style="grid-template-columns: repeat({cols}, 1fr);"' if cols else ""
    st.markdown(f'<div class="cl-grid"{grid_style}>{"".join(parts)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="cl-empty"><h4>{esc(title)}</h4><div>{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "") -> str:
    return f'<span class="cl-badge {esc(tone)}">{esc(text)}</span>'


CLASS_TONES = {"good": "", "inaccuracy": "amber", "mistake": "orange", "blunder": "red"}


def class_pill(classification: str) -> str:
    return badge(classification.title(), CLASS_TONES.get(classification, "weak"))


def confidence_badge(label: str) -> str:
    tone = "" if label.startswith("Strong") else ("mod" if label.startswith("Moderate") else "weak")
    return badge(label, tone)


def board_svg(fen: str, played_uci: str | None, best_uci: str | None, orientation: str = "white",
              size: int = 440, coordinates: bool = True) -> str:
    board = chess.Board(fen)
    arrows = []
    if best_uci and best_uci != played_uci:
        mv = chess.Move.from_uci(best_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color="#2F9E6Bcc"))
    if played_uci:
        mv = chess.Move.from_uci(played_uci)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color="#E5484Dcc"))
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board,
        orientation=chess.WHITE if orientation == "white" else chess.BLACK,
        arrows=arrows,
        check=check,
        size=size,
        coordinates=coordinates,
        colors={
            "square light": "#F0F0EE",
            "square dark": "#B9BCB8",
            "margin": "#FFFFFF",
            "coord": "#A3A3A3",
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
