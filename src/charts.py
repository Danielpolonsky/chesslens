"""Plotly figure builders with a consistent, restrained visual style.

Each chart shows one kind of measure on one axis. Titles say whether higher or lower is
better, and sample sizes are printed where they change how a bar should be read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .ui import CLASS_COLORS, GREEN, INK, ORANGE, OUTCOME_COLORS, RED, SLATE

FONT = "Inter, system-ui, Segoe UI, sans-serif"
NEUTRAL_BAR = "#525252"


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=22, t=(82 if legend else 52) if title else 30, b=16),
        font=dict(family=FONT, size=12, color="#737373"),
        title=dict(text=title, font=dict(size=14, color="#171717", weight=600), x=0, xref="container",
                   xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=16, l=18))
        if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    traceorder="normal", font=dict(color="#525252")),
        hoverlabel=dict(font_family=FONT, bgcolor="#FFFFFF", bordercolor="#E5E5E5",
                        font_color="#171717"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E5E5", ticks="", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#F2F2F2", zeroline=False, title_font=dict(size=12))
    return fig


def _n(n) -> str:
    return f"n={int(n):,}"


# ---------------------------------------------------------------------------
# Results and trends
# ---------------------------------------------------------------------------


def color_results(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in (("win_rate", "Win"), ("draw_rate", "Draw"), ("loss_rate", "Loss")):
        fig.add_bar(
            y=[f"{c.title()} ({int(n)} games)" for c, n in zip(df["color"], df["games"])],
            x=df[col], name=name, orientation="h", marker_color=OUTCOME_COLORS[name],
            hovertemplate="%{y}: %{x:.0%}<extra>" + name + "</extra>",
            text=[f"{v:.0%}" if v >= 0.06 else "" for v in df[col]], textposition="inside",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return style(fig, height=250, title="Results by colour (share of games)")


TIME_CLASS_COLORS = {"bullet": ORANGE, "blitz": GREEN, "rapid": INK,
                     "classical": "#6366F1", "daily": "#A3A3A3", "unknown": "#D4D4D4"}


def rating_trend(games: pd.DataFrame) -> go.Figure:
    """One line per time control - platforms keep separate ratings for each."""
    df = games.dropna(subset=["player_elo", "played_at"]).sort_values("played_at")
    fig = go.Figure()
    for tc, sub in df.groupby("time_class", sort=False):
        if len(sub) < 2:
            continue
        fig.add_scatter(
            x=sub["played_at"], y=sub["player_elo"], mode="lines+markers",
            name=f"{tc.title()} ({len(sub)} games)",
            line=dict(color=TIME_CLASS_COLORS.get(tc, SLATE), width=1.5),
            marker=dict(color=TIME_CLASS_COLORS.get(tc, SLATE), size=5),
            customdata=np.stack([sub["opponent"], sub["opponent_elo"].fillna(0), sub["outcome"]], axis=1),
            hovertemplate="%{x|%b %d, %Y}<br>Rating %{y}<br>vs %{customdata[0]} (%{customdata[1]})"
                          " - %{customdata[2]}<extra>" + tc.title() + "</extra>",
        )
    fig.update_yaxes(title="Rating before the game")
    return style(fig, height=320, title="Rating over time, by time control")


def score_by_period(df: pd.DataFrame, granularity: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["period"], y=df["avg_score"], marker_color=NEUTRAL_BAR,
        text=[f"{s:.0%}<br>{int(g)} games" for s, g in zip(df["avg_score"], df["games"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: score %{y:.0%}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], title="Score")
    return style(fig, height=320, title=f"Score by {granularity} (higher is better)", legend=False)


def rolling_acpl(games: pd.DataFrame, window: int = 10) -> go.Figure:
    df = games.dropna(subset=["acpl"]).sort_values("played_at").reset_index(drop=True)
    df["game_no"] = np.arange(1, len(df) + 1)
    df["rolling"] = df["acpl"].rolling(window, min_periods=max(3, window // 2)).mean()
    fig = go.Figure()
    fig.add_scatter(x=df["game_no"], y=df["acpl"], mode="markers", name="One game",
                    marker=dict(color="#C7C7C7", size=6),
                    customdata=np.stack([df["opponent"], df["outcome"]], axis=1),
                    hovertemplate="Game %{x}: %{y:.0f} cp per move<br>vs %{customdata[0]} "
                                  "(%{customdata[1]})<extra></extra>")
    fig.add_scatter(x=df["game_no"], y=df["rolling"], mode="lines", name=f"{window}-game average",
                    line=dict(color=INK, width=2))
    fig.update_xaxes(title="Game number (oldest first)")
    fig.update_yaxes(title="Average centipawn loss per move", rangemode="tozero")
    return style(fig, height=320, title="Centipawn loss over time (lower is better)")


# ---------------------------------------------------------------------------
# Error patterns (all: share of the selected player's moves; lower is better)
# ---------------------------------------------------------------------------


def phase_error_rates(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = [f"{p.title()}<br><span style='font-size:11px'>{_n(n)} moves</span>"
              for p, n in zip(df["phase"], df["moves"])]
    for col, name in (("inaccuracy_rate", "inaccuracy"), ("mistake_rate", "mistake"),
                      ("blunder_rate", "blunder")):
        fig.add_bar(x=labels, y=df[col], name=name.title(), marker_color=CLASS_COLORS[name],
                    text=[f"{v:.1%}" for v in df[col]], textposition="outside", cliponaxis=False,
                    hovertemplate="%{y:.1%} of moves<extra>" + name.title() + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Share of your moves")
    return style(fig, height=340, title="Errors by game phase (lower is better)")


def move_number_profile(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["label"], y=df["serious_rate"], marker_color=RED,
        text=[f"{v:.0%}<br>{_n(n)}" for v, n in zip(df["serious_rate"], df["moves"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="Moves %{x}: %{y:.1%} mistakes or blunders<extra></extra>",
    ))
    top = float(df["serious_rate"].max()) if len(df) else 0.3
    fig.update_yaxes(tickformat=".0%", title="Mistake + blunder rate", range=[0, top * 1.35 + 0.01])
    fig.update_xaxes(title="Move number")
    return style(fig, height=320, title="Mistakes and blunders by move number (lower is better)",
                 legend=False)


def context_bars(df: pd.DataFrame) -> go.Figure:
    order = {"losing": 0, "balanced": 1, "winning": 2}
    df = df.sort_values("context", key=lambda s: s.map(order))
    labels = {"losing": "You were worse (eval -2 or less)", "balanced": "Roughly equal",
              "winning": "You were better (eval +2 or more)"}
    fig = go.Figure(go.Bar(
        x=[f"{labels[c]}<br><span style='font-size:11px'>{_n(n)} moves</span>"
           for c, n in zip(df["context"], df["moves"])],
        y=df["serious_rate"], marker_color=RED,
        text=[f"{v:.1%}" for v in df["serious_rate"]], textposition="outside", cliponaxis=False,
    ))
    top = float(df["serious_rate"].max()) if len(df) else 0.3
    fig.update_yaxes(tickformat=".0%", title="Mistake + blunder rate", range=[0, top * 1.3 + 0.01])
    return style(fig, height=320, title="Mistakes and blunders by position before the move "
                                        "(lower is better)", legend=False)


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def opening_score_bars(df: pd.DataFrame, overall_score: float) -> go.Figure:
    d = df.sort_values("games", ascending=True)
    fig = go.Figure(go.Bar(
        y=d["opening"], x=d["avg_score"], orientation="h", marker_color=NEUTRAL_BAR,
        text=[f"{s:.0%} · {int(g)} games" for s, g in zip(d["avg_score"], d["games"])],
        textposition="outside", cliponaxis=False,
        customdata=np.stack([d["wins"], d["draws"], d["losses"]], axis=1),
        hovertemplate="%{y}<br>Score %{x:.0%} (%{customdata[0]}W / %{customdata[1]}D / "
                      "%{customdata[2]}L)<extra></extra>",
    ))
    if pd.notna(overall_score):
        fig.add_vline(x=overall_score, line_dash="dash", line_color=SLATE,
                      annotation_text=f"All games {overall_score:.0%}", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.3], title="Score (draw = half a point)")
    return style(fig, height=max(260, 34 * len(d) + 110), title="Score by opening (higher is better)",
                 legend=False)


def opening_scatter(df: pd.DataFrame, min_games: int) -> go.Figure:
    d = df[(df["games"] >= min_games) & df["acpl"].notna()]
    labelled = set(d.nlargest(6, "games")["opening"])
    fig = go.Figure(
        go.Scatter(
            x=d["acpl"], y=d["avg_score"], mode="markers+text",
            text=[o if o in labelled else "" for o in d["opening"]], hovertext=d["opening"],
            textposition="top center", textfont=dict(size=10.5),
            marker=dict(size=np.sqrt(d["games"]) * 7, color=NEUTRAL_BAR, opacity=0.55,
                        line=dict(color="white", width=1)),
            customdata=np.stack([d["games"], d["serious_rate"]], axis=1),
            hovertemplate="%{hovertext}<br>%{customdata[0]} games<br>Score %{y:.0%}<br>"
                          "Centipawn loss %{x:.0f}<br>Mistake+blunder rate %{customdata[1]:.1%}"
                          "<extra></extra>",
        )
    )
    if not d.empty:
        span = max(10.0, float(d["acpl"].max() - d["acpl"].min()))
        fig.update_xaxes(range=[d["acpl"].min() - 0.25 * span, d["acpl"].max() + 0.25 * span])
    fig.update_xaxes(title="Average centipawn loss (lower is better)")
    fig.update_yaxes(title="Score", tickformat=".0%", range=[-0.08, 1.08])
    return style(fig, height=400, title=f"Score vs centipawn loss (openings with ≥ {min_games} games; "
                                        "size = games)", legend=False)


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------


def game_eval_curve(moves: pd.DataFrame, highlight_ply: int, player_color: str) -> go.Figure:
    """Engine evaluation over a game from the player's perspective."""
    sign = np.where(moves["color"] == player_color, 1, -1)
    player_pov = moves["eval_after"] * sign
    fig = go.Figure()
    fig.add_scatter(x=moves["ply"], y=(player_pov / 100).clip(-10, 10), mode="lines",
                    line=dict(color=INK, width=2), fill="tozeroy", fillcolor="rgba(23,23,23,0.05)",
                    name="Evaluation", hovertemplate="Half-move %{x}: %{y:+.2f}<extra></extra>")
    hl = moves[moves["ply"] == highlight_ply]
    if not hl.empty:
        y = float((hl["eval_after"].iloc[0] * (1 if hl["color"].iloc[0] == player_color else -1)) / 100)
        fig.add_scatter(x=[highlight_ply], y=[max(-10, min(10, y))], mode="markers", name="This move",
                        marker=dict(color=RED, size=11, symbol="x"))
    fig.update_yaxes(title="Evaluation for you (pawns)", range=[-10.5, 10.5])
    fig.update_xaxes(title="Half-move number")
    return style(fig, height=260, legend=False)


# ---------------------------------------------------------------------------
# Machine learning (scores are model risk scores, not calibrated probabilities)
# ---------------------------------------------------------------------------


def model_comparison(models: dict) -> go.Figure:
    metrics = [("average_precision", "Average precision"), ("roc_auc", "ROC AUC")]
    palette = {"Dummy (stratified)": "#D4D4D4", "Logistic regression": SLATE, "Random forest": INK}
    fig = go.Figure()
    for name, res in models.items():
        t = res["test"]
        fig.add_bar(x=[m[1] for m in metrics], y=[t[m[0]] for m in metrics], name=name,
                    marker_color=palette.get(name, SLATE),
                    text=[f"{t[m[0]]:.3f}" for m in metrics], textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05], title="Higher is better")
    return style(fig, height=340, title="Ranking quality on the held-out test games")


def pr_curves(curves: dict, base_rate: float) -> go.Figure:
    fig = go.Figure()
    colors = {"Logistic regression": SLATE, "Random forest": INK}
    for name, c in curves.items():
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=name,
                        line=dict(color=colors.get(name, SLATE), width=2))
    fig.add_hline(y=base_rate, line_dash="dash", line_color=SLATE,
                  annotation_text=f"Base rate {base_rate:.1%}", annotation_position="top right")
    fig.update_xaxes(title="Recall (share of real errors flagged)", range=[0, 1])
    fig.update_yaxes(title="Precision (share of flags that were errors)", range=[0, 1])
    return style(fig, height=340, title="Precision-recall curves (held-out test games)")


def calibration_bars(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(x=df["decile"], y=df["observed_rate"], name="Observed error rate", marker_color=RED)
    fig.add_scatter(x=df["decile"], y=df["mean_predicted"], name="Mean model score",
                    mode="lines+markers", line=dict(color=INK))
    fig.update_xaxes(title="Model-score decile (1 = lowest risk)", dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return style(fig, height=340, title="Observed error rate vs model score (held-out test games)")


def importance_bars(rows: list[dict], value_key: str, title: str, top: int = 12) -> go.Figure:
    df = pd.DataFrame(rows).head(top).iloc[::-1]
    colors = [INK if v >= 0 else SLATE for v in df[value_key]]
    fig = go.Figure(go.Bar(y=df["feature"], x=df[value_key], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=df["std"]) if "std" in df else None))
    return style(fig, height=max(300, 26 * len(df) + 70), title=title, legend=False)


def risk_bucket_bars(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["observed_rate"], name="Observed error rate", marker_color=RED,
                text=[f"{v:.0%}<br>{_n(n)}" for v, n in zip(df["observed_rate"], df["positions"])],
                textposition="outside", cliponaxis=False)
    fig.add_scatter(x=df["bucket"], y=df["mean_predicted"], name="Mean model score",
                    mode="lines+markers", line=dict(color=INK))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="Model-score group (each holds one fifth of the moves)")
    return style(fig, height=340, title=title)
