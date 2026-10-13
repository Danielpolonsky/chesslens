"""Plotly figure builders with a consistent, restrained visual style."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .ui import CLASS_COLORS, COLOR_COLORS, NAVY, OUTCOME_COLORS, RED, SLATE, TEAL

FONT = "Inter, system-ui, Segoe UI, sans-serif"


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 18, b=10),
        font=dict(family=FONT, size=12.5, color="#334155"),
        title=dict(text=title, font=dict(size=15, color="#0F172A"), x=0.01) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        hoverlabel=dict(font_family=FONT),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F6", zeroline=False)
    return fig


def results_donut(wins: int, draws: int, losses: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Win", "Draw", "Loss"], values=[wins, draws, losses], hole=0.62,
            marker=dict(colors=[OUTCOME_COLORS[k] for k in ("Win", "Draw", "Loss")]),
            sort=False, textinfo="percent", hovertemplate="%{label}: %{value} games<extra></extra>",
        )
    )
    total = wins + draws + losses
    fig.add_annotation(text=f"<b>{total}</b><br>games", showarrow=False, font=dict(size=15))
    return style(fig, height=300, title="Results")


def color_results(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in (("win_rate", "Win"), ("draw_rate", "Draw"), ("loss_rate", "Loss")):
        fig.add_bar(
            y=[f"{c.title()} ({int(n)} g)" for c, n in zip(df["color"], df["games"])],
            x=df[col], name=name, orientation="h", marker_color=OUTCOME_COLORS[name],
            hovertemplate="%{y}: %{x:.0%}<extra>" + name + "</extra>",
            text=[f"{v:.0%}" if v >= 0.06 else "" for v in df[col]], textposition="inside",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return style(fig, height=300, title="White vs Black")


def rating_trend(games: pd.DataFrame) -> go.Figure:
    df = games.dropna(subset=["player_elo", "played_at"]).sort_values("played_at")
    fig = go.Figure()
    fig.add_scatter(x=df["played_at"], y=df["player_elo"], mode="lines", line=dict(color=NAVY, width=2),
                    name="Rating", hoverinfo="skip")
    for outcome, color in OUTCOME_COLORS.items():
        sub = df[df["outcome"] == outcome]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub["played_at"], y=sub["player_elo"], mode="markers", name=outcome,
            marker=dict(color=color, size=7, line=dict(color="white", width=1)),
            customdata=np.stack([sub["opponent"], sub["opponent_elo"].fillna(0)], axis=1),
            hovertemplate="%{x|%b %d, %Y}<br>Rating %{y}<br>vs %{customdata[0]} (%{customdata[1]})"
                          "<extra>" + outcome + "</extra>",
        )
    return style(fig, height=320, title="Rating over time")


def trend_lines(df: pd.DataFrame, granularity: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["period"], y=df["games"], name="Games", marker_color="#E2E8F0", yaxis="y2",
                hovertemplate="%{x}: %{y} games<extra></extra>")
    fig.add_scatter(x=df["period"], y=df["acpl"], name="Avg centipawn loss", mode="lines+markers",
                    line=dict(color=NAVY, width=2.5))
    fig.add_scatter(x=df["period"], y=df["avg_score"] * 100, name="Score %", mode="lines+markers",
                    line=dict(color=TEAL, width=2.5, dash="dot"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Games"),
                      yaxis=dict(title="ACPL / Score %"))
    return style(fig, height=340, title=f"Performance by {granularity}")


def rolling_acpl(games: pd.DataFrame, window: int = 10) -> go.Figure:
    df = games.dropna(subset=["acpl"]).sort_values("played_at").reset_index(drop=True)
    df["game_no"] = np.arange(1, len(df) + 1)
    df["rolling"] = df["acpl"].rolling(window, min_periods=max(3, window // 2)).mean()
    fig = go.Figure()
    fig.add_scatter(x=df["game_no"], y=df["acpl"], mode="markers", name="Game ACPL",
                    marker=dict(color=[OUTCOME_COLORS.get(o, SLATE) for o in df["outcome"]], size=6,
                                opacity=0.65),
                    customdata=np.stack([df["opponent"], df["outcome"]], axis=1),
                    hovertemplate="Game %{x}: ACPL %{y:.0f}<br>vs %{customdata[0]} (%{customdata[1]})"
                                  "<extra></extra>")
    fig.add_scatter(x=df["game_no"], y=df["rolling"], mode="lines", name=f"{window}-game average",
                    line=dict(color=NAVY, width=3))
    fig.update_xaxes(title="Game # (chronological)")
    fig.update_yaxes(title="Average centipawn loss")
    return style(fig, height=340, title="Accuracy trend (lower is better)")


def phase_error_rates(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = [f"{p.title()}<br><span style='font-size:11px'>n={int(n):,} moves</span>"
              for p, n in zip(df["phase"], df["moves"])]
    for col, name in (("inaccuracy_rate", "inaccuracy"), ("mistake_rate", "mistake"),
                      ("blunder_rate", "blunder")):
        fig.add_bar(x=labels, y=df[col], name=name.title(), marker_color=CLASS_COLORS[name],
                    text=[f"{v:.1%}" for v in df[col]], textposition="outside",
                    hovertemplate="%{y:.1%} of moves<extra>" + name.title() + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Share of evaluated moves")
    return style(fig, height=360, title="Error rate per evaluated move, by game phase")


def cpl_histogram(moves: pd.DataFrame, cap: int = 600) -> go.Figure:
    df = moves.copy()
    df["cpl_capped"] = df["cpl"].clip(upper=cap)
    fig = go.Figure()
    for cls in ("good", "inaccuracy", "mistake", "blunder"):
        sub = df[df["classification"] == cls]
        fig.add_histogram(x=sub["cpl_capped"], name=cls.title(), marker_color=CLASS_COLORS[cls],
                          xbins=dict(start=0, end=cap + 20, size=20))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=f"Centipawn loss per move (values above {cap} shown at {cap})")
    fig.update_yaxes(title="Moves", type="log")
    return style(fig, height=340, title="Distribution of centipawn loss")


def move_number_profile(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["label"], y=df["moves"], name="Moves analyzed", marker_color="#E2E8F0", yaxis="y2")
    fig.add_scatter(x=df["label"], y=df["serious_rate"], name="Mistake + blunder rate",
                    mode="lines+markers", line=dict(color=RED, width=2.5))
    fig.update_layout(yaxis=dict(tickformat=".0%", title="Serious-error rate"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Moves"))
    fig.update_xaxes(title="Move number")
    return style(fig, height=320, title="When do mistakes happen?")


def time_class_bars(df: pd.DataFrame) -> go.Figure:
    labels = [f"{t.title()} ({int(g)} g)" for t, g in zip(df["time_class"], df["games"])]
    fig = go.Figure()
    fig.add_bar(x=labels, y=df["avg_score"], name="Score", marker_color=TEAL,
                text=[f"{v:.0%}" for v in df["avg_score"]], textposition="outside")
    fig.add_bar(x=labels, y=df["serious_rate"], name="Mistake + blunder rate", marker_color=RED,
                text=[f"{v:.0%}" if pd.notna(v) else "" for v in df["serious_rate"]], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, height=320, title="Performance by time control")


def context_bars(df: pd.DataFrame) -> go.Figure:
    order = {"losing": 0, "balanced": 1, "winning": 2}
    df = df.sort_values("context", key=lambda s: s.map(order))
    labels = {"losing": "Clearly worse (<= -2)", "balanced": "Roughly equal", "winning": "Clearly better (>= +2)"}
    fig = go.Figure(
        go.Bar(
            x=[f"{labels[c]}<br><span style='font-size:11px'>n={int(n):,}</span>"
               for c, n in zip(df["context"], df["moves"])],
            y=df["serious_rate"], marker_color=[RED, SLATE, TEAL][: len(df)],
            text=[f"{v:.1%}" for v in df["serious_rate"]], textposition="outside",
        )
    )
    fig.update_yaxes(tickformat=".0%", title="Mistake + blunder rate")
    return style(fig, height=320, title="Error rate by position evaluation before the move", legend=False)


def opening_bars(df: pd.DataFrame, top: int = 12) -> go.Figure:
    d = df.sort_values("games", ascending=False).head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            y=d["opening"], x=d["games"], orientation="h",
            marker=dict(color=d["avg_score"], colorscale=[[0, RED], [0.5, "#E2E8F0"], [1, TEAL]],
                        cmin=0, cmax=1, colorbar=dict(title="Score", tickformat=".0%", thickness=12)),
            customdata=np.stack([d["avg_score"], d["wins"], d["draws"], d["losses"]], axis=1),
            hovertemplate="%{y}<br>%{x} games, score %{customdata[0]:.0%}<br>"
                          "%{customdata[1]}W / %{customdata[2]}D / %{customdata[3]}L<extra></extra>",
            text=[f"{s:.0%}" for s in d["avg_score"]], textposition="outside",
        )
    )
    fig.update_xaxes(title="Games played")
    return style(fig, height=max(320, 30 * len(d) + 80), title="Most-played openings (colour = score)",
                 legend=False)


def opening_scatter(df: pd.DataFrame, min_games: int) -> go.Figure:
    d = df[(df["games"] >= min_games) & df["acpl"].notna()]
    fig = go.Figure(
        go.Scatter(
            x=d["acpl"], y=d["avg_score"], mode="markers+text", text=d["opening"],
            textposition="top center", textfont=dict(size=10.5),
            marker=dict(size=np.sqrt(d["games"]) * 7, color=d["serious_rate"],
                        colorscale=[[0, TEAL], [1, RED]], showscale=True,
                        colorbar=dict(title="Err. rate", tickformat=".0%", thickness=12),
                        line=dict(color="white", width=1)),
            customdata=np.stack([d["games"], d["serious_rate"]], axis=1),
            hovertemplate="%{text}<br>%{customdata[0]} games<br>Score %{y:.0%}<br>ACPL %{x:.0f}"
                          "<br>Mistake+blunder rate %{customdata[1]:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Average centipawn loss (lower is better)")
    fig.update_yaxes(title="Score", tickformat=".0%")
    return style(fig, height=420, title=f"Score vs accuracy by opening (>= {min_games} games; size = games)",
                 legend=False)


def game_eval_curve(moves: pd.DataFrame, highlight_ply: int, player_color: str) -> go.Figure:
    """Engine evaluation over a game from the player's perspective."""
    sign = np.where(moves["color"] == player_color, 1, -1)
    player_pov = moves["eval_after"] * sign
    fig = go.Figure()
    fig.add_scatter(x=moves["ply"], y=(player_pov / 100).clip(-10, 10), mode="lines",
                    line=dict(color=NAVY, width=2), fill="tozeroy", fillcolor="rgba(42,157,143,0.12)",
                    name="Evaluation", hovertemplate="Ply %{x}: %{y:+.2f}<extra></extra>")
    hl = moves[moves["ply"] == highlight_ply]
    if not hl.empty:
        y = float((hl["eval_after"].iloc[0] * (1 if hl["color"].iloc[0] == player_color else -1)) / 100)
        fig.add_scatter(x=[highlight_ply], y=[max(-10, min(10, y))], mode="markers", name="This move",
                        marker=dict(color=RED, size=12, symbol="x"))
    fig.update_yaxes(title="Eval (pawns, your view)", range=[-10.5, 10.5])
    fig.update_xaxes(title="Ply")
    return style(fig, height=260, legend=False)


def model_comparison(models: dict) -> go.Figure:
    metrics = [("average_precision", "Avg precision"), ("roc_auc", "ROC AUC"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1")]
    palette = {"Dummy (stratified)": "#CBD5E1", "Logistic regression": NAVY, "Random forest": TEAL}
    fig = go.Figure()
    for name, res in models.items():
        t = res["test"]
        fig.add_bar(x=[m[1] for m in metrics], y=[t[m[0]] for m in metrics], name=name,
                    marker_color=palette.get(name, SLATE),
                    text=[f"{t[m[0]]:.2f}" for m in metrics], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05])
    return style(fig, height=360, title="Held-out test performance (unseen games)")


def pr_curves(curves: dict, base_rate: float) -> go.Figure:
    fig = go.Figure()
    colors = {"Logistic regression": NAVY, "Random forest": TEAL}
    for name, c in curves.items():
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=name,
                        line=dict(color=colors.get(name, SLATE), width=2.5))
    fig.add_hline(y=base_rate, line_dash="dash", line_color=SLATE,
                  annotation_text=f"Base rate {base_rate:.1%}", annotation_position="top right")
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1])
    return style(fig, height=360, title="Precision-recall curves (test set)")


def calibration_bars(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(x=df["decile"], y=df["observed_rate"], name="Observed error rate", marker_color=TEAL)
    fig.add_scatter(x=df["decile"], y=df["mean_predicted"], name="Mean predicted risk",
                    mode="lines+markers", line=dict(color=NAVY))
    fig.update_xaxes(title="Predicted-risk decile (1 = lowest)", dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return style(fig, height=340, title="Observed serious-error rate by predicted-risk decile")


def importance_bars(rows: list[dict], value_key: str, title: str, top: int = 12) -> go.Figure:
    df = pd.DataFrame(rows).head(top).iloc[::-1]
    colors = [TEAL if v >= 0 else RED for v in df[value_key]]
    fig = go.Figure(go.Bar(y=df["feature"], x=df[value_key], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=df["std"]) if "std" in df else None))
    return style(fig, height=max(300, 26 * len(df) + 70), title=title, legend=False)


def risk_bucket_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["observed_rate"], name="Observed rate", marker_color=TEAL,
                text=[f"{v:.0%}<br>n={n}" for v, n in zip(df["observed_rate"], df["positions"])],
                textposition="outside")
    fig.add_scatter(x=df["bucket"], y=df["mean_predicted"], name="Mean predicted risk",
                    mode="lines+markers", line=dict(color=NAVY))
    fig.update_yaxes(tickformat=".0%")
    return style(fig, height=340, title="Your positions: predicted risk vs actual serious-error rate")
