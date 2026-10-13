"""ChessLens - AI-powered chess analytics (Streamlit entry point).

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src import analytics as an
from src import charts, config, ingest, ml, recommendations, reporting, ui
from src import database as db
from src.engine_analysis import find_stockfish
from src.features import build_ml_frame
from src.pipeline import AnalysisJob, store_report

st.set_page_config(
    page_title="ChessLens - Chess Analytics",
    page_icon="♞",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

SESSION_DIR = Path(tempfile.gettempdir()) / "chesslens_sessions"
SESSION_MAX_AGE_S = 12 * 3600
DEMO_PLAYER = "sani5057"
DEMO_LABEL = "Demo dataset - public Lichess games, January 2013 (CC0)"
USER_LABEL = "Your imported games (private to this browser session)"
ss = st.session_state
CTX: dict = {}


# ---------------------------------------------------------------------------
# Session & data context
# ---------------------------------------------------------------------------


def init_state() -> None:
    ss.setdefault("dataset", None)  # None | "demo" | "user"
    ss.setdefault("user_db", None)
    ss.setdefault("user_player", None)
    ss.setdefault("job", None)
    ss.setdefault("flash", None)
    ss.setdefault("explorer_move_id", None)
    ss.setdefault("import_messages", [])


@st.cache_resource(ttl=3600)
def cleanup_old_sessions() -> None:
    """Remove other sessions' stale databases (runs at most hourly per server process)."""
    if not SESSION_DIR.exists():
        return
    now = time.time()
    for f in SESSION_DIR.glob("*.db*"):
        try:
            if now - f.stat().st_mtime > SESSION_MAX_AGE_S:
                f.unlink()
        except OSError:
            pass


def user_db_path() -> Path:
    if not ss.user_db:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        ss.user_db = str(SESSION_DIR / f"{uuid.uuid4().hex}.db")
    return Path(ss.user_db)


def has_user_data() -> bool:
    return bool(ss.user_db) and Path(ss.user_db).exists()


def open_conn():
    if ss.dataset == "demo" and config.DEMO_DB_PATH.exists():
        return db.connect(config.DEMO_DB_PATH, read_only=True)
    if ss.dataset == "user" and has_user_data():
        return db.connect(user_db_path())
    return None


def dataset_label() -> str:
    return DEMO_LABEL if ss.dataset == "demo" else USER_LABEL


def go_demo() -> None:
    ss.dataset = "demo"


# ---------------------------------------------------------------------------
# Sidebar: dataset, player, filters, analysis progress
# ---------------------------------------------------------------------------


@st.fragment(run_every=1.0)
def job_progress() -> None:
    job: AnalysisJob | None = ss.get("job")
    if job is None:
        return
    if job.finished:
        ss.job = None
        if job.error:
            ss.flash = ("error", f"Analysis stopped: {job.error}")
        else:
            s = job.stats
            ss.flash = (
                "success",
                f"Analyzed {s.get('games', 0)} games ({s.get('positions', 0):,} positions, "
                f"{s.get('cache_hits', 0):,} served from cache) in {s.get('seconds', 0)} s.",
            )
        st.rerun()
    st.markdown("**Stockfish analysis running**")
    st.progress(job.fraction, text=f"Game {min(job.games_done + 1, job.total)} of {job.total}")
    elapsed = time.time() - job.started_at
    if job.fraction > 0.02:
        remaining = elapsed / job.fraction * (1 - job.fraction)
        st.caption(f"Elapsed {elapsed:.0f}s - about {remaining:.0f}s remaining. "
                   "You can keep exploring while it runs.")
    if st.button("Cancel analysis", key="cancel_job"):
        job.cancel()


def sidebar(conn) -> tuple[str | None, an.Filters]:
    with st.sidebar:
        st.markdown('<div class="cl-side-brand">Chess<span>Lens</span></div>', unsafe_allow_html=True)
        st.caption("Built for Polonsky Academy")
        if ss.job is not None:
            job_progress()
            st.divider()

        options = ["demo"] + (["user"] if has_user_data() else [])
        if ss.dataset is None:
            st.info("Load the demo or import your games from the Home page.")
            return None, an.Filters()
        labels = {"demo": "Demo dataset", "user": "My imported games"}
        choice = st.radio("Dataset", options, index=options.index(ss.dataset),
                          format_func=labels.get, horizontal=True)
        if choice != ss.dataset:
            ss.dataset = choice
            st.rerun()
        if conn is None:
            return None, an.Filters()

        players = db.list_players(conn)
        if players.empty:
            st.info("No games in this dataset yet.")
            return None, an.Filters()
        names = players["username"].tolist()
        preferred = DEMO_PLAYER if ss.dataset == "demo" else ss.user_player
        index = next((i for i, n in enumerate(names) if preferred and n.lower() == preferred.lower()), 0)
        player = st.selectbox(
            "Player", names, index=index, key=f"player_{ss.dataset}",
            format_func=lambda n: f"{n} ({int(players.set_index('username').loc[n, 'games'])} games)",
        )
        if ss.dataset == "user":
            ss.user_player = player

        st.markdown("##### Filters")
        opts = an.filter_options(conn, player)
        k = f"{ss.dataset}_{player}"
        color = st.segmented_control("Color", ["All", "White", "Black"], default="All", key=f"color_{k}")
        d_from = d_to = None
        if opts["date_min"] and opts["date_max"] and opts["date_min"] < opts["date_max"]:
            rng = st.date_input("Date range", value=(opts["date_min"], opts["date_max"]),
                                min_value=opts["date_min"], max_value=opts["date_max"], key=f"date_{k}")
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                d_from, d_to = rng[0].isoformat(), rng[1].isoformat()
        tcs = st.multiselect("Time control", opts["time_classes"], key=f"tc_{k}",
                             format_func=str.title, placeholder="All time controls")
        ops = st.multiselect("Opening", opts["openings"], key=f"op_{k}", placeholder="All openings")
        filters = an.Filters(
            color=color.lower() if color and color != "All" else None,
            date_from=d_from, date_to=d_to, time_classes=tuple(tcs), openings=tuple(ops),
        )
        st.markdown(
            '<div class="cl-footer">Engine: Stockfish via python-chess. Error categories use '
            "project-defined centipawn thresholds.</div>",
            unsafe_allow_html=True,
        )
        return player, filters


def require_data() -> tuple | None:
    conn, player = CTX.get("conn"), CTX.get("player")
    if conn is None or player is None:
        ui.empty_state("No data loaded yet",
                       "Open the Home page and click Try Demo, upload a PGN file, or import from Chess.com.")
        st.page_link(PAGES["home"], label="Go to Home", icon=":material/home:")
        return None
    return conn, player, CTX["filters"]


def subtitle(player: str) -> str:
    return f"{player} - {dataset_label()}"


def needs_analysis_note(ov: dict) -> bool:
    if ov["analyzed_games"] == 0:
        ui.empty_state(
            "No engine analysis yet",
            "Move-quality statistics appear after Stockfish analyzes these games. "
            "Run the analysis from the Home page.",
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def page_home() -> None:
    st.markdown(
        """
        <div class="cl-hero">
          <h1>ChessLens</h1>
          <p>Engine-powered analytics for chess improvement. ChessLens analyzes your games with
          Stockfish, stores every move in a SQL database, measures where your mistakes happen and
          turns the statistics into a personalized training plan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 2.4], vertical_alignment="center")
    with c1:
        if st.button("Try Demo", type="primary", icon=":material/play_arrow:", width="stretch",
                     disabled=not config.DEMO_DB_PATH.exists()):
            go_demo()
            st.switch_page(PAGES["overview"])
    with c2:
        st.markdown(
            "**No account or upload needed.** The demo opens 158 precomputed, engine-analyzed public "
            "games of one Lichess player from the CC0 Lichess open database."
        )

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("Engine analysis", "Every move is evaluated by Stockfish; centipawn loss is measured from "
                            "the mover's point of view."),
        ("SQL analytics", "Games, moves and evaluations live in a normalized SQLite database queried "
                          "with joins and GROUP BYs."),
        ("Coaching plan", "Rule-based priorities backed by your own error rates, with concrete "
                          "positions from your games."),
        ("Risk model", "A scikit-learn model estimates serious-error risk from pre-move board "
                       "features, evaluated on unseen games."),
    ]
    for col, (title, body) in zip((f1, f2, f3, f4), features):
        col.markdown(f'<div class="cl-feature"><h4>{ui.esc(title)}</h4><p>{ui.esc(body)}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("### Analyze your own games")
    st.caption("Imported games are stored in a temporary database for this browser session only "
               "and are never shared with other users.")
    for kind, msg in ss.import_messages:
        getattr(st, kind)(msg)
    tab_pgn, tab_cc = st.tabs(["Upload PGN", "Import from Chess.com"])
    with tab_pgn:
        pgn_import()
    with tab_cc:
        chesscom_import()

    if has_user_data():
        analysis_panel()


def _after_import(report: ingest.ParseReport, preferred_player: str | None) -> None:
    messages = []
    if report.games:
        summary = store_report(user_db_path(), report)
        messages.append(("success", f"Imported {summary.inserted} new games"
                         + (f" ({summary.duplicates} duplicates skipped)" if summary.duplicates else "")
                         + "."))
        if preferred_player is None:
            cands = ingest.candidate_players(report.games)
            preferred_player = cands[0][0] if cands else None
        if preferred_player:
            ss.user_player = preferred_player
        ss.dataset = "user"
    if report.skipped or report.errors:
        detail = "; ".join(report.errors[:3])
        messages.append(("warning", f"{report.skipped} game(s) skipped. {detail}"))
    if report.truncated:
        messages.append(("info", f"Only the first {config.MAX_UPLOAD_GAMES} games were imported."))
    ss.import_messages = messages


def pgn_import() -> None:
    with st.form("pgn_form", clear_on_submit=False, border=False):
        up = st.file_uploader("PGN file with one or more games", type=["pgn", "txt"])
        submitted = st.form_submit_button("Import games", type="primary")
    if submitted:
        if up is None:
            st.warning("Choose a PGN file first.")
            return
        raw = up.getvalue()
        if len(raw) > config.MAX_UPLOAD_BYTES:
            st.error(f"File too large (limit {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
            return
        with st.spinner("Parsing and validating games ..."):
            report = ingest.parse_pgn_text(ingest.decode_upload(raw), max_games=config.MAX_UPLOAD_GAMES)
        if not report.games:
            st.error("No valid games found. " + " ".join(report.errors[:3]))
            return
        _after_import(report, None)
        st.rerun()


def chesscom_import() -> None:
    with st.form("cc_form", border=False):
        c1, c2 = st.columns([2, 1])
        username = c1.text_input("Chess.com username", placeholder="e.g. your_username")
        n = c2.number_input("Recent games", min_value=5, max_value=config.CHESSCOM_MAX_GAMES,
                            value=config.CHESSCOM_DEFAULT_GAMES, step=5)
        submitted = st.form_submit_button("Download games", type="primary")
    st.caption("Uses the official Chess.com Published-Data API (public games only, fetched serially "
               "with timeouts and rate-limit back-off).")
    if submitted:
        try:
            with st.status("Contacting Chess.com ...", expanded=False) as status:
                report = ingest.fetch_chesscom_games(
                    username, int(n), progress=lambda m: status.update(label=m))
                status.update(label=f"Downloaded {len(report.games)} games", state="complete")
        except ingest.ChessComError as exc:
            st.error(str(exc))
            return
        if not report.games:
            st.error("No games could be imported. " + " ".join(report.errors[:2]))
            return
        _after_import(report, username.strip())
        st.rerun()


def analysis_panel() -> None:
    st.markdown("### Stockfish analysis")
    conn = db.connect(user_db_path())
    try:
        players = db.list_players(conn)
        if players.empty:
            return
        names = players["username"].tolist()
        idx = next((i for i, n in enumerate(names)
                    if ss.user_player and n.lower() == ss.user_player.lower()), 0)
        c1, c2 = st.columns([1.3, 2])
        player = c1.selectbox("Player to analyze", names, index=idx,
                              help="Players are ordered by number of games in your import.")
        ss.user_player = player
        row = players.set_index("username").loc[player]
        pending = db.unanalyzed_games(conn, player, 100_000)
        with c2:
            ui.stat_cards([("Games imported", f"{int(row['games'])}", "for this player"),
                           ("Analyzed", f"{int(row['analyzed'])}", "with Stockfish", "accent"),
                           ("Waiting", f"{len(pending)}", "not yet analyzed")], cols=3)
    finally:
        conn.close()

    if ss.job is not None:
        st.info("Analysis in progress - follow it in the sidebar. Dashboards update as games finish.")
        return
    if not pending:
        st.success("All games for this player are analyzed. Explore the dashboard pages above.")
        return
    engine = find_stockfish()
    if engine is None:
        st.error("Stockfish was not found on this machine. Install it (see README) or set the "
                 "STOCKFISH_PATH environment variable. Game-level statistics remain available.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    depth = c1.slider("Engine depth", config.MIN_DEPTH, config.MAX_DEPTH, config.DEFAULT_DEPTH,
                      help="Higher depth = stronger but slower analysis. Depth 8-10 is a good balance.")
    max_n = min(config.MAX_ANALYSIS_GAMES, len(pending))
    n = max_n if max_n == 1 else c2.slider("Games to analyze (most recent first)", 1, max_n, min(25, max_n))
    est = n * 75 * 0.035 * (1.45 ** (depth - 10))
    c3.metric("Estimated time", f"~{max(5, est):.0f} s",
              help="Rough estimate; positions already in the cache are free.")
    if len(pending) > config.MAX_ANALYSIS_GAMES:
        st.caption(f"Up to {config.MAX_ANALYSIS_GAMES} games per run on this server; run again to "
                   "analyze more.")
    if st.button(f"Analyze {n} game{'s' if n != 1 else ''} with Stockfish", type="primary",
                 icon=":material/bolt:"):
        ss.job = AnalysisJob(str(user_db_path()), pending[:n], depth).start()
        ss.dataset = "user"
        ss.import_messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def page_overview() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Overview", subtitle(player))
    ov = an.overview(conn, player, f)
    if ov["games"] == 0:
        ui.empty_state("No games match these filters", "Try widening the filters in the sidebar.")
        return
    analyzed = ov["analyzed_games"] > 0
    ui.stat_cards([
        ("Games", f"{ov['games']}", f"{ov['analyzed_games']} engine-analyzed"),
        ("Score", ui.fmt_pct(ov["avg_score"], 0), f"{ov['wins']}W / {ov['draws']}D / {ov['losses']}L", "accent"),
        ("Win / Draw / Loss",
         f"{ui.fmt_pct(ov['wins'] / max(ov['decided_games'], 1), 0)} / "
         f"{ui.fmt_pct(ov['draws'] / max(ov['decided_games'], 1), 0)} / "
         f"{ui.fmt_pct(ov['losses'] / max(ov['decided_games'], 1), 0)}", "share of decided games"),
        ("Avg centipawn loss", ui.fmt_num(ov["acpl"]) if analyzed else "-", "per move, lower is better"),
    ])
    if analyzed:
        ui.stat_cards([
            ("Blunder rate", ui.fmt_pct(ov["blunder_rate"]), "moves losing 200+ cp", "warn"),
            ("Mistake rate", ui.fmt_pct(ov["mistake_rate"]), "moves losing 100-199 cp"),
            ("Inaccuracy rate", ui.fmt_pct(ov["inaccuracy_rate"]), "moves losing 50-99 cp"),
            ("Moves analyzed", f"{ov['moves']:,}", f"avg accuracy {ui.fmt_num(ov['accuracy'], 1)}"),
        ])

    games = an.games_table(conn, player, f)
    colors = an.by_color(conn, player, f)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.plotly_chart(charts.results_donut(ov["wins"], ov["draws"], ov["losses"]), width="stretch")
    with c2:
        if not colors.empty:
            st.plotly_chart(charts.color_results(colors), width="stretch")
    if not colors.empty:
        tbl = colors.rename(columns={"color": "Color", "games": "Games"})
        tbl["Color"] = tbl["Color"].str.title()
        st.dataframe(
            tbl[["Color", "Games", "avg_score", "win_rate", "acpl", "blunder_rate", "serious_rate"]],
            hide_index=True, width="stretch",
            column_config={
                "avg_score": st.column_config.NumberColumn("Score", format="percent"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                "acpl": st.column_config.NumberColumn("Avg CP loss", format="%.0f"),
                "blunder_rate": st.column_config.NumberColumn("Blunder rate", format="percent"),
                "serious_rate": st.column_config.NumberColumn("Mistake + blunder rate", format="percent"),
            },
        )
    if games["player_elo"].notna().sum() >= 2:
        st.plotly_chart(charts.rating_trend(games), width="stretch")
    else:
        st.caption("Rating trend unavailable - the imported games have no rating information.")

    st.markdown("#### Recent games")
    recent = games.sort_values("played_at", ascending=False).head(12)
    st.dataframe(
        recent[["played_at", "color", "opponent", "opponent_elo", "outcome", "opening_family",
                "time_class", "acpl", "blunders"]],
        hide_index=True, width="stretch",
        column_config={
            "played_at": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY"),
            "color": "Color", "opponent": "Opponent",
            "opponent_elo": st.column_config.NumberColumn("Opp. rating", format="%d"),
            "outcome": "Result", "opening_family": "Opening", "time_class": "Time control",
            "acpl": st.column_config.NumberColumn("Avg CP loss", format="%.0f"),
            "blunders": st.column_config.NumberColumn("Blunders", format="%d"),
        },
    )


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------


def page_performance() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Performance Analytics", subtitle(player))
    ov = an.overview(conn, player, f)
    if ov["games"] == 0:
        ui.empty_state("No games match these filters", "Try widening the filters in the sidebar.")
        return
    if needs_analysis_note(ov):
        return

    games = an.games_table(conn, player, f)
    tr = an.trend(conn, player, f)
    c1, c2 = st.columns(2)
    with c1:
        if len(tr) >= 2:
            st.plotly_chart(charts.trend_lines(tr, tr.attrs.get("granularity", "period")), width="stretch")
        else:
            ui.empty_state("Not enough dates for a trend", "Games span a single period.")
    with c2:
        st.plotly_chart(charts.rolling_acpl(games), width="stretch")

    st.markdown("#### Game phases")
    ph = an.phase_stats(conn, player, f)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.plotly_chart(charts.phase_error_rates(ph), width="stretch")
    with c2:
        st.dataframe(
            ph[["phase", "moves", "acpl", "inaccuracy_rate", "mistake_rate", "blunder_rate"]],
            hide_index=True, width="stretch",
            column_config={
                "phase": "Phase", "moves": st.column_config.NumberColumn("Moves", format="%d"),
                "acpl": st.column_config.NumberColumn("Avg CP loss", format="%.0f"),
                "inaccuracy_rate": st.column_config.NumberColumn("Inaccuracies", format="percent"),
                "mistake_rate": st.column_config.NumberColumn("Mistakes", format="percent"),
                "blunder_rate": st.column_config.NumberColumn("Blunders", format="percent"),
            },
        )
        small = ph[ph["moves"] < config.MIN_MOVES_FOR_PHASE_CLAIM]
        if not small.empty:
            st.warning(f"Small sample: {', '.join(small['phase'])} has fewer than "
                       f"{config.MIN_MOVES_FOR_PHASE_CLAIM} analyzed moves - interpret with care.")
        with st.expander("How phases are classified"):
            st.markdown(
                "- **Endgame**: at most 6 queens, rooks, bishops and knights remain (both sides).\n"
                "- **Opening**: move 12 or earlier and at least 11 of those pieces remain.\n"
                "- **Middlegame**: everything else.\n\n"
                "Rates are normalized per evaluated move, so phases with more moves are not "
                "over-weighted. This is a project heuristic inspired by Lichess's game divider."
            )

    moves = an.move_rows(conn, player, f)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.cpl_histogram(moves), width="stretch")
    with c2:
        st.plotly_chart(charts.move_number_profile(an.move_number_profile(conn, player, f)), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        tc = an.time_class_stats(conn, player, f)
        if len(tc) >= 1:
            st.plotly_chart(charts.time_class_bars(tc), width="stretch")
    with c2:
        ctx = an.eval_context_stats(conn, player, f)
        if not ctx.empty:
            st.plotly_chart(charts.context_bars(ctx), width="stretch")

    clock = an.time_pressure_stats(conn, player, f)
    if set(clock["clock"]) == {"low_clock", "normal"}:
        lo = clock.set_index("clock").loc["low_clock"]
        no = clock.set_index("clock").loc["normal"]
        ui.stat_cards([
            ("Low-clock error rate", ui.fmt_pct(lo["serious_rate"]),
             f"{int(lo['moves'])} moves with < 10% of base time", "warn"),
            ("Normal-clock error rate", ui.fmt_pct(no["serious_rate"]), f"{int(no['moves'])} moves"),
        ], cols=2)
    else:
        st.caption("Clock-based (time-pressure) statistics need %clk data, which these games do not include.")


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def page_openings() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Opening Analysis", subtitle(player))
    df = an.opening_stats(conn, player, f)
    if df.empty:
        ui.empty_state("No games match these filters", "Try widening the filters in the sidebar.")
        return
    min_games = st.slider("Minimum games per opening for comparisons", 1, 10,
                          config.MIN_GAMES_FOR_OPENING_CLAIM)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.plotly_chart(charts.opening_bars(df), width="stretch")
    with c2:
        if df["acpl"].notna().any() and (df["games"] >= min_games).any():
            st.plotly_chart(charts.opening_scatter(df, min_games), width="stretch")
        else:
            ui.empty_state("No analyzed openings yet", "Run Stockfish analysis to compare accuracy by opening.")

    table = df.copy()
    ci = table.apply(lambda r: an.wilson_interval(r["wins"], int(r["games"])), axis=1)
    table["win_ci"] = [f"{lo:.0%} - {hi:.0%}" for lo, hi in ci]
    table["sample"] = table["games"].map(
        lambda n: "Too small to judge" if n < min_games else ("Moderate" if n < 10 else "Reasonable"))
    st.markdown("#### All openings")
    st.dataframe(
        table[["opening", "games", "wins", "draws", "losses", "avg_score", "win_rate", "win_ci", "acpl",
               "serious_rate", "opening_inaccuracy_rate", "sample"]],
        hide_index=True, width="stretch",
        column_config={
            "opening": "Opening family", "games": "Games", "wins": "W", "draws": "D", "losses": "L",
            "avg_score": st.column_config.NumberColumn("Score", format="percent"),
            "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
            "win_ci": st.column_config.TextColumn("Win rate 95% CI", help="Wilson score interval"),
            "acpl": st.column_config.NumberColumn("Avg CP loss", format="%.0f"),
            "serious_rate": st.column_config.NumberColumn("Mistake + blunder rate", format="percent"),
            "opening_inaccuracy_rate": st.column_config.NumberColumn(
                "Opening-phase error rate", format="percent",
                help="Share of moves in the opening phase that were inaccuracies or worse"),
            "sample": "Sample size",
        },
    )
    eligible = df[(df["games"] >= min_games) & df["serious_rate"].notna()]
    if not eligible.empty:
        st.markdown("#### Openings associated with higher error rates")
        worst = eligible.sort_values("serious_rate", ascending=False).head(5)
        ui.stat_cards([
            (r.opening, ui.fmt_pct(r.serious_rate), f"mistake+blunder rate - {int(r.games)} games, "
                                                     f"{int(r.moves)} moves", "warn")
            for r in worst.itertuples()
        ])
    st.caption("Opening names come from the PGN headers (Lichess `Opening` tag or Chess.com `ECOUrl`) and "
               "are grouped into families. Win-rate intervals are wide for small samples; differences "
               "between openings with only a few games are not reliable.")


# ---------------------------------------------------------------------------
# Mistake explorer
# ---------------------------------------------------------------------------


def page_mistakes() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Mistake Explorer", subtitle(player))
    ov = an.overview(conn, player, f)
    if needs_analysis_note(ov):
        return

    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    cats = c1.multiselect("Categories", ["blunder", "mistake", "inaccuracy"],
                          default=["blunder", "mistake"], format_func=str.title)
    phases = c2.multiselect("Phases", ["opening", "middlegame", "endgame"], format_func=str.title,
                            placeholder="All phases")
    order = c3.selectbox("Sort by", ["Biggest loss", "Most recent"])
    if not cats:
        st.info("Select at least one category.")
        return
    where, params = "", []
    phases = [p for p in phases if p in ("opening", "middlegame", "endgame")]
    if phases:
        where = f"AND pm.phase IN ({','.join('?' * len(phases))})"
        params = phases
    df = an.mistakes(conn, player, f, categories=cats, extra_where=where, extra_params=params, limit=300)
    if df.empty:
        ui.empty_state("No matching mistakes", "Nothing in these categories - nice!")
        return
    if order == "Most recent":
        df = df.sort_values(["played_at", "ply"], ascending=[False, True]).reset_index(drop=True)

    if ss.explorer_move_id is not None:
        match = df.index[df["move_id"] == ss.explorer_move_id].tolist()
        if match:
            ss.mistake_idx = match[0]
        ss.explorer_move_id = None
    idx = min(ss.get("mistake_idx", 0), len(df) - 1)

    nav1, nav2, nav3 = st.columns([1, 4, 1], vertical_alignment="bottom")
    if nav1.button("Previous", icon=":material/chevron_left:", width="stretch", disabled=idx == 0):
        idx -= 1
    if nav3.button("Next", icon=":material/chevron_right:", width="stretch", disabled=idx >= len(df) - 1):
        idx += 1
    labels = [
        f"#{i + 1}  {r.move_number}{'.' if r.color == 'white' else '...'}{r.san}  ({r.classification}, "
        f"-{int(r.cpl)} cp) vs {r.opponent}, {str(r.played_at)[:10]}"
        for i, r in enumerate(df.itertuples())
    ]
    idx = nav2.selectbox(f"Mistake {idx + 1} of {len(df)}", range(len(df)), index=idx,
                         format_func=lambda i: labels[i])
    ss.mistake_idx = idx
    r = df.iloc[idx]

    left, right = st.columns([1.05, 1])
    with left:
        ui.render_board(ui.board_svg(r["fen_before"], r["uci"], r["best_move_uci"], r["color"]))
        st.caption("Red arrow: move played. Green arrow: Stockfish's preferred move. "
                   "Board shown from your side, before the move.")
    with right:
        move_label = f"{r['move_number']}{'.' if r['color'] == 'white' else '...'}{r['san']}"
        st.markdown(f"### {ui.esc(move_label)} {ui.class_pill(r['classification'])}",
                    unsafe_allow_html=True)
        ui.stat_cards([
            ("Eval before", ui.fmt_eval(r["eval_before"]), "your perspective"),
            ("Eval after", ui.fmt_eval(r["eval_after"]), "after your move"),
            ("Centipawns lost", f"{int(r['cpl'])}", r["classification"].title(), "warn"),
        ], cols=3)
        result_text = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}.get(r["score"], "Unfinished")
        rows = [
            ("Move played", r["san"]),
            ("Engine preferred", r["best_move_san"] or "-"),
            ("Phase", r["phase"].title()),
            ("Opponent", f"{r['opponent']} ({int(r['opponent_elo']) if pd.notna(r['opponent_elo']) else '?'})"),
            ("You played", f"{r['color'].title()} ({int(r['player_elo']) if pd.notna(r['player_elo']) else '?'})"),
            ("Date", str(r["played_at"])[:16] if r["played_at"] else "-"),
            ("Opening", r["opening"] or r["opening_family"]),
            ("Time control", f"{r['time_control'] or '?'} ({r['time_class']})"),
            ("Game result", result_text),
        ]
        st.markdown(
            '<div class="cl-kv">' + "".join(
                f'<div class="k">{ui.esc(k)}</div><div class="v">{ui.esc(v)}</div>' for k, v in rows)
            + "</div>",
            unsafe_allow_html=True,
        )
        site = r["site"]
        if isinstance(site, str) and site.startswith("https://"):
            st.link_button("Open original game", site, icon=":material/open_in_new:")
    game = an.game_analysis(conn, int(r["game_id"]))
    if not game.empty:
        st.plotly_chart(charts.game_eval_curve(game, int(r["ply"]), r["color"]), width="stretch")


# ---------------------------------------------------------------------------
# Coaching
# ---------------------------------------------------------------------------


def page_coaching() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Coaching Recommendations", subtitle(player))
    ov = an.overview(conn, player, f)
    recs, note = recommendations.generate(conn, player, f)
    if not recs:
        ui.empty_state("Not enough data yet", note or "Analyze more games to unlock recommendations.")
    else:
        st.markdown("Your **top practice priorities**, ranked by how strongly your own statistics "
                    "support them (a two-proportion z statistic against the rest of your games).")
        for i, rec in enumerate(recs, 1):
            examples = "".join(f"<li>{ui.esc(e['label'])}</li>" for e in rec.examples)
            st.markdown(
                f'<div class="cl-rec"><h3>{i}. {ui.esc(rec.title)}{ui.confidence_badge(rec.confidence)}</h3>'
                f'<div class="evidence">{ui.esc(rec.evidence)}</div>'
                f'<div class="advice"><b>Practice:</b> {ui.esc(rec.advice)}</div>'
                + (f'<div class="evidence" style="margin-top:.6rem"><b>From your games:</b>'
                   f'<ul style="margin:.2rem 0 0 0">{examples}</ul></div>' if examples else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            if rec.examples:
                cols = st.columns(len(rec.examples) + 2)
                for j, ex in enumerate(rec.examples):
                    if cols[j].button(f"View example {j + 1}", key=f"ex_{i}_{j}", icon=":material/visibility:"):
                        ss.explorer_move_id = ex["move_id"]
                        st.switch_page(PAGES["mistakes"])
        if note:
            st.caption(note)
        st.caption("Recommendations are rule-based and only reference phases, colours, openings, time "
                   "controls and evaluation contexts that ChessLens measures. ChessLens does not detect "
                   "specific tactical motifs such as forks or pins.")

    st.markdown("#### Export")
    summary = reporting.coaching_summary(
        player, ov, an.phase_stats(conn, player, f), an.opening_stats(conn, player, f), recs, note,
        dataset_label(),
    )
    safe = "".join(ch for ch in player if ch.isalnum() or ch in "-_") or "player"
    c1, c2, c3 = st.columns(3)
    c1.download_button("Coaching summary (.md)", summary, file_name=f"chesslens_{safe}_coaching.md",
                       mime="text/markdown", icon=":material/description:", width="stretch")
    c2.download_button("Move analysis (.csv)", reporting.to_csv_bytes(an.move_rows(conn, player, f)),
                       file_name=f"chesslens_{safe}_moves.csv", mime="text/csv",
                       icon=":material/table:", width="stretch")
    c3.download_button("Game summary (.csv)", reporting.to_csv_bytes(an.games_table(conn, player, f)),
                       file_name=f"chesslens_{safe}_games.csv", mime="text/csv",
                       icon=":material/table_chart:", width="stretch")
    with st.expander("Preview coaching summary"):
        st.markdown(summary)


# ---------------------------------------------------------------------------
# ML insights
# ---------------------------------------------------------------------------


@st.cache_resource
def load_risk_model():
    model = ml.load_model(config.ML_MODEL_PATH)
    if model is not None:
        return model
    frame_path = config.MODELS_DIR / "ml_training_frame.csv.gz"
    metrics = ml.load_metrics(config.ML_METRICS_PATH)
    if not frame_path.exists() or metrics is None:
        return None
    frame = pd.read_csv(frame_path)
    name = metrics["selected_model"]
    train_idx, _ = ml.grouped_split(frame)
    train = frame.iloc[train_idx]
    return ml.candidate_models()[name].fit(train[ml.FEATURE_COLUMNS], train[ml.TARGET_COLUMN])


def page_ml() -> None:
    ui.page_header("ML Insights", "Predicting serious-error risk from the position before a move")
    m = ml.load_metrics(config.ML_METRICS_PATH)
    if m is None:
        ui.empty_state("Model not trained yet", "Run `python scripts/prepare_demo.py` to train the model.")
        return
    sel = m["models"][m["selected_model"]]["test"]
    base = m["models"]["Dummy (stratified)"]["test"]
    ui.stat_cards([
        ("Training data", f"{m['n_positions']:,}", f"positions from {m['n_games']} games"),
        ("Train / test games", f"{m['n_train_games']} / {m['n_test_games']}", "split by whole game"),
        ("Base rate", ui.fmt_pct(m["positive_rate_test"]), "test moves that were mistakes/blunders"),
        ("Avg precision", f"{sel['average_precision']:.3f}",
         f"{m['selected_model']} vs {base['average_precision']:.3f} baseline", "accent"),
        ("Top-decile lift", f"{sel['lift_top_decile']:.2f}x",
         f"error rate {ui.fmt_pct(sel['top_decile_error_rate'])} in riskiest 10%"),
    ])
    st.markdown(
        f"**Question:** {m['target']}. Features describe only the board *before* the move "
        "(material, mobility, checks, attacked pieces, phase) plus ratings and time control - no engine "
        f"evaluations or information about the move played. Selected model: **{m['selected_model']}** "
        f"({m['selection_rule'].lower()})."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.model_comparison(m["models"]), width="stretch")
    with c2:
        st.plotly_chart(charts.pr_curves(m["pr_curves"], m["positive_rate_test"]), width="stretch")
    rows = []
    for name, r in m["models"].items():
        t = r["test"]
        rows.append({"Model": name, "CV avg precision": f"{r['cv_average_precision_mean']:.3f} "
                     f"± {r['cv_average_precision_std']:.3f}", "Test avg precision": t["average_precision"],
                     "ROC AUC": t["roc_auc"], "Precision": t["precision"], "Recall": t["recall"],
                     "F1": t["f1"], "Flagged share": t["flagged_share"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ("Test avg precision", "ROC AUC", "Precision", "Recall", "F1",
                                          "Flagged share")})
    st.caption("Precision / recall / F1 use a 0.5 threshold on class-balanced models, which flags a large "
               "share of moves; average precision and ROC AUC are threshold-free.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.calibration_bars(m["calibration_deciles"]), width="stretch")
    with c2:
        st.plotly_chart(charts.importance_bars(
            m["permutation_importance"], "importance",
            f"Permutation importance ({m['selected_model']}, drop in avg precision)"), width="stretch")
    with st.expander("Logistic regression coefficients (standardized features)"):
        coefs = pd.DataFrame(m["logistic_coefficients"])
        coefs["std"] = 0.0
        st.plotly_chart(charts.importance_bars(coefs.to_dict("records"), "coefficient",
                                               "Positive = higher error risk", top=20), width="stretch")
        st.caption("Coefficients describe associations in the training data, not causal effects.")

    ext = m.get("external_evaluation")
    if ext:
        st.markdown("#### Out-of-sample check on the demo player")
        st.markdown(
            f"{ext['description']}: {ext['n_positions']:,} positions from {ext['n_games']} games, "
            f"base rate {ext['positive_rate']:.1%}. Average precision **{ext['average_precision']:.3f}**, "
            f"ROC AUC **{ext['roc_auc']:.3f}**, riskiest-decile error rate {ext['top_decile_error_rate']:.1%} "
            f"({ext['lift_top_decile']:.2f}x the base rate)."
        )

    st.markdown("#### Apply the model to the selected player")
    conn, player = CTX.get("conn"), CTX.get("player")
    if conn is None or player is None:
        st.caption("Load a dataset to score its positions.")
    else:
        frame = build_ml_frame(ml.load_ml_moves(conn, player))
        model = load_risk_model()
        if model is None or len(frame) < 50 or frame[ml.TARGET_COLUMN].nunique() < 2:
            st.caption("Need at least 50 analyzed moves (with some mistakes) for this player.")
        else:
            frame["risk"] = ml.predict_risk(model, frame)
            frame["bucket"] = pd.qcut(frame["risk"].rank(method="first"), 5,
                                      labels=["Lowest", "Low", "Medium", "High", "Highest"])
            buckets = (frame.groupby("bucket", observed=True)
                       .agg(mean_predicted=("risk", "mean"), observed_rate=(ml.TARGET_COLUMN, "mean"),
                            positions=("risk", "size")).reset_index())
            st.plotly_chart(charts.risk_bucket_bars(buckets), width="stretch")
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(frame[ml.TARGET_COLUMN], frame["risk"])
            st.caption(f"{player}: {len(frame):,} analyzed moves, base rate "
                       f"{frame[ml.TARGET_COLUMN].mean():.1%}, average precision {ap:.3f}. "
                       "Predicted probabilities are inflated by class balancing; read them as a ranking.")

    st.markdown("#### Limitations")
    st.markdown("\n".join(f"- {x}" for x in m["limitations"]))


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------


def page_about() -> None:
    ui.page_header("Methodology", "How ChessLens turns games into statistics")
    st.markdown(
        f"""
**Pipeline:** Import → Parse → Clean → Validate → Analyze → Store → Query → Visualize

1. **Import** - PGN upload, Chess.com Published-Data API (most recent monthly archives, serial requests,
   {config.CHESSCOM_TIMEOUT_S}s timeouts, back-off on HTTP 429), or the precomputed demo.
2. **Parse / clean / validate** - python-chess replays every move; games with illegal moves or
   non-standard variants are skipped and reported. Missing ratings, dates and openings become NULL /
   "Unknown". Duplicate games are detected with a hash of players, date, result and move list.
3. **Analyze** - Stockfish evaluates every position once (N+1 positions for N moves) at a configurable
   depth (default {config.DEFAULT_DEPTH}). Evaluations are cached by position, in memory and in SQLite.
4. **Store** - normalized SQLite tables: `players`, `games`, `moves`, `position_evals`,
   `move_analysis`, `analysis_runs`, plus the `v_player_games` view.
5. **Query / visualize** - every dashboard number comes from SQL (CTEs, joins, GROUP BY, HAVING-style
   filters) rendered with Plotly.

**Centipawn loss.** For a move by player *P*: `eval_before` is the engine's evaluation of the position
before the move and `eval_after` the evaluation after the played move, both from *P*'s point of view and
clamped to ±{config.EVAL_CAP_CP} cp. `CPL = max(0, eval_before − eval_after)`; if *P* played the engine's
top move the loss is 0. Mate scores are converted to ±({config.MATE_SCORE_CP} − plies to mate) before
clamping, so missing a mate counts as at most a {2 * config.EVAL_CAP_CP} cp swing.

**Categories (project-defined, not official Chess.com/Lichess classifications):**
good < {config.INACCURACY_CP} cp ≤ inaccuracy < {config.MISTAKE_CP} cp ≤ mistake < {config.BLUNDER_CP} cp ≤ blunder.

**Accuracy** uses Lichess's published win-probability formula averaged per move (an approximation of
their game accuracy score).

**Statistics.** Error rates are normalized per evaluated move. Opening win rates show Wilson 95%
intervals and sample-size labels. Recommendations rank weaknesses by a two-proportion z statistic
(or a score-difference z for results) and require minimum sample sizes.

**Privacy.** Uploaded and imported games go to a temporary per-session SQLite file that other users
cannot access and that is deleted after {SESSION_MAX_AGE_S // 3600} hours. The demo database is opened
read-only.

**Demo data provenance.** All demo games are from the
[Lichess open database](https://database.lichess.org/) (January 2013 standard rated games), released
under CC0. The demo player was chosen by documented criteria (60-160 games that month, average rating
1350-1850, mixed time controls). No private student data is included.
        """
    )
    if CTX.get("conn") is not None:
        st.markdown("**Current database contents**")
        st.dataframe(pd.DataFrame([db.table_counts(CTX["conn"])]), hide_index=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

init_state()
cleanup_old_sessions()
if ss.dataset is None and st.query_params.get("demo") == "1":
    go_demo()

PAGES = {
    "home": st.Page(page_home, title="Home", icon=":material/home:", url_path="home", default=True),
    "overview": st.Page(page_overview, title="Overview", icon=":material/dashboard:", url_path="overview"),
    "performance": st.Page(page_performance, title="Performance", icon=":material/monitoring:",
                           url_path="performance"),
    "openings": st.Page(page_openings, title="Openings", icon=":material/menu_book:", url_path="openings"),
    "mistakes": st.Page(page_mistakes, title="Mistake Explorer", icon=":material/search:",
                        url_path="mistakes"),
    "coaching": st.Page(page_coaching, title="Coaching", icon=":material/school:", url_path="coaching"),
    "ml": st.Page(page_ml, title="ML Insights", icon=":material/psychology:", url_path="ml"),
    "about": st.Page(page_about, title="Methodology", icon=":material/info:", url_path="methodology"),
}

nav = st.navigation(list(PAGES.values()), position="top")
conn = open_conn()
try:
    player, filters = sidebar(conn)
    CTX.update(conn=conn, player=player, filters=filters)
    if ss.flash:
        kind, msg = ss.flash
        getattr(st, kind)(msg)
        ss.flash = None
    nav.run()
finally:
    if conn is not None:
        conn.close()
