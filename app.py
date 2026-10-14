"""ChessLens - import games, review mistakes, and choose what to practice.

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
from src.pipeline import AnalysisJob, describe_analysis, describe_import, store_report
from src.ui_state import describe_filters, pick_move, sample_unit, step_move

st.set_page_config(
    page_title="ChessLens",
    page_icon="♞",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()

SESSION_DIR = Path(tempfile.gettempdir()) / "chesslens_sessions"
SESSION_MAX_AGE_S = 12 * 3600
DEMO_PLAYER = "sani5057"
DEMO_LABEL = "Demo - public Lichess games, January 2013 (CC0)"
USER_LABEL = "Your imported games (this browser session)"
WORKFLOW = ["Try demo or import", "Analyze", "Understand results", "Review mistakes", "Practice"]
ss = st.session_state
CTX: dict = {}


# ---------------------------------------------------------------------------
# Session and data context
# ---------------------------------------------------------------------------


def init_state() -> None:
    ss.setdefault("dataset", None)  # None | "demo" | "user"
    ss.setdefault("user_db", None)
    ss.setdefault("user_player", None)
    ss.setdefault("job", None)
    ss.setdefault("flash", None)
    ss.setdefault("explorer_move_id", None)
    ss.setdefault("mistake_pick", None)
    ss.setdefault("import_messages", [])
    ss.setdefault("pending_player", None)


@st.cache_resource(ttl=3600)
def cleanup_old_sessions() -> None:
    """Remove session files untouched for 12 hours. Runs at most hourly, and only while the app is used."""
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


def dataset_name() -> str:
    return "Demo" if ss.dataset == "demo" else "Your games"


def dataset_label() -> str:
    return DEMO_LABEL if ss.dataset == "demo" else USER_LABEL


def go_demo() -> None:
    ss.dataset = "demo"


def match_name(names: list[str], wanted: str | None) -> str | None:
    if not wanted:
        return None
    return next((n for n in names if n.lower() == wanted.lower()), None)


def set_user_player(name: str | None) -> None:
    if name:
        ss.user_player = name
        ss.pending_player = name


def reset_filters() -> None:
    prefix = f"{ss.dataset}_"
    for key in list(ss.keys()):
        if isinstance(key, str) and (key.startswith(("color_", "date_", "tc_", "op_")) and prefix in key):
            del ss[key]


def remember_import(outcome) -> None:
    ss.import_messages = [(outcome.kind, outcome.headline)]
    if outcome.details:
        ss.import_messages.append(("info", " ".join(outcome.details)))
    if outcome.rejections:
        ss.import_messages.append(("warning", "Rejected games:\n" + "\n".join(outcome.rejections[:12])))


def show_import_messages() -> None:
    for kind, msg in ss.import_messages:
        getattr(st, kind)(msg)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


@st.fragment(run_every=1.0)
def job_progress() -> None:
    job: AnalysisJob | None = ss.get("job")
    if job is None:
        return
    if job.finished:
        ss.job = None
        ss.flash = describe_analysis(job)
        st.rerun()
    who = f" for {job.player}" if job.player else ""
    if job.queued:
        st.markdown(f"**Queued{who}**")
        st.caption("Another analysis is using the engine. This one starts automatically.")
        if st.button("Cancel analysis", key="cancel_queued"):
            job.cancel()
        return
    st.markdown(f"**Analyzing{who}**")
    st.progress(job.fraction, text=f"Game {min(job.games_done + 1, job.total)} of {job.total}")
    elapsed = time.time() - job.started_at
    if job.fraction > 0.02:
        remaining = elapsed / job.fraction * (1 - job.fraction)
        st.caption(f"About {remaining:.0f}s left. You can keep looking around.")
    if st.button("Cancel analysis", key="cancel_job"):
        job.cancel()


def sidebar(conn) -> tuple[str | None, an.Filters]:
    with st.sidebar:
        st.markdown('<div class="cl-side-brand"><span class="cl-logo">♞</span>ChessLens</div>',
                    unsafe_allow_html=True)
        st.caption("Built for Polonsky Academy")
        if ss.job is not None:
            job_progress()
            st.divider()

        options = ["demo"] + (["user"] if has_user_data() else [])
        if ss.dataset not in options:
            st.caption("No dataset loaded. Try the demo or import games from Start.")
            return None, an.Filters()
        labels = {"demo": "Demo dataset", "user": "My imported games"}
        choice = st.radio("Dataset", options, index=options.index(ss.dataset),
                          format_func=labels.get, horizontal=True)
        if choice != ss.dataset:
            ss.dataset = choice
            st.rerun()
        if conn is None:
            st.caption("This dataset is not available.")
            return None, an.Filters()

        players = db.list_players(conn)
        if players.empty:
            st.caption("No games in this dataset yet.")
            return None, an.Filters()
        names = players["username"].tolist()
        key = f"player_{ss.dataset}"
        if ss.dataset == "user" and ss.get("pending_player"):
            match = match_name(names, ss.pending_player)
            if match:
                ss[key] = match
            ss.pending_player = None
        if ss.get(key) not in names:
            preferred = DEMO_PLAYER if ss.dataset == "demo" else ss.user_player
            ss[key] = match_name(names, preferred) or names[0]
        counts = players.set_index("username")
        player = st.selectbox(
            "Player", names, key=key,
            format_func=lambda n: (
                f"{n} ({int(counts.loc[n, 'games'])} games, {int(counts.loc[n, 'analyzed'])} analyzed)"
            ),
        )
        if ss.dataset == "user":
            ss.user_player = player

        st.markdown("##### Filters")
        opts = an.filter_options(conn, player)
        k = f"{ss.dataset}_{player}"
        color = st.segmented_control("Color", ["All", "White", "Black"], default="All", key=f"color_{k}")
        tcs = st.multiselect("Time control", opts["time_classes"], key=f"tc_{k}",
                             format_func=str.title, placeholder="All time controls")
        d_from = d_to = None
        ops: list[str] = []
        with st.expander("More filters"):
            if opts["date_min"] and opts["date_max"] and opts["date_min"] < opts["date_max"]:
                rng = st.date_input(
                    "Date range", value=(opts["date_min"], opts["date_max"]),
                    min_value=opts["date_min"], max_value=opts["date_max"], key=f"date_{k}",
                )
                if isinstance(rng, (tuple, list)) and len(rng) == 2:
                    d_from, d_to = rng[0].isoformat(), rng[1].isoformat()
            ops = st.multiselect("Opening", opts["openings"], key=f"op_{k}", placeholder="All openings")
            st.button("Reset filters", on_click=reset_filters, width="stretch")
        filters = an.Filters(
            color=color.lower() if color and color != "All" else None,
            date_from=d_from, date_to=d_to, time_classes=tuple(tcs), openings=tuple(ops),
        )
        st.markdown(
            '<div class="cl-footer">Move counts are this player\'s moves. Centipawn loss is from '
            "their side of the board.</div>",
            unsafe_allow_html=True,
        )
        CTX["date_bounds"] = (opts["date_min"], opts["date_max"])
        return player, filters


def require_data() -> tuple | None:
    conn, player = CTX.get("conn"), CTX.get("player")
    if conn is None or player is None:
        ui.empty_state(
            "No games loaded",
            "Open Start and try the demo, or import a PGN or Chess.com games.",
        )
        st.page_link(PAGES["start"], label="Go to Start", icon=":material/home:")
        return None
    return conn, player, CTX["filters"]


def show_context(player: str, filters: an.Filters, extra: str | None = None) -> None:
    items = [("Dataset", dataset_name()), ("Player", player)]
    parts = describe_filters(filters, CTX.get("date_bounds", (None, None)))
    items.append(("Filters", ", ".join(parts) if parts else "none"))
    if extra:
        items.append(("View", extra))
    ui.context_line(items)


def as_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].astype(float) * 100
    return out


def pct_col(label: str, **kwargs):
    return st.column_config.NumberColumn(label, format="%.1f%%", **kwargs)


def empty_filters() -> None:
    ui.empty_state("No games match these filters", "Widen the filters in the sidebar, or reset them.")


def needs_analysis(ov: dict) -> bool:
    if ov["analyzed_games"] == 0:
        ui.empty_state(
            "These games are not analyzed yet",
            "Move quality appears after Stockfish runs. Start the analysis from the Start page.",
        )
        st.page_link(PAGES["start"], label="Go to Start", icon=":material/home:")
        return True
    return False


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


DEMO_PREVIEW_SQL = """
SELECT m.fen_before, m.uci, a.best_move_uci, m.color, m.move_number, m.san, a.cpl
FROM moves m
JOIN move_analysis a ON a.move_id = m.move_id
JOIN games g ON g.game_id = m.game_id
JOIN players p ON p.player_id = CASE m.color WHEN 'white' THEN g.white_id ELSE g.black_id END
WHERE p.username = ? AND a.classification = 'blunder' AND m.phase = 'middlegame'
ORDER BY a.cpl DESC
LIMIT 1
"""


@st.cache_data
def demo_summary() -> dict | None:
    if not config.DEMO_DB_PATH.exists():
        return None
    conn = db.connect(config.DEMO_DB_PATH, read_only=True)
    try:
        players = db.list_players(conn).set_index("username")
        if DEMO_PLAYER not in players.index:
            return None
        preview = conn.execute(DEMO_PREVIEW_SQL, (DEMO_PLAYER,)).fetchone()
        depth = conn.execute("SELECT MAX(depth) FROM analysis_runs").fetchone()[0]
        return {
            "games": int(players.loc[DEMO_PLAYER, "games"]),
            "moves": int(players.loc[DEMO_PLAYER, "analyzed"] and an.overview(conn, DEMO_PLAYER)["moves"]),
            "analyzed": int(players.loc[DEMO_PLAYER, "analyzed"]),
            "depth": depth,
            "preview": tuple(preview) if preview else None,
        }
    finally:
        conn.close()


def page_start() -> None:
    st.markdown(
        '<div class="cl-greeting">Import your chess games, review your biggest mistakes, '
        "and find what to practice next.</div>"
        '<div class="cl-lead">ChessLens answers three questions: how you are scoring, where the '
        "costly moves are, and which area is worth practicing.</div>",
        unsafe_allow_html=True,
    )
    analyzed = 0
    if has_user_data():
        conn = db.connect(user_db_path())
        try:
            players = db.list_players(conn)
            if not players.empty:
                name = match_name(players["username"].tolist(), ss.user_player) or players.iloc[0]["username"]
                analyzed = int(players.set_index("username").loc[name, "analyzed"])
        finally:
            conn.close()
    done = 2 if analyzed else (1 if has_user_data() else 0)
    ui.workflow(WORKFLOW, current=done if done < len(WORKFLOW) else None, done=done)

    demo = demo_summary()
    with st.container(border=True):
        left, right = st.columns([1, 1.4], gap="large", vertical_alignment="center")
        with left:
            if demo and demo["preview"]:
                fen, uci, best, color, *_ = demo["preview"]
                svg = ui.board_svg(fen, uci, best, color, size=340, coordinates=False)
                st.markdown(f'<div class="cl-preview">{svg}</div>', unsafe_allow_html=True)
        with right:
            if demo is None:
                st.markdown(
                    '<div class="cl-meta-title">Demo dataset</div>'
                    '<div class="cl-meta">The demo database is not on this server.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="cl-meta-title">Demo dataset {ui.badge("Ready")}</div>'
                    f'<div class="cl-meta"><b>{demo["games"]} games</b> by {ui.esc(DEMO_PLAYER)}, '
                    f'already analyzed at depth {demo["depth"]}. Move counts below are that '
                    "player's moves only.</div>"
                    '<div class="cl-label">Source</div>'
                    '<div class="cl-value">Lichess open database, January 2013 (CC0)</div>',
                    unsafe_allow_html=True,
                )
            if st.button("Try demo", type="primary", icon=":material/play_arrow:",
                         disabled=demo is None, width="stretch"):
                go_demo()
                st.switch_page(PAGES["dashboard"])

    ui.section("Import games")
    ui.note("Imported games stay in a temporary file for this browser session. "
            "Other visitors cannot open them. A refresh starts a new session.")
    show_import_messages()
    tab_pgn, tab_cc = st.tabs(["Upload PGN", "Chess.com"])
    with tab_pgn:
        pgn_import()
    with tab_cc:
        chesscom_import()

    if has_user_data():
        analysis_panel()


def _after_import(report: ingest.ParseReport, preferred_player: str | None,
                  encoding_warning: str | None = None) -> None:
    summary = store_report(user_db_path(), report) if report.games else None
    outcome = describe_import(report, summary, config.MAX_UPLOAD_GAMES, encoding_warning)
    remember_import(outcome)
    if outcome.inserted:
        if preferred_player is None:
            cands = ingest.candidate_players(report.games)
            preferred_player = cands[0][0] if cands else None
        set_user_player(preferred_player)
        ss.dataset = "user"


def pgn_import() -> None:
    with st.form("pgn_form", clear_on_submit=False, border=False):
        up = st.file_uploader("PGN file with one or more games", type=["pgn", "txt"])
        submitted = st.form_submit_button("Import games", type="primary")
    if not submitted:
        return
    if up is None:
        st.warning("Choose a PGN file first.")
        return
    raw = up.getvalue()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        st.error(f"File too large. The limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        return
    try:
        text, warning = ingest.decode_pgn_bytes(raw)
    except ingest.UploadError as exc:
        st.error(str(exc))
        return
    with st.spinner("Checking the games ..."):
        report = ingest.parse_pgn_text(text, max_games=config.MAX_UPLOAD_GAMES)
    _after_import(report, None, warning)
    st.rerun()


def chesscom_import() -> None:
    with st.form("cc_form", border=False):
        c1, c2 = st.columns([2, 1])
        username = c1.text_input("Chess.com username", placeholder="e.g. your_username")
        n = c2.number_input("Recent games", min_value=5, max_value=config.CHESSCOM_MAX_GAMES,
                            value=config.CHESSCOM_DEFAULT_GAMES, step=5)
        submitted = st.form_submit_button("Download games", type="primary")
    ui.note("Public games only, from the Chess.com Published-Data API.")
    if not submitted:
        return
    try:
        with st.status("Contacting Chess.com ...", expanded=False) as status:
            report = ingest.fetch_chesscom_games(username, int(n), progress=lambda m: status.update(label=m))
            status.update(label=f"Downloaded {len(report.games)} games", state="complete")
    except ingest.ChessComError as exc:
        st.error(str(exc))
        return
    _after_import(report, username.strip())
    st.rerun()


def analysis_panel() -> None:
    ui.section("Analyze with Stockfish")
    conn = db.connect(user_db_path())
    try:
        players = db.list_players(conn)
        if players.empty:
            ui.note("Nothing to analyze yet. Import a file that contains at least one valid game.")
            return
        names = players["username"].tolist()
        ss.analysis_player = match_name(names, ss.user_player) or names[0]
        c1, c2 = st.columns([1.2, 2])
        player = c1.selectbox(
            "Player to analyze", names, key="analysis_player",
            on_change=lambda: set_user_player(ss.analysis_player),
        )
        ss.user_player = player
        row = players.set_index("username").loc[player]
        pending = db.unanalyzed_games(conn, player, 100_000)
        imported, analyzed = int(row["games"]), int(row["analyzed"])
        with c2:
            ui.stat_cards([
                ("Imported", f"{imported}", "games for this player"),
                ("Analyzed", f"{analyzed}", "finished by Stockfish", "accent"),
                ("Waiting", f"{len(pending)}", "not analyzed yet"),
            ], cols=3)
    finally:
        conn.close()

    job = ss.job
    if job is not None and job.active:
        st.info("Analysis is running. Progress is in the sidebar. Finished games stay saved if you cancel.")
        return
    if not pending:
        st.success(f"All {analyzed} games for {player} are analyzed.")
        if st.button("View results", type="primary", icon=":material/arrow_forward:"):
            ss.dataset = "user"
            st.switch_page(PAGES["dashboard"])
        return

    engine = find_stockfish()
    if engine is None:
        st.error("Stockfish is not available on this machine. Game results still work; "
                 "move quality will not until Stockfish is installed (see the README) or STOCKFISH_PATH is set.")
        return

    ui.note("Next: Stockfish scores every move. Then the dashboard can show centipawn loss and mistakes.")
    max_n = min(config.MAX_ANALYSIS_GAMES, len(pending))
    c1, c2 = st.columns(2)
    depth = c1.slider("Engine depth", config.MIN_DEPTH, config.MAX_DEPTH, config.DEFAULT_DEPTH,
                      key="analysis_depth",
                      help="Higher depth is slower and a bit stronger. 8-10 is a practical balance.")
    default_n = min(25, max_n)
    if ss.get("analysis_games", default_n) > max_n:
        ss.analysis_games = max_n
    n = max_n if max_n == 1 else c2.slider(
        "Games this run (most recent first)", 1, max_n, default_n, key="analysis_games",
    )
    est = n * 75 * 0.035 * (1.45 ** (depth - 10))
    ui.note(f"Roughly {max(5, est):.0f} seconds. Positions already cached for this engine and depth are free.")
    if len(pending) > config.MAX_ANALYSIS_GAMES:
        ui.note(f"This server analyzes at most {config.MAX_ANALYSIS_GAMES} games per run. Run again for the rest.")
    if st.button(f"Analyze {n} game{'s' if n != 1 else ''}", type="primary", icon=":material/bolt:",
                 key="start_analysis"):
        if ss.job is not None and ss.job.active:
            st.warning("An analysis is already running.")
            return
        ss.job = AnalysisJob(str(user_db_path()), pending[:n], depth, player=player).start()
        ss.dataset = "user"
        ss.import_messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def page_dashboard() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Dashboard", "How you are performing, and where the costly moves are.")
    view = st.segmented_control(
        "Dashboard view", ["Overview", "Trends", "Error patterns", "Openings"],
        default="Overview", key="dash_view", label_visibility="collapsed",
    )
    show_context(player, f, view)
    ov = an.overview(conn, player, f)
    if ov["games"] == 0:
        empty_filters()
        return
    if view == "Overview":
        view_overview(conn, player, f, ov)
    elif view == "Trends":
        view_trends(conn, player, f, ov)
    elif view == "Error patterns":
        view_errors(conn, player, f, ov)
    else:
        view_openings(conn, player, f, ov)


def view_overview(conn, player, f, ov) -> None:
    analyzed = ov["analyzed_games"] > 0
    ui.stat_cards([
        ("Games analyzed", f"{ov['analyzed_games']}", f"of {ov['games']} games in this view"),
        ("Score", ui.fmt_pct(ov["avg_score"], 1), "1 for a win, 0.5 for a draw, 0 for a loss", "accent"),
        ("Average centipawn loss", ui.fmt_num(ov["acpl"]) if analyzed else "-",
         "this player's moves; lower is better" if analyzed else "run analysis to compute this"),
    ], cols=3)
    if st.button("Review your biggest mistakes", type="primary", icon=":material/search:"):
        st.switch_page(PAGES["mistakes"])

    games = an.games_table(conn, player, f)
    n = max(ov["games"], 1)
    with st.expander("Result details and error rates"):
        ui.note("Win, draw and loss shares are of all games in this view, including draws. "
                "Win rate ignores draws. Score counts a draw as half a point.")
        ui.stat_cards([
            ("Wins", ui.fmt_pct(ov["wins"] / n, 1), f"{ov['wins']} games"),
            ("Draws", ui.fmt_pct(ov["draws"] / n, 1), f"{ov['draws']} games"),
            ("Losses", ui.fmt_pct(ov["losses"] / n, 1), f"{ov['losses']} games"),
            ("Win rate", ui.fmt_pct(ov["wins"] / max(ov["wins"] + ov["losses"], 1), 1),
             "wins divided by wins plus losses"),
        ])
        if analyzed:
            ui.note("A centipawn is one hundredth of a pawn. Average centipawn loss is how much "
                    "the engine's evaluation dropped, on average, after this player's moves. "
                    "Approximate accuracy uses the Lichess win-probability formula; it is not a "
                    "percentage of good moves. Counts below are this player's moves only.")
            ui.stat_cards([
                ("Blunder rate", ui.fmt_pct(ov["blunder_rate"]), "200+ centipawns, this player's moves", "warn"),
                ("Mistake rate", ui.fmt_pct(ov["mistake_rate"]), "100-199 centipawns"),
                ("Inaccuracy rate", ui.fmt_pct(ov["inaccuracy_rate"]), "50-99 centipawns"),
                ("Approximate accuracy", ui.fmt_num(ov["accuracy"], 1), f"{ov['moves']:,} moves by this player"),
            ])
        else:
            ui.note("Error rates appear after Stockfish analyzes these games.")

    colors = an.by_color(conn, player, f)
    if not colors.empty:
        st.plotly_chart(charts.color_results(colors), width="stretch")
        tbl = as_pct(colors, ["avg_score", "win_rate", "blunder_rate"])
        tbl["Color"] = tbl["color"].str.title()
        st.dataframe(
            tbl[["Color", "games", "avg_score", "win_rate", "acpl", "blunder_rate"]],
            hide_index=True, width="stretch",
            column_config={
                "games": "Games",
                "avg_score": pct_col("Score"),
                "win_rate": pct_col("Win rate (draws excluded)"),
                "acpl": st.column_config.NumberColumn("Avg centipawn loss", format="%.0f"),
                "blunder_rate": pct_col("Blunder rate"),
            },
        )

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
            "acpl": st.column_config.NumberColumn("Avg centipawn loss", format="%.0f"),
            "blunders": st.column_config.NumberColumn("Blunders", format="%d"),
        },
    )


def view_trends(conn, player, f, ov) -> None:
    games = an.games_table(conn, player, f)
    if games["player_elo"].notna().sum() >= 2:
        st.plotly_chart(charts.rating_trend(games), width="stretch")
    else:
        ui.note("No rating chart: these games have no ratings.")
    tr = an.trend(conn, player, f)
    if len(tr) >= 2:
        st.plotly_chart(charts.score_by_period(tr, tr.attrs.get("granularity", "period")), width="stretch")
    else:
        ui.note("Score over time needs games in at least two periods.")
    if ov["analyzed_games"] == 0:
        ui.note("Centipawn loss over time appears after analysis.")
        return
    if games["acpl"].notna().sum() >= 3:
        st.plotly_chart(charts.rolling_acpl(games), width="stretch")
    else:
        ui.note("Not enough analyzed games to plot centipawn loss over time.")


def view_errors(conn, player, f, ov) -> None:
    if needs_analysis(ov):
        return
    ui.note("Rates are shares of this player's moves. Lower is better. "
            "A phase with few moves is a weak comparison.")
    ph = an.phase_stats(conn, player, f)
    st.plotly_chart(charts.phase_error_rates(ph), width="stretch")
    small = ph[ph["moves"] < config.MIN_MOVES_FOR_PHASE_CLAIM]
    if not small.empty:
        st.warning(f"Small sample: {', '.join(small['phase'])} has fewer than "
                   f"{config.MIN_MOVES_FOR_PHASE_CLAIM} of this player's moves.")
    with st.expander("How phases are classified"):
        st.markdown(
            "- **Endgame**: at most 6 queens, rooks, bishops and knights remain (both sides).\n"
            "- **Opening**: move 12 or earlier, and at least 11 of those pieces remain.\n"
            "- **Middlegame**: everything else."
        )

    profile = an.move_number_profile(conn, player, f)
    if not profile.empty:
        st.plotly_chart(charts.move_number_profile(profile), width="stretch")
    ctx = an.eval_context_stats(conn, player, f)
    if not ctx.empty:
        st.plotly_chart(charts.context_bars(ctx), width="stretch")

    tc = an.time_class_stats(conn, player, f)
    if not tc.empty:
        st.markdown("#### Time controls")
        show = as_pct(tc, ["avg_score", "serious_rate"])
        show["time_class"] = show["time_class"].str.title()
        st.dataframe(
            show[["time_class", "games", "avg_score", "acpl", "serious_rate"]],
            hide_index=True, width="stretch",
            column_config={
                "time_class": "Time control", "games": "Games",
                "avg_score": pct_col("Score"),
                "acpl": st.column_config.NumberColumn("Avg centipawn loss", format="%.0f"),
                "serious_rate": pct_col("Mistake + blunder rate"),
            },
        )
    clock = an.time_pressure_stats(conn, player, f)
    if set(clock["clock"]) == {"low_clock", "normal"}:
        lo = clock.set_index("clock").loc["low_clock"]
        no = clock.set_index("clock").loc["normal"]
        ui.stat_cards([
            ("Low-clock error rate", ui.fmt_pct(lo["serious_rate"]),
             f"{int(lo['moves'])} moves with under 10% of the starting time", "warn"),
            ("Normal-clock error rate", ui.fmt_pct(no["serious_rate"]), f"{int(no['moves'])} moves"),
        ], cols=2)
    else:
        ui.note("Clock data is not in these games, so there is no time-pressure comparison.")


def view_openings(conn, player, f, ov) -> None:
    df = an.opening_stats(conn, player, f)
    if df.empty:
        empty_filters()
        return
    ui.note("Score counts a draw as half a point. Win rate is wins divided by wins plus losses, "
            "so draws are left out. A low score means worse results in that opening, not that "
            "the opening caused them.")
    min_games = st.slider(
        "Minimum games before an opening is compared", 1, 10, config.MIN_GAMES_FOR_OPENING_CLAIM,
        help="Openings below this stay in the table but drop out of the comparison chart.",
        key=f"opening_min_{ss.dataset}",
    )
    table = df.copy()
    ci = table.apply(lambda r: an.wilson_interval(r["wins"], int(r["games"])), axis=1)
    table["win_ci"] = [f"{lo:.0%} – {hi:.0%}" for lo, hi in ci]
    table["sample"] = table["games"].map(
        lambda n: "Too few games" if n < min_games else ("Modest" if n < 10 else "Enough to compare"))
    show = as_pct(table, ["avg_score", "win_rate"])
    st.dataframe(
        show[["opening", "games", "wins", "draws", "losses", "avg_score", "win_rate", "win_ci",
              "acpl", "sample"]],
        hide_index=True, width="stretch",
        column_config={
            "opening": "Opening family", "games": "Games", "wins": "W", "draws": "D", "losses": "L",
            "avg_score": pct_col("Score"),
            "win_rate": pct_col("Win rate"),
            "win_ci": st.column_config.TextColumn("Win-rate range (95%)",
                                                  help="Wilson interval. Wide ranges mean the rate is uncertain."),
            "acpl": st.column_config.NumberColumn("Avg centipawn loss", format="%.0f"),
            "sample": "Sample",
        },
    )
    compared = df[df["games"] >= min_games]
    if compared.empty:
        ui.note("No opening has enough games at this minimum. Lower it to see a comparison.")
    else:
        overall = ov["avg_score"]
        overall = 0.0 if overall is None or overall != overall else float(overall)
        st.plotly_chart(charts.opening_score_bars(compared, overall), width="stretch")
    with st.expander("Score against centipawn loss"):
        ui.note("Each point is an opening with at least the minimum number of games. "
                "This is a comparison, not a cause.")
        if df["acpl"].notna().any() and not compared.empty:
            st.plotly_chart(charts.opening_scatter(df, min_games), width="stretch")
        else:
            ui.note("This chart needs analyzed games.")


# ---------------------------------------------------------------------------
# Review mistakes
# ---------------------------------------------------------------------------


def _shift_mistake(move_ids: list[int], delta: int) -> None:
    """Previous/Next share the selectbox value, so the list does not jump back on rerun."""
    nxt = step_move(move_ids, ss.get("mistake_menu"), delta)
    ss.mistake_menu = nxt
    ss.mistake_pick = nxt


def page_mistakes() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Review mistakes", "The move you played, the engine move, and what it cost.")
    show_context(player, f)
    ov = an.overview(conn, player, f)
    if needs_analysis(ov):
        return

    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    cats = c1.multiselect("Categories", ["blunder", "mistake", "inaccuracy"],
                          default=["blunder", "mistake"], format_func=str.title, key="mistake_cats")
    phases = c2.multiselect("Phases", ["opening", "middlegame", "endgame"], format_func=str.title,
                            placeholder="All phases", key="mistake_phases")
    order = c3.selectbox("Sort by", ["Biggest loss", "Most recent"], key="mistake_order")
    if not cats:
        st.info("Select at least one category.")
        return
    phases = [p for p in phases if p in ("opening", "middlegame", "endgame")]
    where, params = "", []
    if phases:
        where = f"AND pm.phase IN ({','.join('?' * len(phases))})"
        params = phases
    limit = 1000
    df = an.mistakes(conn, player, f, categories=cats, extra_where=where, extra_params=params, limit=limit)
    if df.empty:
        ui.empty_state("No matching mistakes", "Nothing in these categories for the current filters.")
        return
    if order == "Most recent":
        df = df.sort_values(["played_at", "ply"], ascending=[False, True]).reset_index(drop=True)
    if len(df) == limit:
        ui.note(f"Showing the {limit} largest losses.")

    move_ids = [int(x) for x in df["move_id"]]
    requested = ss.get("explorer_move_id")
    if requested in move_ids:
        ss.mistake_menu = requested
    elif ss.get("mistake_menu") not in move_ids:
        ss.mistake_menu = pick_move(move_ids, None, ss.get("mistake_pick"))
    ss.explorer_move_id = None
    ss.mistake_pick = ss.mistake_menu
    idx = move_ids.index(ss.mistake_menu)
    row = df.iloc[idx]

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    nav1.button("Previous", icon=":material/chevron_left:", width="stretch", disabled=idx == 0,
                on_click=_shift_mistake, args=(move_ids, -1), key="mistake_prev")
    nav3.button("Next", icon=":material/chevron_right:", width="stretch", disabled=idx >= len(df) - 1,
                on_click=_shift_mistake, args=(move_ids, 1), key="mistake_next")
    labels = {
        int(r.move_id): (
            f"{r.move_number}{'.' if r.color == 'white' else '...'}{r.san}  "
            f"({r.classification}, {int(r.cpl)} cp) vs {r.opponent}"
        )
        for r in df.itertuples()
    }
    picked = nav2.selectbox(
        f"Mistake {idx + 1} of {len(df)}", move_ids, index=idx, format_func=labels.get, key="mistake_menu",
    )
    if int(picked) != int(row["move_id"]):
        ss.mistake_pick = int(picked)
        st.rerun()

    move_label = f"{row['move_number']}{'.' if row['color'] == 'white' else '...'}{row['san']}"
    best = row["best_move_san"] or "the same move"
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        ui.render_board(
            ui.board_svg(row["fen_before"], row["uci"], row["best_move_uci"], row["color"]),
            label=(f"Board from {row['color']}'s side, before {move_label}. "
                   f"Solid arrow: {row['san']}. Dashed arrow: {best}."),
        )
        st.markdown(ui.board_legend(), unsafe_allow_html=True)
        ui.note(f"Board is shown from {row['color']}'s side, the side you played.")
    with right:
        st.markdown(
            f'<div class="cl-move-title">{ui.esc(move_label)} {ui.class_pill(row["classification"])}</div>'
            f'<div class="cl-move-sub">Evaluations are from your side of the board, before and after this move.</div>',
            unsafe_allow_html=True,
        )
        ui.stat_cards([
            ("Played", row["san"], "solid arrow"),
            ("Engine move", row["best_move_san"] or "same", "dashed arrow"),
            ("Centipawns lost", f"{int(row['cpl'])}", "lower would have been better", "warn"),
        ], cols=3)
        ui.kv_list([
            ("Eval before", ui.fmt_eval(row["eval_before"])),
            ("Eval after", ui.fmt_eval(row["eval_after"])),
            ("Phase", str(row["phase"]).title()),
        ])

    with st.expander("Game details"):
        result_text = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}.get(row["score"], "Unfinished")
        ui.kv_list([
            ("Opponent", f"{row['opponent']} ({int(row['opponent_elo']) if pd.notna(row['opponent_elo']) else '?'})"),
            ("You played", f"{row['color'].title()} ({int(row['player_elo']) if pd.notna(row['player_elo']) else '?'})"),
            ("Date", str(row["played_at"])[:16] if row["played_at"] else "-"),
            ("Opening", row["opening"] or row["opening_family"]),
            ("Time control", f"{row['time_control'] or '?'} ({row['time_class']})"),
            ("Result", result_text),
        ])
        site = row["site"]
        if isinstance(site, str) and site.startswith("https://"):
            st.link_button("Open original game", site, icon=":material/open_in_new:")
    game = an.game_analysis(conn, int(row["game_id"]))
    if not game.empty:
        with st.expander("Evaluation through the game"):
            ui.note("The line is the engine evaluation from your side. The marked ply is this move.")
            st.plotly_chart(charts.game_eval_curve(game, int(row["ply"]), row["color"]), width="stretch")


# ---------------------------------------------------------------------------
# Practice
# ---------------------------------------------------------------------------


def page_practice() -> None:
    data = require_data()
    if not data:
        return
    conn, player, f = data
    ui.page_header("Practice", "One priority, based on your own games.")
    show_context(player, f)
    ov = an.overview(conn, player, f)
    recs, note = recommendations.generate(conn, player, f)
    if not recs:
        ui.empty_state("Not enough analyzed games yet",
                       note or "Analyze at least a few games before asking for a practice priority.")
    else:
        top = recs[0]
        with st.container(border=True):
            st.markdown(
                f'<div class="cl-rec"><div class="cl-rec-head"><span class="cl-rec-num">1</span>'
                f'<span class="cl-rec-title">{ui.esc(top.title)}</span>'
                f'{ui.confidence_badge(top.confidence)}</div>'
                f'<div class="evidence">{ui.esc(top.evidence)}</div>'
                f'<div class="advice"><b>Practice:</b> {ui.esc(top.advice)}</div></div>',
                unsafe_allow_html=True,
            )
            ui.note(f"Based on {top.sample_size:,} {sample_unit(top.area)}. "
                    "This is a rule comparing your own numbers, not a detected tactic.")
            if top.examples:
                ex = top.examples[0]
                if st.button("Open this position", type="primary", icon=":material/visibility:",
                             key="open_top_example"):
                    ss.explorer_move_id = ex["move_id"]
                    st.switch_page(PAGES["mistakes"])
                ui.note(ex["label"])

        if len(recs) > 1:
            st.markdown("#### Other priorities")
            for i, rec in enumerate(recs[1:], 2):
                with st.container(border=True):
                    st.markdown(
                        f'<div class="cl-rec"><div class="cl-rec-head"><span class="cl-rec-num">{i}</span>'
                        f'<span class="cl-rec-title">{ui.esc(rec.title)}</span>'
                        f'{ui.confidence_badge(rec.confidence)}</div>'
                        f'<div class="evidence">{ui.esc(rec.evidence)}</div>'
                        f'<div class="advice"><b>Practice:</b> {ui.esc(rec.advice)}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if rec.examples and st.button(
                        "Open example", key=f"open_ex_{i}", icon=":material/visibility:",
                    ):
                        ss.explorer_move_id = rec.examples[0]["move_id"]
                        st.switch_page(PAGES["mistakes"])
        if note:
            ui.note(note)
        ui.note("Recommendations compare phases, colours, openings, time controls and evaluations "
                "that ChessLens measures. They do not detect forks, pins or other motifs.")

    with st.expander("Export"):
        summary = reporting.coaching_summary(
            player, ov, an.phase_stats(conn, player, f), an.opening_stats(conn, player, f), recs, note,
            dataset_label(),
        )
        safe = "".join(ch for ch in player if ch.isalnum() or ch in "-_") or "player"
        c1, c2, c3 = st.columns(3)
        c1.download_button("Coaching summary (.md)", summary, file_name=f"chesslens_{safe}_coaching.md",
                           mime="text/markdown", icon=":material/description:", width="stretch")
        c2.download_button("This player's moves (.csv)", reporting.to_csv_bytes(an.move_rows(conn, player, f)),
                           file_name=f"chesslens_{safe}_moves.csv", mime="text/csv",
                           icon=":material/table:", width="stretch")
        c3.download_button("Games (.csv)", reporting.to_csv_bytes(an.games_table(conn, player, f)),
                           file_name=f"chesslens_{safe}_games.csv", mime="text/csv",
                           icon=":material/table_chart:", width="stretch")
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
    ui.page_header(
        "ML insights",
        "The model ranks positions by the risk of a serious error. It does not guarantee a mistake "
        "and does not understand specific tactics.",
    )
    m = ml.load_metrics(config.ML_METRICS_PATH)
    if m is None:
        ui.empty_state("Model not trained yet", "Run python scripts/prepare_demo.py to train it.")
        return
    sel = m["models"][m["selected_model"]]["test"]
    base = m["models"]["Dummy (stratified)"]["test"]
    ui.note("The numbers in this first block are from the held-out test games "
            f"({m['n_test_games']} games, {m['n_test_positions']:,} positions). "
            "Average precision and ROC AUC measure ranking quality. They are not accuracy.")
    ui.stat_cards([
        ("Test games", f"{m['n_test_games']}", f"{m['n_train_games']} games were used only for training"),
        ("Serious-error rate", ui.fmt_pct(m["positive_rate_test"]), "share of test moves that lost 100+ cp"),
        ("Average precision", f"{sel['average_precision']:.3f}",
         f"{m['selected_model']}; baseline {base['average_precision']:.3f}", "accent"),
        ("ROC AUC", f"{sel['roc_auc']:.3f}", "0.5 is no better than chance"),
    ])
    st.markdown(
        f"The selected model is **{m['selected_model']}**, chosen by {m['selection_rule'].lower()} "
        "on the training games only. Features describe the board before the move. "
        "Class-balanced scores are not calibrated probabilities."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.model_comparison(m["models"]), width="stretch")
    with c2:
        st.plotly_chart(charts.pr_curves(m["pr_curves"], m["positive_rate_test"]), width="stretch")
    rows = []
    for name, r in m["models"].items():
        t = r["test"]
        rows.append({
            "Model": name,
            "CV average precision": f"{r['cv_average_precision_mean']:.3f} ± {r['cv_average_precision_std']:.3f}",
            "Test average precision": t["average_precision"],
            "ROC AUC": t["roc_auc"],
            "Precision at 0.5": t["precision"],
            "Recall at 0.5": t["recall"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ("Test average precision", "ROC AUC",
                                          "Precision at 0.5", "Recall at 0.5")})
    ui.note("Precision and recall use a 0.5 cutoff on class-balanced scores, so they flag many moves. "
            "Average precision and ROC AUC do not use that cutoff.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.calibration_bars(m["calibration_deciles"]), width="stretch")
    with c2:
        st.plotly_chart(charts.importance_bars(
            m["permutation_importance"], "importance",
            f"Permutation importance ({m['selected_model']}, drop in average precision)"), width="stretch")
    ui.note("The bars compare the model's score with the error rate actually observed. "
            "A higher score than the observed rate means the score is not a probability.")

    ext = m.get("external_evaluation")
    if ext:
        st.markdown("#### Check on the demo games")
        ui.note(ext["description"] + ".")
        ui.stat_cards([
            ("Positions", f"{ext['n_positions']:,}", f"{ext['n_games']} games, both players' moves"),
            ("Average precision", f"{ext['average_precision']:.3f}", "ranking quality, not accuracy"),
            ("ROC AUC", f"{ext['roc_auc']:.3f}", f"base rate {ext['positive_rate']:.1%}"),
        ], cols=3)

    st.markdown("#### Selected player's own moves")
    conn, player = CTX.get("conn"), CTX.get("player")
    if conn is None or player is None:
        ui.note("Load a dataset to score that player's moves. This section is separate from the demo-game check above.")
    else:
        ui.note(f"These figures use only {player}'s analyzed moves in the active dataset, not both sides.")
        frame = build_ml_frame(ml.load_ml_moves(conn, player))
        model = load_risk_model()
        if model is None or len(frame) < 50 or frame[ml.TARGET_COLUMN].nunique() < 2:
            ui.note("Need at least 50 analyzed moves by this player, including some mistakes.")
        else:
            frame["risk"] = ml.predict_risk(model, frame)
            frame["bucket"] = pd.qcut(frame["risk"].rank(method="first"), 5,
                                      labels=["Lowest", "Low", "Medium", "High", "Highest"])
            buckets = (frame.groupby("bucket", observed=True)
                       .agg(mean_predicted=("risk", "mean"), observed_rate=(ml.TARGET_COLUMN, "mean"),
                            positions=("risk", "size")).reset_index())
            st.plotly_chart(
                charts.risk_bucket_bars(buckets, f"Risk ranking on {player}'s moves"),
                width="stretch",
            )
            from sklearn.metrics import average_precision_score, roc_auc_score
            ap = average_precision_score(frame[ml.TARGET_COLUMN], frame["risk"])
            auc = roc_auc_score(frame[ml.TARGET_COLUMN], frame["risk"])
            ui.note(f"{player}: {len(frame):,} of this player's moves, "
                    f"serious-error rate {frame[ml.TARGET_COLUMN].mean():.1%}, "
                    f"average precision {ap:.3f}, ROC AUC {auc:.3f}. "
                    "The scores rank risk. They are not probabilities.")

    st.markdown("#### Limitations")
    st.markdown("\n".join(f"- {x}" for x in m["limitations"]))


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------


def page_about() -> None:
    ui.page_header("Methodology", "How the numbers on the dashboard are defined.")
    st.markdown(
        f"""
**What the product does.** You import games or open the demo, Stockfish scores the moves, and the
dashboard, mistake review and practice page answer how you scored, where the costly moves were,
and what is worth practicing.

**Centipawn loss, in short.** After your move, the engine's evaluation of the position usually
changes. Centipawn loss is how far that evaluation moved against you, in hundredths of a pawn.
Zero means you played the engine's move, or the evaluation did not get worse. Lower is better.
The average on the dashboard is over the selected player's moves only, not both players.

**Score, win rate and approximate accuracy** are different things:
- **Score** is the average game result: 1 for a win, 0.5 for a draw, 0 for a loss.
- **Win rate** is wins divided by wins plus losses. Draws are excluded.
- **Win, draw and loss shares** on the dashboard are shares of all games, including draws.
- **Approximate accuracy** converts the evaluation change into the published Lichess
  win-probability formula and averages it per move. It is not the share of moves that were good.

**Formula.** For a move by the selected player, `eval_before` and `eval_after` are the engine
scores before and after the move, from that player's point of view, clamped to
±{config.EVAL_CAP_CP} centipawns. `CPL = max(0, eval_before − eval_after)`. Mate scores are first
converted to ±({config.MATE_SCORE_CP} − plies to mate), then clamped, so a missed mate is at most
a {2 * config.EVAL_CAP_CP} centipawn swing.

**Categories** are project-defined, not official Chess.com or Lichess labels:
good < {config.INACCURACY_CP} cp ≤ inaccuracy < {config.MISTAKE_CP} cp ≤ mistake < {config.BLUNDER_CP} cp ≤ blunder.

**Import checks.** Each game is replayed. Illegal moves, unreadable move text, unsupported variants,
invalid starting positions and games with no moves are rejected and counted, not shortened into a
legal game. A game repeated in the same file is kept once. A game already stored is not inserted
again. Missing dates, ratings and opening names are kept as unknown and called out in the import
summary. Files that are not text, or that are larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB,
are refused.

**Engine cache.** A cached evaluation is reused only when the position, the search depth and the
engine settings match. Settings recorded are the engine name plus the threads and hash that were
actually applied. An older cache that did not record the engine is discarded. The position key
ignores the halfmove clock, so two positions that differ only by that counter share one entry.

**Privacy.** Imported games go into a temporary SQLite file tied to this browser session. Other
visitors cannot open it. Refreshing the page starts a new session, so the previous import is no
longer shown. A cleanup that runs at most once an hour, and only while the app is being used,
deletes session files that have not been touched for {SESSION_MAX_AGE_S // 3600} hours. The demo
database is opened read-only and is never mixed with imported games. The person who operates the
server can still read files on that server.

**Practice rules** compare slices of your own games (phase, colour, opening, time control,
evaluation context) and require minimum sample sizes. They do not detect tactical motifs.

**Model.** Whole games are held out for evaluation, and the model is chosen using training games
only. It ranks the risk of a mistake or blunder. Class-balanced scores are not probabilities.
Labels come from a depth-{config.DEFAULT_DEPTH} search of 2013 games.
        """
    )
    if CTX.get("conn") is not None:
        st.markdown("**Tables in the database you have open**")
        st.dataframe(pd.DataFrame([db.table_counts(CTX["conn"])]), hide_index=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

init_state()
cleanup_old_sessions()
if ss.dataset is None and st.query_params.get("demo") == "1":
    go_demo()

PAGES = {
    "start": st.Page(page_start, title="Start", icon=":material/home:", url_path="start", default=True),
    "dashboard": st.Page(page_dashboard, title="Dashboard", icon=":material/dashboard:",
                         url_path="dashboard"),
    "mistakes": st.Page(page_mistakes, title="Review mistakes", icon=":material/search:",
                        url_path="mistakes"),
    "practice": st.Page(page_practice, title="Practice", icon=":material/school:", url_path="practice"),
    "ml": st.Page(page_ml, title="ML insights", icon=":material/psychology:", url_path="ml"),
    "about": st.Page(page_about, title="Methodology", icon=":material/info:", url_path="methodology"),
}

nav = st.navigation({
    "": [PAGES["start"], PAGES["dashboard"], PAGES["mistakes"], PAGES["practice"]],
    "Technical": [PAGES["ml"], PAGES["about"]],
}, position="top")

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
