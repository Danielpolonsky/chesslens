import math
import sqlite3

import pandas as pd
import pytest

from src import analytics as an
from src import database as db
from src.engine_analysis import PositionEval
from src.ingest import parse_pgn_text
from tests.conftest import SCHOLARS_MATE


def test_schema_and_insert(tmp_db, sample_pgn):
    conn, _ = tmp_db
    summary = db.insert_games(conn, parse_pgn_text(sample_pgn).games)
    assert summary.inserted == 3 and summary.duplicates == 0
    counts = db.table_counts(conn)
    assert counts["games"] == 3
    assert counts["players"] == 3  # Alice, Bob, Carol
    assert counts["moves"] == 7 + 6 + 8


def test_duplicate_games_are_skipped(tmp_db, sample_pgn):
    conn, _ = tmp_db
    games = parse_pgn_text(sample_pgn).games
    db.insert_games(conn, games)
    again = db.insert_games(conn, games)
    assert again.inserted == 0 and again.duplicates == 3
    assert db.table_counts(conn)["games"] == 3


def test_player_names_are_case_insensitive(tmp_db):
    conn, _ = tmp_db
    a = db.get_or_create_player(conn, "Alice")
    b = db.get_or_create_player(conn, "alice")
    assert a == b


def test_view_scores_from_each_players_perspective(tmp_db, sample_pgn):
    conn, _ = tmp_db
    db.insert_games(conn, parse_pgn_text(sample_pgn).games)
    df = db.query_df(
        conn,
        """SELECT p.username, v.color, v.score FROM v_player_games v
           JOIN players p ON p.player_id = v.player_id ORDER BY v.game_id, v.color DESC""",
    )
    rows = list(df.itertuples(index=False, name=None))
    assert rows[:2] == [("Alice", "white", 1.0), ("Bob", "black", 0.0)]
    assert ("Alice", "black", 0.5) in rows
    assert ("Carol", "black", 1.0) in rows


def test_overview_results_and_error_rates(analyzed_db):
    conn, _ = analyzed_db
    ov = an.overview(conn, "Alice")
    assert (ov["games"], ov["wins"], ov["draws"], ov["losses"]) == (3, 1, 1, 1)
    assert ov["avg_score"] == pytest.approx(0.5)
    # Alice's moves: game1 white plies 1,3,5,7 (ply 3 blunder); game2 black plies 2,4,6
    # (ply 4 mistake); game3 white plies 1,3,5,7 (ply 3 blunder) -> 11 moves
    assert ov["moves"] == 11
    assert ov["blunder_rate"] == pytest.approx(2 / 11)
    assert ov["mistake_rate"] == pytest.approx(1 / 11)
    assert ov["acpl"] == pytest.approx((250 * 2 + 120) / 11)


def test_by_color_group_by(analyzed_db):
    conn, _ = analyzed_db
    df = an.by_color(conn, "Alice").set_index("color")
    assert df.loc["white", "games"] == 2 and df.loc["black", "games"] == 1
    assert df.loc["white", "avg_score"] == pytest.approx(0.5)
    assert df.loc["white", "blunder_rate"] == pytest.approx(2 / 8)
    assert df.loc["black", "serious_rate"] == pytest.approx(1 / 3)


def test_filters_restrict_rows(analyzed_db):
    conn, _ = analyzed_db
    assert an.overview(conn, "Alice", an.Filters(color="black"))["games"] == 1
    assert an.overview(conn, "Alice", an.Filters(time_classes=("bullet",)))["games"] == 1
    assert an.overview(conn, "Alice", an.Filters(date_from="2024-03-06"))["games"] == 2
    assert an.overview(conn, "Alice", an.Filters(openings=("Sicilian Defense",)))["losses"] == 1


def test_opening_stats(analyzed_db):
    conn, _ = analyzed_db
    df = an.opening_stats(conn, "Alice").set_index("opening")
    assert set(df.index) == {"King's Pawn Game", "Scandinavian Defense", "Sicilian Defense"}
    assert df.loc["Sicilian Defense", "losses"] == 1
    assert bool(df.loc["Sicilian Defense", "small_sample"]) is True


def test_phase_stats_are_normalized_per_move(analyzed_db):
    conn, _ = analyzed_db
    df = an.phase_stats(conn, "Alice")
    assert df["moves"].sum() == 11
    assert (df["serious_rate"] == df["serious_errors"] / df["moves"]).all()


def test_mistakes_query_returns_context(analyzed_db):
    conn, _ = analyzed_db
    df = an.mistakes(conn, "Alice")
    assert len(df) == 3
    assert df.iloc[0]["cpl"] == 250
    assert {"fen_before", "best_move_uci", "opponent", "opening"} <= set(df.columns)


def test_empty_dataset_is_safe(tmp_db):
    conn, _ = tmp_db
    ov = an.overview(conn, "nobody")
    assert ov["games"] == 0 and ov["moves"] == 0
    assert ov["acpl"] is None or (isinstance(ov["acpl"], float) and math.isnan(ov["acpl"]))
    assert an.by_color(conn, "nobody").empty
    assert an.opening_stats(conn, "nobody").empty
    assert an.phase_stats(conn, "nobody").empty
    assert an.games_table(conn, "nobody").empty
    assert an.trend(conn, "nobody").empty
    assert an.filter_options(conn, "nobody")["date_min"] is None


def test_unanalyzed_games_and_cache(tmp_db):
    conn, _ = tmp_db
    db.insert_games(conn, parse_pgn_text(SCHOLARS_MATE).games)
    assert db.unanalyzed_games(conn, "alice", 10) == [1]
    engine = "Stockfish 19|hash=64|threads=1"
    db.cache_put(conn, "key", 12, engine, PositionEval(35, None, "e2e4"))
    assert db.cache_get(conn, "key", 12, engine).score_cp == 35
    assert db.cache_get(conn, "key", 10, engine) is None  # a different depth is a different result
    assert db.cache_get(conn, "key", 12, "other-engine") is None


def test_wilson_interval_and_z():
    lo, hi = an.wilson_interval(5, 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert an.wilson_interval(0, 3)[0] == 0.0
    assert an.two_proportion_z(30, 100, 10, 100) > 3
    assert an.two_proportion_z(10, 100, 10, 100) == 0
    assert an.two_proportion_z(1, 0, 1, 1) == 0.0


def test_legacy_cache_without_engine_is_dropped(tmp_path):
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.execute(
        """CREATE TABLE position_evals (
               position_key TEXT, depth INTEGER, score_cp INTEGER, mate INTEGER,
               best_move_uci TEXT, PRIMARY KEY (position_key, depth))"""
    )
    raw.execute("INSERT INTO position_evals VALUES ('k', 10, 35, NULL, 'e2e4')")
    raw.commit()
    raw.close()
    conn = db.connect(path)
    assert db.cache_get(conn, "k", 10, "Stockfish 19|hash=64|threads=1") is None
    cols = {row[1] for row in conn.execute("PRAGMA table_info(position_evals)")}
    assert "engine" in cols
    conn.close()


def test_list_players_counts(analyzed_db):
    conn, _ = analyzed_db
    players = db.list_players(conn)
    assert players.iloc[0]["username"] == "Alice"
    assert int(players.iloc[0]["games"]) == 3 and int(players.iloc[0]["analyzed"]) == 3
    assert isinstance(players, pd.DataFrame)
