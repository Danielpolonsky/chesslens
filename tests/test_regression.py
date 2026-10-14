"""Reference checks against the pre-analysed demo and the coaching hand-off."""

import pytest

from src import analytics as an
from src import config, recommendations
from src import database as db
from src.ui_state import pick_move


@pytest.fixture
def demo_conn():
    if not config.DEMO_DB_PATH.exists():
        pytest.skip("demo database not built")
    conn = db.connect(config.DEMO_DB_PATH, read_only=True)
    yield conn
    conn.close()


def test_demo_totals_match_the_reviewed_dataset(demo_conn):
    player = "sani5057"
    ov = an.overview(demo_conn, player)
    assert ov["games"] == 158
    assert (ov["wins"], ov["draws"], ov["losses"]) == (65, 6, 87)
    assert ov["moves"] == 4744
    assert ov["acpl"] == pytest.approx(59.719, abs=0.001)
    assert ov["avg_score"] == pytest.approx((65 + 0.5 * 6) / 158)
    counts = dict(demo_conn.execute(
        """SELECT a.classification, COUNT(*)
           FROM move_analysis a
           JOIN moves m ON m.move_id = a.move_id
           JOIN games g ON g.game_id = m.game_id
           JOIN players p ON p.player_id = CASE m.color WHEN 'white' THEN g.white_id ELSE g.black_id END
           WHERE p.username = ?
           GROUP BY a.classification""",
        (player,),
    ))
    assert counts["blunder"] == 382
    assert counts["mistake"] == 442
    assert counts["inaccuracy"] == 534
    # Move rows are the selected player, not both sides of every game.
    assert len(an.move_rows(demo_conn, player)) == 4744
    assert demo_conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0] == 9513


def test_demo_color_partitions(demo_conn):
    colors = an.by_color(demo_conn, "sani5057").set_index("color")
    assert int(colors.loc["white", "games"]) == 64
    assert int(colors.loc["black", "games"]) == 94
    assert int(colors.loc["white", "moves"]) == 1926
    assert int(colors.loc["black", "moves"]) == 2818
    assert int(colors["games"].sum()) == 158
    assert int(colors["moves"].sum()) == 4744


def test_demo_filters_narrow_games_and_keep_draws_in_the_score(demo_conn):
    player = "sani5057"
    full = an.overview(demo_conn, player)
    opts = an.filter_options(demo_conn, player)
    time_class = opts["time_classes"][0]
    opening = opts["openings"][0]
    by_time = an.overview(demo_conn, player, an.Filters(time_classes=(time_class,)))
    by_opening = an.overview(demo_conn, player, an.Filters(openings=(opening,)))
    assert 0 < by_time["games"] < full["games"]
    assert 0 < by_opening["games"] < full["games"]
    midpoint = "2013-01-15"
    later = an.overview(demo_conn, player, an.Filters(date_from=midpoint))
    assert 0 < later["games"] < full["games"]
    # A draw is half a point, and it is included in the game count behind the score.
    assert full["decided_games"] == full["games"]
    assert full["avg_score"] == pytest.approx(
        (full["wins"] + 0.5 * full["draws"]) / full["games"]
    )


def test_coaching_example_resolves_to_that_players_mistake(demo_conn):
    recs, note = recommendations.generate(demo_conn, "sani5057")
    assert recs, note
    move_id = recs[0].examples[0]["move_id"]
    listed = an.mistakes(
        demo_conn, "sani5057", categories=("blunder", "mistake", "inaccuracy"), limit=5000,
    )
    ids = [int(x) for x in listed["move_id"]]
    assert move_id in ids
    assert pick_move(ids, move_id, ids[0]) == move_id
    row = listed.set_index("move_id").loc[move_id]
    assert row["san"] in recs[0].examples[0]["label"]


def test_small_sample_blocks_recommendations(analyzed_db):
    conn, _ = analyzed_db
    recs, note = recommendations.generate(conn, "Alice")
    assert recs == []
    assert note and "3" in note
