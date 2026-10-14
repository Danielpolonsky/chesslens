from datetime import date

from src import analytics as an
from src.ui_state import describe_filters, pick_move, sample_unit, step_move


def test_pick_move_prefers_the_coaching_example():
    ids = [10, 20, 30]
    assert pick_move(ids, 20, 10) == 20
    assert pick_move(ids, 99, 30) == 30
    assert pick_move(ids, None, None) == 10
    assert pick_move([], 1, 1) is None


def test_step_move_stays_inside_the_list():
    ids = [10, 20, 30]
    assert step_move(ids, 20, 1) == 30
    assert step_move(ids, 30, 1) == 30
    assert step_move(ids, 10, -1) == 10
    assert step_move([], 1, 1) is None


def test_describe_filters_only_lists_real_narrowing():
    full = an.Filters(date_from="2013-01-02", date_to="2013-01-25")
    assert describe_filters(full, (date(2013, 1, 2), date(2013, 1, 25))) == []
    narrowed = an.Filters(
        color="white", time_classes=("blitz", "rapid"), openings=("Sicilian Defense", "French Defense"),
        date_from="2013-01-10", date_to="2013-01-25",
    )
    parts = describe_filters(narrowed, (date(2013, 1, 2), date(2013, 1, 25)))
    assert parts == ["as White", "Blitz, Rapid", "2 openings", "2013-01-10 to 2013-01-25"]


def test_sample_unit_matches_the_rule():
    assert sample_unit("phase:opening") == "moves"
    assert sample_unit("context:time") == "moves"
    assert sample_unit("opening:Sicilian") == "games"
    assert sample_unit("color:black") == "games"
