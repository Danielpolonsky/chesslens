import pytest
import requests

from src import ingest
from src.ingest import (
    ChessComError,
    candidate_players,
    classify_time_control,
    fetch_chesscom_games,
    opening_family,
    opening_from_eco_url,
    parse_elo,
    parse_pgn_date,
    parse_pgn_text,
)
from tests.conftest import ALICE_LOSS, CHESSCOM_DRAW, ILLEGAL, NO_HEADERS, SCHOLARS_MATE


def test_parse_single_game_metadata():
    report = parse_pgn_text(SCHOLARS_MATE)
    assert len(report.games) == 1 and not report.errors
    g = report.games[0]
    assert (g.white, g.black) == ("Alice", "Bob")
    assert (g.white_elo, g.black_elo) == (1500, 1400)
    assert g.result == "1-0"
    assert g.played_at == "2024-03-05"
    assert g.time_control == "300+2" and g.time_class == "blitz"
    assert g.eco == "C20"
    assert g.opening_family == "King's Pawn Game"
    assert g.num_plies == 7
    assert [m.san for m in g.moves][-1] == "Qxf7#"


def test_move_colors_and_numbers():
    g = parse_pgn_text(SCHOLARS_MATE).games[0]
    assert [m.color for m in g.moves[:3]] == ["white", "black", "white"]
    assert [m.move_number for m in g.moves[:4]] == [1, 1, 2, 2]
    assert g.moves[0].fen_before.startswith("rnbqkbnr/pppppppp")


def test_multiple_games_and_chesscom_headers(sample_pgn):
    report = parse_pgn_text(sample_pgn)
    assert len(report.games) == 3
    draw = report.games[1]
    assert draw.result == "1/2-1/2"
    assert draw.opening == "Scandinavian Defense Mieses Kotroc Variation"
    assert draw.opening_family == "Scandinavian Defense"
    assert draw.played_at == "2024-03-06 18:00:00"
    assert draw.time_class == "rapid"
    # %clk comments: clock before Black's 2nd move is Black's clock after move 1 (9:55)
    assert draw.moves[3].clock_before_s == pytest.approx(595.0)
    # Games without %clk data get no clock values at all
    assert all(m.clock_before_s is None for m in report.games[0].moves)


def test_illegal_move_game_is_skipped_not_raised():
    report = parse_pgn_text(ILLEGAL + "\n\n" + SCHOLARS_MATE)
    assert len(report.games) == 1
    assert report.skipped == 1
    assert any("illegal" in e.lower() for e in report.errors)


@pytest.mark.parametrize("text", ["", "   \n", "this is not a chess file", "<html>oops</html>"])
def test_invalid_files_return_errors(text):
    report = parse_pgn_text(text)
    assert report.games == []
    assert report.errors


def test_missing_metadata_is_handled():
    g = parse_pgn_text(NO_HEADERS).games[0]
    assert g.white == "Unknown" and g.black == "Unknown"
    assert g.result == "*"
    assert g.played_at is None
    assert g.white_elo is None
    assert g.time_class == "unknown"
    assert g.opening_family == "Unknown"


def test_variant_games_are_skipped():
    pgn = '[Variant "Chess960"]\n[FEN "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1"]\n[SetUp "1"]\n\n1. e4 *'
    report = parse_pgn_text(pgn)
    assert report.games == [] and report.skipped == 1


def test_max_games_truncation(sample_pgn):
    report = parse_pgn_text(sample_pgn, max_games=2)
    assert len(report.games) == 2 and report.truncated


def test_game_hash_is_stable_and_distinct(sample_pgn):
    a = parse_pgn_text(sample_pgn).games
    b = parse_pgn_text(sample_pgn).games
    assert [g.game_hash for g in a] == [g.game_hash for g in b]
    assert len({g.game_hash for g in a}) == 3


@pytest.mark.parametrize(
    "tc, expected",
    [("60+0", "bullet"), ("180+2", "blitz"), ("300+0", "blitz"), ("600", "rapid"),
     ("900+10", "rapid"), ("1800+0", "classical"), ("1/86400", "daily"), ("-", "daily"),
     (None, "unknown"), ("abc", "unknown")],
)
def test_time_control_classes(tc, expected):
    assert classify_time_control(tc) == expected


def test_header_cleaning_helpers():
    assert parse_elo("1500") == 1500
    assert parse_elo("?") is None and parse_elo("") is None and parse_elo("99999") is None
    assert parse_pgn_date("2013.01.05") == "2013-01-05"
    assert parse_pgn_date("????.??.??") is None
    assert parse_pgn_date("2013.02.30") is None
    assert opening_from_eco_url("https://www.chess.com/openings/Italian-Game-Two-Knights-Defense-4.d3") == \
        "Italian Game Two Knights Defense"
    assert opening_family("Sicilian Defense: Najdorf Variation") == "Sicilian Defense"
    assert opening_family("Queens Pawn Opening Zukertort Variation") == "Queens Pawn Opening"
    assert opening_family(None, "B12") == "ECO B"
    assert opening_family(None, None) == "Unknown"


def test_candidate_players_orders_by_frequency(sample_pgn):
    players = candidate_players(parse_pgn_text(sample_pgn).games)
    assert players[0] == ("Alice", 3)


# ---------------------------------------------------------------------------
# Chess.com client with a fake HTTP session
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes = {k: list(v) if isinstance(v, list) else [v] for k, v in routes.items()}
        self.calls = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append((url, timeout, headers))
        queue = self.routes.get(url)
        if not queue:
            return FakeResponse(404)
        return queue.pop(0) if len(queue) > 1 else queue[0]


ARCHIVES = "https://api.chess.com/pub/player/alice/games/archives"
M1 = "https://api.chess.com/pub/player/alice/games/2024/03"
M2 = "https://api.chess.com/pub/player/alice/games/2024/04"


def _month(games):
    return FakeResponse(200, {"games": games})


def test_chesscom_fetch_most_recent_first_and_filters_variants():
    session = FakeSession({
        ARCHIVES: FakeResponse(200, {"archives": [M1, M2]}),
        M2: _month([
            {"pgn": ALICE_LOSS, "rules": "chess", "end_time": 1711929600, "time_class": "bullet",
             "url": "https://www.chess.com/game/live/1"},
            {"pgn": SCHOLARS_MATE, "rules": "chess960", "end_time": 1711929700},
        ]),
        M1: _month([{"pgn": CHESSCOM_DRAW, "rules": "chess", "end_time": 1709740000, "time_class": "rapid"}]),
    })
    report = fetch_chesscom_games("Alice", max_games=2, session=session, sleep=lambda s: None)
    assert [g.result for g in report.games] == ["1/2-1/2", "0-1"]  # oldest first
    assert report.games[1].site == "https://www.chess.com/game/live/1"
    assert all(call[1] is not None for call in session.calls)  # every request has a timeout
    assert "ChessLens" in session.calls[0][2]["User-Agent"]


def test_chesscom_respects_max_games():
    session = FakeSession({
        ARCHIVES: FakeResponse(200, {"archives": [M1, M2]}),
        M2: _month([{"pgn": ALICE_LOSS, "rules": "chess", "end_time": 2},
                    {"pgn": SCHOLARS_MATE, "rules": "chess", "end_time": 1}]),
    })
    report = fetch_chesscom_games("alice", max_games=1, session=session, sleep=lambda s: None)
    assert len(report.games) == 1
    assert [c[0] for c in session.calls] == [ARCHIVES, M2]  # stopped before older month


def test_chesscom_user_not_found():
    with pytest.raises(ChessComError, match="not found"):
        fetch_chesscom_games("alice", session=FakeSession({}), sleep=lambda s: None)


def test_chesscom_invalid_username():
    with pytest.raises(ChessComError, match="valid"):
        fetch_chesscom_games("bad name!", session=FakeSession({}))


def test_chesscom_rate_limit_retries_then_succeeds():
    sleeps = []
    session = FakeSession({
        ARCHIVES: [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, {"archives": [M1]})],
        M1: _month([{"pgn": SCHOLARS_MATE, "rules": "chess", "end_time": 1}]),
    })
    report = fetch_chesscom_games("alice", max_games=5, session=session, sleep=sleeps.append)
    assert len(report.games) == 1 and sleeps == [1.0]


def test_chesscom_persistent_rate_limit_raises():
    session = FakeSession({ARCHIVES: FakeResponse(429)})
    with pytest.raises(ChessComError, match="rate limit"):
        fetch_chesscom_games("alice", session=session, sleep=lambda s: None)


def test_chesscom_timeout_is_reported():
    class TimeoutSession:
        def get(self, *a, **k):
            raise requests.Timeout()

    with pytest.raises(ChessComError, match="respond in time"):
        fetch_chesscom_games("alice", session=TimeoutSession())


def test_chesscom_max_games_is_capped():
    games = [{"pgn": SCHOLARS_MATE, "rules": "chess", "end_time": i} for i in range(150)]
    session = FakeSession({ARCHIVES: FakeResponse(200, {"archives": [M1]}), M1: _month(games)})
    report = fetch_chesscom_games("alice", max_games=10_000, session=session, sleep=lambda s: None)
    assert len(report.games) == ingest.config.CHESSCOM_MAX_GAMES == 100


def test_chesscom_account_without_games():
    session = FakeSession({ARCHIVES: FakeResponse(200, {"archives": []})})
    with pytest.raises(ChessComError, match="no public games"):
        fetch_chesscom_games("alice", session=session)
