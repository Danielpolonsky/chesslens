from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database as db  # noqa: E402
from src.engine_analysis import MoveEval, classify_cpl  # noqa: E402
from src.ingest import parse_pgn_text  # noqa: E402

SCHOLARS_MATE = """[Event "Club game"]
[Site "?"]
[Date "2024.03.05"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1400"]
[TimeControl "300+2"]
[ECO "C20"]
[Opening "King's Pawn Game: Wayward Queen Attack"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

CHESSCOM_DRAW = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.03.06"]
[White "Bob"]
[Black "Alice"]
[Result "1/2-1/2"]
[WhiteElo "1410"]
[BlackElo "1510"]
[TimeControl "600"]
[ECO "B01"]
[ECOUrl "https://www.chess.com/openings/Scandinavian-Defense-Mieses-Kotroc-Variation-3.Nc3"]
[UTCDate "2024.03.06"]
[UTCTime "18:00:00"]

1. e4 {[%clk 0:09:58]} d5 {[%clk 0:09:55]} 2. exd5 {[%clk 0:09:50]} Qxd5 {[%clk 0:09:40]}
3. Nc3 {[%clk 0:09:45]} Qa5 {[%clk 0:09:30]} 1/2-1/2
"""

ALICE_LOSS = """[Event "Club game"]
[Date "2024.04.01"]
[White "Alice"]
[Black "Carol"]
[Result "0-1"]
[WhiteElo "1490"]
[BlackElo "1600"]
[TimeControl "60+0"]
[Opening "Sicilian Defense: Open"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 0-1
"""

ILLEGAL = """[White "X"]
[Black "Y"]
[Result "0-1"]

1. e4 e5 2. Ke3 Nc6 0-1
"""

NO_HEADERS = "1. d4 d5 2. c4 *\n"


@pytest.fixture
def sample_pgn() -> str:
    return "\n\n".join([SCHOLARS_MATE, CHESSCOM_DRAW, ALICE_LOSS])


@pytest.fixture
def tmp_db(tmp_path: Path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    yield conn, path
    conn.close()


def fake_move_eval(ply: int, cpl: int) -> MoveEval:
    return MoveEval(
        ply=ply, eval_before=0, eval_after=-cpl, best_move_uci=None, best_move_san=None,
        cpl=cpl, classification=classify_cpl(cpl), win_prob_before=50.0,
        win_prob_after=50.0, accuracy=100.0 if cpl == 0 else 50.0,
    )


@pytest.fixture
def analyzed_db(tmp_db, sample_pgn):
    """Three games with deterministic synthetic 'engine' results.

    Every move gets cpl 0 except ply 3 of each game (a white move) which gets 250 (blunder)
    and ply 4 (a black move) which gets 120 (mistake).
    """
    conn, path = tmp_db
    report = parse_pgn_text(sample_pgn)
    summary = db.insert_games(conn, report.games)
    for gid in summary.game_ids:
        rows = db.game_moves(conn, gid)
        evals = [fake_move_eval(r["ply"], {3: 250, 4: 120}.get(r["ply"], 0)) for r in rows]
        db.save_game_analysis(conn, gid, [r["move_id"] for r in rows], evals, 10, "fake")
    return conn, path
