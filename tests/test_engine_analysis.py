import chess
import chess.engine
import pytest

from src import config
from src.engine_analysis import (
    PositionEval,
    StockfishAnalyzer,
    build_move_evals,
    centipawn_loss,
    clamp_eval,
    classify_cpl,
    find_stockfish,
    move_accuracy,
    position_key,
    score_to_cp,
    terminal_eval,
    to_mover_pov,
    win_probability,
)

START = chess.STARTING_FEN


@pytest.mark.parametrize(
    "cpl, expected",
    [(0, "good"), (49, "good"), (50, "inaccuracy"), (99, "inaccuracy"), (100, "mistake"),
     (199, "mistake"), (200, "blunder"), (1500, "blunder")],
)
def test_classification_thresholds(cpl, expected):
    assert classify_cpl(cpl) == expected


def test_mover_perspective():
    assert to_mover_pov(150, chess.WHITE) == 150
    assert to_mover_pov(150, chess.BLACK) == -150


def test_centipawn_loss_never_negative_and_clamped():
    assert centipawn_loss(100, 40) == 60
    assert centipawn_loss(40, 100) == 0  # improving on the engine's estimate is not a loss
    assert centipawn_loss(5000, -5000) == 2 * config.EVAL_CAP_CP
    assert clamp_eval(-99999) == -config.EVAL_CAP_CP


def test_mate_scores_are_converted_and_bounded():
    assert score_to_cp(chess.engine.Mate(3)) == config.MATE_SCORE_CP - 3
    assert score_to_cp(chess.engine.Mate(-2)) == -(config.MATE_SCORE_CP - 2)
    assert score_to_cp(chess.engine.Cp(-45)) == -45
    # Winning with mate vs winning with +12 pawns: both clamp to the cap, so no loss
    assert centipawn_loss(score_to_cp(chess.engine.Mate(2)), 1200) == 0
    # Allowing a forced mate from equality is a maximal blunder
    loss = centipawn_loss(0, score_to_cp(chess.engine.Mate(-1)))
    assert loss == config.EVAL_CAP_CP and classify_cpl(loss) == "blunder"


def test_terminal_positions():
    mated = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        mated.push_san(san)
    ev = terminal_eval(mated)
    assert ev.score_cp == -config.MATE_SCORE_CP  # White (to move) is mated
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal_eval(stalemate).score_cp == 0
    assert terminal_eval(chess.Board()) is None


def _fens_and_moves(sans):
    board = chess.Board()
    fens, ucis = [], []
    for san in sans:
        fens.append(board.fen())
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
    return fens, ucis


def test_build_move_evals_uses_each_movers_perspective():
    fens, ucis = _fens_and_moves(["e4", "e5", "Qh5"])
    # White POV evaluations of the 4 positions (before e4, before e5, before Qh5, after Qh5)
    evals = [
        PositionEval(30, None, "e2e4"),   # white plays the best move
        PositionEval(30, None, "c7c5"),   # black plays e5 instead of c5
        PositionEval(-150, None, "g1f3"), # after e5 black is better by 1.5 (white POV -150)
        PositionEval(-400, None, None),   # Qh5 drops a lot (white POV -400)
    ]
    out = build_move_evals(fens, ucis, evals)
    assert out[0].cpl == 0 and out[0].classification == "good"
    # Black: before = -30 (black POV), after = +150 (black POV) -> improved, no loss
    assert out[1].eval_before == -30 and out[1].eval_after == 150 and out[1].cpl == 0
    # White: before = -150, after = -400 -> 250 cp lost
    assert out[2].eval_before == -150 and out[2].eval_after == -400
    assert out[2].cpl == 250 and out[2].classification == "blunder"
    assert out[2].best_move_san == "Nf3"


def test_black_blunder_detected_from_black_perspective():
    fens, ucis = _fens_and_moves(["e4", "f6"])
    evals = [PositionEval(30, None, "e2e4"), PositionEval(40, None, "e7e5"), PositionEval(160, None, None)]
    out = build_move_evals(fens, ucis, evals)
    assert out[1].eval_before == -40 and out[1].eval_after == -160
    assert out[1].cpl == 120 and out[1].classification == "mistake"


def test_build_move_evals_requires_n_plus_one_positions():
    fens, ucis = _fens_and_moves(["e4"])
    with pytest.raises(ValueError):
        build_move_evals(fens, ucis, [PositionEval(0, None, None)])


def test_win_probability_and_accuracy_are_monotone():
    assert win_probability(0) == pytest.approx(50.0)
    assert win_probability(300) > win_probability(100) > 50 > win_probability(-100)
    assert move_accuracy(60, 60) == pytest.approx(100.0, abs=0.01)
    assert move_accuracy(60, 20) < move_accuracy(60, 50)


def test_position_key_ignores_move_counters():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 5 30")
    assert position_key(a) == position_key(b)


# ---------------------------------------------------------------------------
# Integration tests with the real engine (skipped when Stockfish is unavailable)
# ---------------------------------------------------------------------------

needs_engine = pytest.mark.skipif(find_stockfish() is None, reason="Stockfish not installed")


@needs_engine
def test_engine_finds_mate_in_one_and_flags_missing_it():
    # White to move: Qxf7# is mate (scholar's mate pattern)
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    with StockfishAnalyzer(depth=8) as sf:
        ev = sf.evaluate(chess.Board(fen))
        assert ev.best_move_uci == "h5f7"
        assert ev.mate == 1
        missed = sf.analyze_moves([fen], ["h5h4"])[0]
        assert missed.classification == "blunder"
        found = sf.analyze_moves([fen], ["h5f7"])[0]
        assert found.cpl == 0


@needs_engine
def test_engine_cache_avoids_repeat_analysis():
    store = {}
    with StockfishAnalyzer(
        depth=6,
        cache_get=lambda k, d: store.get((k, d)),
        cache_put=lambda k, d, e: store.__setitem__((k, d), e),
    ) as sf:
        sf.evaluate(chess.Board())
        sf.evaluate(chess.Board())
        assert sf.engine_calls == 1 and sf.cache_hits == 1
    with StockfishAnalyzer(depth=6, cache_get=lambda k, d: store.get((k, d))) as sf2:
        sf2.evaluate(chess.Board())
        assert sf2.engine_calls == 0  # served from the persistent cache
