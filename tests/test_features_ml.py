import chess
import numpy as np
import pandas as pd
import pytest

from src import ml
from src.features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_ml_frame,
    classify_phase,
    material_balance,
    parse_time_control,
    position_features,
    vulnerable_pieces,
)


def test_phase_classification():
    assert classify_phase(chess.Board()) == "opening"
    endgame = chess.Board("8/5k2/8/3r4/8/2R5/5K2/8 w - - 0 45")
    assert classify_phase(endgame) == "endgame"
    late_full = chess.Board()
    late_full.fullmove_number = 20
    assert classify_phase(late_full) == "middlegame"


def test_material_balance_from_movers_view():
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert material_balance(board, chess.WHITE) == 9
    assert material_balance(board, chess.BLACK) == -9


def test_vulnerable_pieces():
    # White knight on e5 attacked by black pawn on d6 (cheaper attacker) and undefended rook a1? no
    board = chess.Board("4k3/8/3p4/4N3/8/8/8/R3K3 w - - 0 1")
    assert vulnerable_pieces(board, chess.WHITE) == 1
    assert vulnerable_pieces(board, chess.BLACK) == 0


def test_position_features_keys_and_values():
    feats = position_features(chess.Board())
    assert feats["legal_moves"] == 20
    assert feats["in_check"] == 0
    assert feats["material_balance"] == 0
    assert feats["total_material"] == 78
    assert feats["phase"] == "opening"


def test_parse_time_control():
    assert parse_time_control("300+2") == (300.0, 2.0)
    assert parse_time_control("600") == (600.0, 0.0)
    assert all(np.isnan(x) for x in parse_time_control("1/86400"))
    assert all(np.isnan(x) for x in parse_time_control(None))


def _moves_frame(n_games=60, plies=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        w, b = rng.integers(1200, 2000, size=2)
        for ply in range(1, plies + 1):
            legal = int(rng.integers(5, 45))
            vuln = int(rng.integers(0, 3))
            # risk rises with vulnerable pieces -> a learnable signal
            cpl = int(rng.random() < 0.08 + 0.15 * vuln) * 300
            rows.append({
                "game_id": g, "move_number": (ply + 1) // 2,
                "color": "white" if ply % 2 else "black",
                "phase": "opening" if ply < 12 else "middlegame",
                "material_balance": int(rng.integers(-3, 4)), "total_material": 70,
                "legal_moves": legal, "in_check": 0, "own_vulnerable": vuln, "opp_vulnerable": 0,
                "captures_available": 2, "checks_available": 1, "cpl": cpl,
                "white_elo": w, "black_elo": b, "time_control": "300+0",
            })
    return pd.DataFrame(rows)


def test_build_ml_frame_excludes_post_move_information():
    frame = build_ml_frame(_moves_frame(5))
    assert not (LEAKAGE_COLUMNS & set(FEATURE_COLUMNS))
    assert not (LEAKAGE_COLUMNS & set(frame.columns))
    assert set(frame[TARGET_COLUMN]) <= {0, 1}
    first = frame.iloc[0]
    assert first["is_white"] == 1 and first["elo_diff"] == first["player_elo"] - first["opponent_elo"]
    assert first["base_time_s"] == 300


def test_player_and_opponent_elo_swap_for_black():
    raw = _moves_frame(1, plies=2)
    frame = build_ml_frame(raw)
    assert frame.iloc[0]["player_elo"] == raw.iloc[0]["white_elo"]
    assert frame.iloc[1]["player_elo"] == raw.iloc[1]["black_elo"]


def test_target_threshold():
    raw = _moves_frame(2)
    frame = build_ml_frame(raw, serious_cp=100)
    assert frame[TARGET_COLUMN].sum() == (raw["cpl"] >= 100).sum()


def test_grouped_split_keeps_games_separate():
    frame = build_ml_frame(_moves_frame(50))
    tr, te = ml.grouped_split(frame, test_size=0.3, seed=1)
    train_games = set(frame.iloc[tr]["game_id"])
    test_games = set(frame.iloc[te]["game_id"])
    assert train_games and test_games
    assert not train_games & test_games
    assert len(tr) + len(te) == len(frame)


def test_check_no_leakage_rejects_engine_columns():
    with pytest.raises(ValueError):
        ml.check_no_leakage(FEATURE_COLUMNS + ["cpl"])


def test_train_and_evaluate_reports_real_held_out_metrics():
    frame = build_ml_frame(_moves_frame(80, seed=3))
    result = ml.train_and_evaluate(frame, seed=0, cv_folds=3)
    m = result.metrics
    assert m["n_train_games"] + m["n_test_games"] == 80
    assert set(m["models"]) == {"Dummy (stratified)", "Logistic regression", "Random forest"}
    for res in m["models"].values():
        t = res["test"]
        assert 0 <= t["precision"] <= 1 and 0 <= t["recall"] <= 1 and 0 <= t["average_precision"] <= 1
    lr = m["models"]["Logistic regression"]["test"]["average_precision"]
    dummy = m["models"]["Dummy (stratified)"]["test"]["average_precision"]
    assert lr > dummy  # the synthetic signal is learnable
    top_feature = m["logistic_coefficients"][0]["feature"]
    assert top_feature == "own_vulnerable"
    assert ml.predict_risk(result.model, frame).shape == (len(frame),)


def test_insufficient_data_is_refused():
    frame = build_ml_frame(_moves_frame(5))
    with pytest.raises(ValueError, match="Not enough data"):
        ml.train_and_evaluate(frame)
