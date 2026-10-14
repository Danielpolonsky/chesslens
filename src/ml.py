"""Serious-error risk model.

Question: given only what is visible on the board before a move (plus ratings and time
control), how likely is the player's next move to be a mistake or blunder
(centipawn loss >= 100)?

Methodology
-----------
* Features come exclusively from the pre-move position (see ``features.FEATURE_COLUMNS``).
  Engine evaluations and anything about the played move are excluded.
* Train/test split is done by **whole games** (``GroupShuffleSplit``) so positions from one
  game never appear on both sides. Model selection uses grouped 5-fold CV on the training
  games only; the held-out test games are scored once at the end.
* Class imbalance is handled with ``class_weight='balanced'`` and evaluated with
  precision / recall / F1 and average precision (area under the PR curve).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

MIN_POSITIVES = 50
MIN_GAMES = 40

LIMITATIONS = [
    "Labels come from a shallow Stockfish search (depth 10), so some 'errors' are engine noise "
    "and some real errors are missed.",
    "The 100-centipawn threshold is a project-defined cut-off, not an official classification.",
    "Training games come from one month (January 2013) of Lichess rated games; players, "
    "rating pools and time controls today differ, so the model may not transfer.",
    "Features are coarse, static board descriptors; the model knows nothing about concrete "
    "tactics, the player's clock (not recorded in 2013 data) or their history.",
    "Predicted risks are associations, not causes, and are only moderately better than the "
    "base rate - useful for ranking positions, not for confident individual predictions.",
]


ML_MOVES_SQL = """
SELECT m.game_id, m.move_number, m.color, m.phase, m.material_balance, m.total_material,
       m.legal_moves, m.in_check, m.own_vulnerable, m.opp_vulnerable, m.captures_available,
       m.checks_available, a.cpl, g.white_elo, g.black_elo, g.time_control
FROM moves m
JOIN move_analysis a ON a.move_id = m.move_id
JOIN games g ON g.game_id = m.game_id
"""


def load_ml_moves(conn, player: str | None = None) -> pd.DataFrame:
    """Analysed moves with pre-move features. With ``player``, only that player's moves."""
    from .database import query_df

    if player is None:
        return query_df(conn, ML_MOVES_SQL)
    return query_df(
        conn,
        ML_MOVES_SQL
        + """
        JOIN v_player_games vpg ON vpg.game_id = m.game_id AND vpg.color = m.color
        JOIN players p ON p.player_id = vpg.player_id
        WHERE p.username = ?""",
        [player],
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "Dummy (stratified)": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", DummyClassifier(strategy="stratified", random_state=seed))]
        ),
        "Logistic regression": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))]
        ),
        "Random forest": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", RandomForestClassifier(
                 n_estimators=300, max_depth=10, min_samples_leaf=25,
                 class_weight="balanced_subsample", n_jobs=-1, random_state=seed))]
        ),
    }


def grouped_split(frame: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Split row indices so that every game lands entirely in train or test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame[GROUP_COLUMN]))
    train_games = set(frame.iloc[train_idx][GROUP_COLUMN])
    test_games = set(frame.iloc[test_idx][GROUP_COLUMN])
    assert not train_games & test_games, "game leakage between train and test"
    return train_idx, test_idx


def check_no_leakage(columns) -> None:
    bad = LEAKAGE_COLUMNS & set(columns)
    if bad:
        raise ValueError(f"Post-move / engine columns must not be features: {sorted(bad)}")


def threshold_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true)
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else float("nan"),
        "flagged_share": float(pred.mean()),
    }
    k = max(1, int(round(0.1 * len(proba))))
    top = np.argsort(-np.asarray(proba), kind="stable")[:k]
    out["top_decile_error_rate"] = float(y_true[top].mean())
    out["lift_top_decile"] = float(y_true[top].mean() / max(y_true.mean(), 1e-9))
    return out


@dataclass
class TrainingResult:
    metrics: dict
    model: Pipeline


def train_and_evaluate(frame: pd.DataFrame, seed: int = 42, test_size: float = 0.25,
                       cv_folds: int = 5, external: pd.DataFrame | None = None) -> TrainingResult:
    check_no_leakage(FEATURE_COLUMNS)
    n_games = frame[GROUP_COLUMN].nunique()
    n_pos = int(frame[TARGET_COLUMN].sum())
    if n_games < MIN_GAMES or n_pos < MIN_POSITIVES:
        raise ValueError(
            f"Not enough data for a meaningful evaluation ({n_games} games, {n_pos} positive "
            f"examples; need >= {MIN_GAMES} games and >= {MIN_POSITIVES} positives)."
        )

    train_idx, test_idx = grouped_split(frame, test_size, seed)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN].to_numpy()
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].to_numpy()

    models = candidate_models(seed)
    gkf = GroupKFold(n_splits=cv_folds)
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in models.items():
        cv_ap = []
        for tr, va in gkf.split(X_train, y_train, groups=train[GROUP_COLUMN]):
            p = candidate_models(seed)[name].fit(X_train.iloc[tr], y_train[tr])
            cv_ap.append(average_precision_score(y_train[va], p.predict_proba(X_train.iloc[va])[:, 1]))
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_average_precision_mean": float(np.mean(cv_ap)),
            "cv_average_precision_std": float(np.std(cv_ap)),
            "test": threshold_metrics(y_test, proba),
        }

    learned = {k: v for k, v in results.items() if not k.startswith("Dummy")}
    selected = max(learned, key=lambda k: learned[k]["cv_average_precision_mean"])
    best = fitted[selected]
    best_proba = best.predict_proba(X_test)[:, 1]

    pr_curves = {}
    for name in learned:
        pp, rr, _ = precision_recall_curve(y_test, fitted[name].predict_proba(X_test)[:, 1])
        s = max(1, len(pp) // 200)
        pr_curves[name] = {"precision": pp[::s].round(4).tolist(), "recall": rr[::s].round(4).tolist()}

    # Observed error rate by predicted-risk decile on the held-out games (calibration view).
    deciles = pd.qcut(pd.Series(best_proba).rank(method="first"), 10, labels=False)
    calib = (
        pd.DataFrame({"decile": deciles + 1, "pred": best_proba, "obs": y_test})
        .groupby("decile").agg(mean_predicted=("pred", "mean"), observed_rate=("obs", "mean"),
                               positions=("obs", "size")).reset_index()
    )

    # Feature analysis: LR coefficients (standardized) and permutation importance.
    lr = fitted["Logistic regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    coef_table = sorted(
        ({"feature": n.split("__", 1)[-1], "coefficient": float(c)} for n, c in zip(names, coefs)),
        key=lambda d: abs(d["coefficient"]), reverse=True,
    )
    perm = permutation_importance(
        best, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=1
    )
    perm_table = sorted(
        ({"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(FEATURE_COLUMNS, perm.importances_mean, perm.importances_std)),
        key=lambda d: d["importance"], reverse=True,
    )

    metrics = {
        "target": "Next move is a mistake or blunder (centipawn loss >= 100)",
        "selected_model": selected,
        "selection_rule": "Highest mean grouped-CV average precision on training games",
        "n_positions": int(len(frame)),
        "n_games": int(n_games),
        "n_train_games": int(train[GROUP_COLUMN].nunique()),
        "n_test_games": int(test[GROUP_COLUMN].nunique()),
        "n_train_positions": int(len(train)),
        "n_test_positions": int(len(test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "models": results,
        "pr_curves": pr_curves,
        "calibration_deciles": calib.to_dict(orient="records"),
        "logistic_coefficients": coef_table,
        "permutation_importance": perm_table,
        "features": FEATURE_COLUMNS,
        "limitations": LIMITATIONS,
    }
    if external is not None and len(external) and external[TARGET_COLUMN].nunique() > 1:
        ext_proba = best.predict_proba(external[FEATURE_COLUMNS])[:, 1]
        metrics["external_evaluation"] = {
            "description": ("Every position in the demo games, moves by the demo player and by "
                            "their opponents (games never used in training or model selection)"),
            "n_positions": int(len(external)),
            "n_games": int(external[GROUP_COLUMN].nunique()),
            "positive_rate": float(external[TARGET_COLUMN].mean()),
            **threshold_metrics(external[TARGET_COLUMN].to_numpy(), ext_proba),
        }
    return TrainingResult(metrics=metrics, model=best)


def save_artifacts(result: TrainingResult, metrics_path: Path, model_path: Path) -> None:
    import joblib

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    joblib.dump(result.model, model_path, compress=3)


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    import joblib

    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_risk(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
