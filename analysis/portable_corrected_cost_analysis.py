"""Portable leakage-controlled construction-cost analysis.

This is the standalone model-fitting code used for the corrected manuscript
analysis.  It needs only this file and ``Data_ML2.xlsx``; it does not import any
project-local modules.  Outputs are written as CSV/JSON task artifacts that can
be consumed directly by ``portable_corrected_cost_figures.py``.

Algorithms
----------
* 10 repeated 70/30 outer splits (seeds 42--51), with the outer test sealed.
* Training-only Mean, Q3, and Q3+IQR/2 cost thresholds.
* CatBoost, XGBoost, monotonic LightGBM, and a tuned single ANN.
* SVM cost-segment classification and two-stage segment-specific ANNs.
* Conservative, Balanced, and Aggressive transfer-learning strategies.
* Training-only cutoff/strategy selection and component-cost ablation.

Google Colab example
--------------------
Upload this file and Data_ML2.xlsx, then run::

    !pip -q install numpy pandas scipy scikit-learn tensorflow \
        catboost xgboost lightgbm openpyxl
    !python portable_corrected_cost_analysis.py \
        --data Data_ML2.xlsx --output-dir corrected_analysis_output

Use ``--smoke --seeds 42 --feature-sets A_all_21`` for a quick installation
check.  A full 3-feature-set x 10-seed run is computationally intensive.

Local Python example
--------------------
    python portable_corrected_cost_analysis.py --data Data_ML2.xlsx \
        --output-dir corrected_analysis_output

Expected input columns are C1, N1...N20, and target O.  Python 3.10+ is
recommended.  Tested reference versions are recorded in every run metadata
file rather than assumed from the host system.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp") / "mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor


SEEDS = tuple(range(42, 52))
TARGET = "O"
ALL_FEATURES = ["C1", *[f"N{i}" for i in range(1, 21)]]
COMPONENT_COST_FEATURES = {"N10", "N11", "N14", "N15"}
MONOTONE_FEATURES = {"N3", "N4", "N6", "N8", "N10", "N16"}

FEATURE_SETS: dict[str, list[str]] = {
    "A_all_21": ALL_FEATURES,
    "B_no_component_costs": [
        feature for feature in ALL_FEATURES if feature not in COMPONENT_COST_FEATURES
    ],
    "C_physical_quantity": [
        "C1",
        *[f"N{i}" for i in range(2, 10)],
        "N12",
        "N13",
        *[f"N{i}" for i in range(16, 21)],
    ],
}

CUTOFF_RULES = ("Mean", "Q3", "Q3+IQR/2")
TRANSFER_STRATEGIES: dict[str, dict[str, Any]] = {
    "Conservative": {"trainable_hidden": 0, "lr": 1e-3},
    "Balanced": {"trainable_hidden": 1, "lr": 1e-2},
    "Aggressive": {"trainable_hidden": "all", "lr": 5e-2},
}
TL_BASE_CONFIG: dict[str, Any] = {
    "layers": (128, 64),
    "activation": "relu",
    "lr": 1e-3,
    "batch_size": 32,
}
FULL_MAX_EPOCHS = 200
FULL_EARLY_STOPPING_PATIENCE = 20
SMOKE_MAX_EPOCHS = 40
SMOKE_EARLY_STOPPING_PATIENCE = 5

TREE_GRIDS: dict[str, dict[str, list[Any]]] = {
    "CatBoost": {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "iterations": [300, 600],
    },
    "XGBoost": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [300, 600],
    },
    "Monotonic GBM": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [300, 600],
    },
}

SVM_GRID = {
    "C": [0.1, 1, 10, 50, 100],
    "kernel": ["rbf", "linear"],
    "gamma": ["auto"],
}

ANN_GRID = {
    "layers": [(64,), (128,), (128, 64)],
    "activation": ["relu"],
    "lr": [1e-4, 1e-3, 1e-2, 5e-2],
    "batch_size": [32, 64],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=json_default, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_indices(indices: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tensorflow",
        "catboost",
        "xgboost",
        "lightgbm",
        "openpyxl",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def configure_tensorflow(threads: int) -> None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(max(1, threads))
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass


def set_random_state(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    expected = [*ALL_FEATURES, TARGET]
    if list(frame.columns) != expected:
        raise ValueError(
            "Unexpected columns. "
            f"Expected {expected}, received {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(f"Missing values detected: {missing[missing > 0].to_dict()}")
    if frame.duplicated().any():
        raise ValueError(f"Duplicate rows detected: {int(frame.duplicated().sum())}")
    frame = frame.copy()
    frame["C1"] = frame["C1"].astype(int)
    return frame


def calculate_thresholds(y_train: np.ndarray) -> dict[str, float]:
    q1 = float(np.quantile(y_train, 0.25))
    q3 = float(np.quantile(y_train, 0.75))
    return {
        "Mean": float(np.mean(y_train)),
        "Q3": q3,
        "Q3+IQR/2": q3 + ((q3 - q1) / 2.0),
    }


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric_features = [feature for feature in features if feature != "C1"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", MinMaxScaler(), numeric_features))
    if "C1" in features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                ["C1"],
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def monotone_vector(features: list[str]) -> list[int]:
    numeric_features = [feature for feature in features if feature != "C1"]
    result = [1 if feature in MONOTONE_FEATURES else 0 for feature in numeric_features]
    if "C1" in features:
        result.append(0)
    return result


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q3_threshold: float,
    q3_iqr_threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")

    def segment_rmse(threshold: float) -> tuple[float, int]:
        mask = y_true >= threshold
        count = int(mask.sum())
        if count == 0:
            return float("nan"), 0
        return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), count

    rmse_q3, n_q3 = segment_rmse(q3_threshold)
    rmse_q3_iqr, n_q3_iqr = segment_rmse(q3_iqr_threshold)
    denominator = np.maximum(np.abs(y_true), 1e-8)
    errors = y_pred - y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs(errors) / denominator) * 100.0),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmse_q3": rmse_q3,
        "rmse_q3_iqr2": rmse_q3_iqr,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mean_error": float(np.mean(errors)),
        "underestimation_rate": float(np.mean(y_pred < y_true)),
        "n_rmse_q3": n_q3,
        "n_rmse_q3_iqr2": n_q3_iqr,
    }


def classifier_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "classifier_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "classifier_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "classifier_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_ann(
    input_dim: int,
    layers: Iterable[int],
    activation: str,
    learning_rate: float,
) -> tf.keras.Model:
    model = tf.keras.Sequential(name="cost_ann")
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in layers:
        model.add(tf.keras.layers.Dense(int(units), activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
    )
    return model


def _train_validation_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_rows)
    if n_rows < 10:
        return indices, np.array([], dtype=int)
    validation_size = max(2, int(round(n_rows * 0.2)))
    validation_size = min(validation_size, n_rows - 2)
    train_indices, validation_indices = train_test_split(
        indices, test_size=validation_size, random_state=seed, shuffle=True
    )
    return np.asarray(train_indices), np.asarray(validation_indices)


def fit_ann_probe(
    X: np.ndarray,
    y_scaled: np.ndarray,
    config: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[tf.keras.Model, int]:
    set_random_state(seed)
    train_indices, validation_indices = _train_validation_indices(len(X), seed)
    model = build_ann(
        X.shape[1], config["layers"], config["activation"], config["lr"]
    )
    if len(validation_indices) == 0:
        epochs = min(max_epochs, 100)
        model.fit(
            X,
            y_scaled,
            epochs=epochs,
            batch_size=min(int(config["batch_size"]), len(X)),
            verbose=0,
            shuffle=True,
        )
        return model, epochs

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=min(patience, max_epochs - 1),
        min_delta=1e-7,
        restore_best_weights=True,
    )
    history = model.fit(
        X[train_indices],
        y_scaled[train_indices],
        validation_data=(X[validation_indices], y_scaled[validation_indices]),
        epochs=max_epochs,
        batch_size=min(int(config["batch_size"]), len(train_indices)),
        callbacks=[callback],
        verbose=0,
        shuffle=True,
    )
    validation_loss = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(validation_loss) + 1) if validation_loss else len(history.epoch)
    return model, max(1, best_epoch)


def fit_ann_final(
    X: np.ndarray,
    y_scaled: np.ndarray,
    config: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[tf.keras.Model, int]:
    probe, best_epoch = fit_ann_probe(
        X, y_scaled, config, seed, max_epochs=max_epochs, patience=patience
    )
    del probe
    gc.collect()
    set_random_state(seed)
    model = build_ann(
        X.shape[1], config["layers"], config["activation"], config["lr"]
    )
    model.fit(
        X,
        y_scaled,
        epochs=best_epoch,
        batch_size=min(int(config["batch_size"]), len(X)),
        verbose=0,
        shuffle=True,
    )
    return model, best_epoch


def configure_transfer_layers(
    model: tf.keras.Model, strategy_name: str
) -> list[str]:
    hidden_layers = model.layers[:-1]
    setting = TRANSFER_STRATEGIES[strategy_name]["trainable_hidden"]
    if setting == "all":
        trainable_hidden = len(hidden_layers)
    else:
        trainable_hidden = int(setting)
    frozen_count = max(0, len(hidden_layers) - trainable_hidden)
    frozen_names: list[str] = []
    for index, layer in enumerate(hidden_layers):
        layer.trainable = index >= frozen_count
        if not layer.trainable:
            frozen_names.append(layer.name)
    model.layers[-1].trainable = True
    return frozen_names


def clone_for_transfer(
    base_model: tf.keras.Model, strategy_name: str
) -> tuple[tf.keras.Model, list[str]]:
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    frozen_names = configure_transfer_layers(model, strategy_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(TRANSFER_STRATEGIES[strategy_name]["lr"])
        ),
        loss="mse",
    )
    return model, frozen_names


def fit_transfer_final(
    base_model: tf.keras.Model,
    X_high: np.ndarray,
    y_high_scaled: np.ndarray,
    strategy_name: str,
    batch_size: int,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[tf.keras.Model, int, list[str]]:
    set_random_state(seed)
    train_indices, validation_indices = _train_validation_indices(len(X_high), seed)
    probe, frozen_names = clone_for_transfer(base_model, strategy_name)
    if len(validation_indices) == 0:
        best_epoch = min(max_epochs, 100)
        probe.fit(
            X_high,
            y_high_scaled,
            epochs=best_epoch,
            batch_size=min(batch_size, len(X_high)),
            verbose=0,
            shuffle=True,
        )
    else:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=min(patience, max_epochs - 1),
            min_delta=1e-7,
            restore_best_weights=True,
        )
        history = probe.fit(
            X_high[train_indices],
            y_high_scaled[train_indices],
            validation_data=(
                X_high[validation_indices],
                y_high_scaled[validation_indices],
            ),
            epochs=max_epochs,
            batch_size=min(batch_size, len(train_indices)),
            callbacks=[callback],
            verbose=0,
            shuffle=True,
        )
        validation_loss = history.history.get("val_loss", [])
        best_epoch = (
            int(np.argmin(validation_loss) + 1)
            if validation_loss
            else len(history.epoch)
        )
    del probe
    gc.collect()

    set_random_state(seed)
    model, frozen_names = clone_for_transfer(base_model, strategy_name)
    model.fit(
        X_high,
        y_high_scaled,
        epochs=max(1, best_epoch),
        batch_size=min(batch_size, len(X_high)),
        verbose=0,
        shuffle=True,
    )
    return model, max(1, best_epoch), frozen_names


def inverse_target(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def ann_candidates(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "layers": (32, 16),
                "activation": "relu",
                "lr": 1e-3,
                "batch_size": 32,
            }
        ]
    candidates: list[dict[str, Any]] = []
    for layers, activation, learning_rate, batch_size in itertools.product(
        ANN_GRID["layers"],
        ANN_GRID["activation"],
        ANN_GRID["lr"],
        ANN_GRID["batch_size"],
    ):
        candidates.append(
            {
                "layers": tuple(layers),
                "activation": activation,
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
            }
        )
    return candidates


def tune_single_ann(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    seed: int,
    cv_folds: int,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for candidate_index, candidate in enumerate(ann_candidates(smoke)):
        fold_mae: list[float] = []
        fold_epochs: list[int] = []
        for fold, (development, validation) in enumerate(splitter.split(X_train)):
            preprocessor = make_preprocessor(features)
            X_development = preprocessor.fit_transform(X_train.iloc[development])
            X_validation = preprocessor.transform(X_train.iloc[validation])
            target_scaler = StandardScaler()
            y_development_scaled = target_scaler.fit_transform(
                y_train[development].reshape(-1, 1)
            ).reshape(-1)
            model, best_epoch = fit_ann_probe(
                np.asarray(X_development, dtype=np.float32),
                y_development_scaled.astype(np.float32),
                candidate,
                seed + candidate_index * 100 + fold,
                max_epochs=max_epochs,
                patience=patience,
            )
            prediction_scaled = model.predict(
                np.asarray(X_validation, dtype=np.float32), verbose=0
            ).reshape(-1)
            prediction = inverse_target(target_scaler, prediction_scaled)
            fold_mae.append(float(mean_absolute_error(y_train[validation], prediction)))
            fold_epochs.append(best_epoch)
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        rows.append(
            {
                "search_type": "ANN_grid",
                "candidate": json_dumps(candidate),
                "mean_validation_mae": float(np.mean(fold_mae)),
                "std_validation_mae": float(np.std(fold_mae, ddof=1))
                if len(fold_mae) > 1
                else 0.0,
                "mean_best_epoch": float(np.mean(fold_epochs)),
                "fold_scores": json_dumps(fold_mae),
            }
        )
    rows.sort(key=lambda row: (row["mean_validation_mae"], row["candidate"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
    best = json.loads(rows[0]["candidate"])
    best["layers"] = tuple(best["layers"])
    return best, rows


def tune_single_mlp(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    seed: int,
    cv_folds: int,
    grid_jobs: int,
    smoke: bool,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]], float, int]:
    regressor = MLPRegressor(
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.20,
        n_iter_no_change=40,
        random_state=seed,
        tol=1e-4,
        verbose=False,
    )
    target_scaled_regressor = TransformedTargetRegressor(
        regressor=regressor, transformer=StandardScaler()
    )
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            ("model", target_scaled_regressor),
        ]
    )
    if smoke:
        raw_grid = {
            "hidden_layer_sizes": [(32, 16)],
            "activation": ["relu"],
            "learning_rate_init": [1e-3],
            "batch_size": [32],
        }
    else:
        raw_grid = {
            "hidden_layer_sizes": ANN_GRID["layers"],
            "activation": ANN_GRID["activation"],
            "learning_rate_init": ANN_GRID["lr"],
            "batch_size": ANN_GRID["batch_size"],
        }
    parameter_grid = {
        f"model__regressor__{key}": values for key, values in raw_grid.items()
    }
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    started = time.perf_counter()
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=splitter,
        n_jobs=grid_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    best_params = {
        key.replace("model__regressor__", ""): value
        for key, value in search.best_params_.items()
    }
    rows: list[dict[str, Any]] = []
    results = pd.DataFrame(search.cv_results_)
    for _, result in results.iterrows():
        candidate = {
            key.replace("model__regressor__", ""): value
            for key, value in result["params"].items()
        }
        rows.append(
            {
                "search_type": "ANN_MLP_grid",
                "candidate": json_dumps(candidate),
                "mean_validation_mae": float(-result["mean_test_score"]),
                "std_validation_mae": float(result["std_test_score"]),
                "rank": int(result["rank_test_score"]),
                "selected": int(result["rank_test_score"]) == 1,
            }
        )
    fitted_regressor = search.best_estimator_.named_steps["model"].regressor_
    training_iterations = int(getattr(fitted_regressor, "n_iter_", 0))
    return search.best_estimator_, best_params, rows, elapsed, training_iterations


def make_tree_estimator(
    model_name: str,
    seed: int,
    features: list[str],
    model_threads: int,
) -> Any:
    if model_name == "CatBoost":
        return CatBoostRegressor(
            verbose=0,
            random_seed=seed,
            thread_count=max(1, model_threads),
            allow_writing_files=False,
            loss_function="RMSE",
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=seed,
            verbosity=0,
            n_jobs=max(1, model_threads),
            objective="reg:squarederror",
            tree_method="hist",
        )
    if model_name == "Monotonic GBM":
        return LGBMRegressor(
            monotone_constraints=monotone_vector(features),
            random_state=seed,
            verbosity=-1,
            n_jobs=max(1, model_threads),
        )
    raise KeyError(model_name)


def tune_tree_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    seed: int,
    cv_folds: int,
    grid_jobs: int,
    model_threads: int,
    smoke: bool,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]], float]:
    estimator = make_tree_estimator(model_name, seed, features, model_threads)
    pipeline = Pipeline(
        [("preprocess", make_preprocessor(features)), ("model", estimator)]
    )
    raw_grid = TREE_GRIDS[model_name]
    if smoke:
        raw_grid = {key: [values[0]] for key, values in raw_grid.items()}
    parameter_grid = {f"model__{key}": values for key, values in raw_grid.items()}
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    started = time.perf_counter()
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=splitter,
        n_jobs=grid_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    best_params = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    candidate_rows: list[dict[str, Any]] = []
    results = pd.DataFrame(search.cv_results_)
    for _, result in results.iterrows():
        candidate = {
            key.replace("model__", ""): value
            for key, value in result["params"].items()
        }
        candidate_rows.append(
            {
                "search_type": f"{model_name}_grid",
                "candidate": json_dumps(candidate),
                "mean_validation_mae": float(-result["mean_test_score"]),
                "std_validation_mae": float(result["std_test_score"]),
                "rank": int(result["rank_test_score"]),
                "selected": int(result["rank_test_score"]) == 1,
            }
        )
    return search.best_estimator_, best_params, candidate_rows, elapsed


def tune_svm(
    X_train: pd.DataFrame,
    y_train_class: np.ndarray,
    features: list[str],
    seed: int,
    cv_folds: int,
    grid_jobs: int,
    smoke: bool,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]], float, float]:
    pipeline = Pipeline(
        [("preprocess", make_preprocessor(features)), ("model", SVC())]
    )
    raw_grid = SVM_GRID
    if smoke:
        raw_grid = {"C": [1], "kernel": ["rbf"], "gamma": ["auto"]}
    parameter_grid = {f"model__{key}": value for key, value in raw_grid.items()}
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    started = time.perf_counter()
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1",
        cv=splitter,
        n_jobs=grid_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train_class)
    elapsed = time.perf_counter() - started
    best_params = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    candidate_rows: list[dict[str, Any]] = []
    results = pd.DataFrame(search.cv_results_)
    for _, result in results.iterrows():
        candidate = {
            key.replace("model__", ""): value
            for key, value in result["params"].items()
        }
        candidate_rows.append(
            {
                "search_type": "SVM_grid",
                "candidate": json_dumps(candidate),
                "mean_validation_f1": float(result["mean_test_score"]),
                "std_validation_f1": float(result["std_test_score"]),
                "rank": int(result["rank_test_score"]),
                "selected": int(result["rank_test_score"]) == 1,
            }
        )
    return (
        search.best_estimator_,
        best_params,
        candidate_rows,
        float(search.best_score_),
        elapsed,
    )


@dataclass
class TaskArtifacts:
    metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    split_indices: list[dict[str, Any]] = field(default_factory=list)
    hyperparameter_search: list[dict[str, Any]] = field(default_factory=list)
    svm_results: list[dict[str, Any]] = field(default_factory=list)
    selection_cv: list[dict[str, Any]] = field(default_factory=list)
    selected_configurations: list[dict[str, Any]] = field(default_factory=list)


def task_context(
    seed: int,
    feature_set: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "split_id": f"seed_{seed}",
        "feature_set": feature_set,
        "train_index_hash": hash_indices(train_indices),
        "test_index_hash": hash_indices(test_indices),
    }


def add_prediction_rows(
    artifacts: TaskArtifacts,
    context: dict[str, Any],
    model_family: str,
    model: str,
    cutoff_rule: str,
    strategy: str,
    row_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict[str, float],
    true_class: np.ndarray | None = None,
    routed_class: np.ndarray | None = None,
) -> None:
    for position, row_index in enumerate(row_indices):
        artifacts.predictions.append(
            {
                **context,
                "model_family": model_family,
                "model": model,
                "cutoff_rule": cutoff_rule,
                "transfer_strategy": strategy,
                "row_index": int(row_index),
                "y_true": float(y_true[position]),
                "y_pred": float(y_pred[position]),
                "residual": float(y_pred[position] - y_true[position]),
                "true_cost_class": ""
                if true_class is None
                else int(true_class[position]),
                "routed_cost_class": ""
                if routed_class is None
                else int(routed_class[position]),
                "at_or_above_train_q3": bool(y_true[position] >= thresholds["Q3"]),
                "at_or_above_train_q3_iqr2": bool(
                    y_true[position] >= thresholds["Q3+IQR/2"]
                ),
            }
        )


def add_metric_row(
    artifacts: TaskArtifacts,
    context: dict[str, Any],
    model_family: str,
    model: str,
    cutoff_rule: str,
    strategy: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict[str, float],
    selected_hyperparameters: dict[str, Any],
    elapsed_seconds: float,
    run_status: str = "complete",
    classifier: dict[str, float] | None = None,
    counts: dict[str, int] | None = None,
    frozen_layers: list[str] | None = None,
    learning_rate: float | None = None,
    training_epochs: dict[str, int] | None = None,
) -> dict[str, Any]:
    regression = evaluate_regression(
        y_true, y_pred, thresholds["Q3"], thresholds["Q3+IQR/2"]
    )
    row = {
        **context,
        "model_family": model_family,
        "model": model,
        "cutoff_rule": cutoff_rule,
        "training_derived_cutoff_value": thresholds.get(cutoff_rule, np.nan),
        "transfer_strategy": strategy,
        "selected_hyperparameters": json_dumps(selected_hyperparameters),
        "frozen_layers": json_dumps(frozen_layers or []),
        "learning_rate": learning_rate,
        "training_epochs": json_dumps(training_epochs or {}),
        "run_status": run_status,
        "elapsed_seconds": float(elapsed_seconds),
        **regression,
    }
    if classifier:
        row.update(classifier)
    else:
        row.update(
            {
                "classifier_precision": np.nan,
                "classifier_recall": np.nan,
                "classifier_f1": np.nan,
            }
        )
    row.update(counts or {})
    artifacts.metrics.append(row)
    return row


def select_transfer_configuration(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    seed: int,
    svm_params: dict[str, dict[str, Any]],
    cv_folds: int,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed + 7000)
    strategies = list(TRANSFER_STRATEGIES)
    if smoke:
        strategies = ["Conservative", "Balanced", "Aggressive"]

    for fold, (development, validation) in enumerate(splitter.split(X_train)):
        X_development_raw = X_train.iloc[development]
        X_validation_raw = X_train.iloc[validation]
        y_development = y_train[development]
        y_validation = y_train[validation]
        fold_thresholds = calculate_thresholds(y_development)

        preprocessor = make_preprocessor(features)
        X_development = np.asarray(
            preprocessor.fit_transform(X_development_raw), dtype=np.float32
        )
        X_validation = np.asarray(
            preprocessor.transform(X_validation_raw), dtype=np.float32
        )
        target_scaler = StandardScaler()
        y_development_scaled = target_scaler.fit_transform(
            y_development.reshape(-1, 1)
        ).reshape(-1)

        for cutoff_index, cutoff_rule in enumerate(CUTOFF_RULES):
            cutoff = fold_thresholds[cutoff_rule]
            development_class = (y_development >= cutoff).astype(int)
            classifier = SVC(**svm_params[cutoff_rule])
            classifier.fit(X_development, development_class)
            validation_route = classifier.predict(X_validation)

            low_mask = development_class == 0
            high_mask = development_class == 1
            base_model, base_epochs = fit_ann_final(
                X_development[low_mask],
                y_development_scaled[low_mask].astype(np.float32),
                TL_BASE_CONFIG,
                seed + fold * 1000 + cutoff_index * 100,
                max_epochs=max_epochs,
                patience=patience,
            )

            for strategy_index, strategy_name in enumerate(strategies):
                transferred_model, high_epochs, frozen_names = fit_transfer_final(
                    base_model,
                    X_development[high_mask],
                    y_development_scaled[high_mask].astype(np.float32),
                    strategy_name,
                    int(TL_BASE_CONFIG["batch_size"]),
                    seed
                    + fold * 1000
                    + cutoff_index * 100
                    + strategy_index
                    + 1,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                prediction_scaled = np.empty(len(validation_route), dtype=float)
                low_validation = validation_route == 0
                high_validation = validation_route == 1
                if low_validation.any():
                    prediction_scaled[low_validation] = base_model.predict(
                        X_validation[low_validation], verbose=0
                    ).reshape(-1)
                if high_validation.any():
                    prediction_scaled[high_validation] = transferred_model.predict(
                        X_validation[high_validation], verbose=0
                    ).reshape(-1)
                prediction = inverse_target(target_scaler, prediction_scaled)
                metrics = evaluate_regression(
                    y_validation,
                    prediction,
                    fold_thresholds["Q3"],
                    fold_thresholds["Q3+IQR/2"],
                )
                rows.append(
                    {
                        "record_type": "fold",
                        "fold": fold,
                        "cutoff_rule": cutoff_rule,
                        "transfer_strategy": strategy_name,
                        "fold_training_cutoff_value": cutoff,
                        "base_epochs": base_epochs,
                        "high_epochs": high_epochs,
                        "frozen_layers": json_dumps(frozen_names),
                        **metrics,
                    }
                )
                del transferred_model
                gc.collect()
            del base_model
            tf.keras.backend.clear_session()
            gc.collect()

    fold_frame = pd.DataFrame(rows)
    metric_columns = ["rmse_q3", "rmse", "mape_pct", "r2", "rmse_q3_iqr2"]
    aggregate = (
        fold_frame.groupby(["cutoff_rule", "transfer_strategy"], as_index=False)[
            metric_columns
        ]
        .mean()
        .sort_values(
            [
                "rmse_q3",
                "rmse",
                "mape_pct",
                "cutoff_rule",
                "transfer_strategy",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    for rank, (_, record) in enumerate(aggregate.iterrows(), start=1):
        rows.append(
            {
                "record_type": "aggregate",
                "fold": "",
                "cutoff_rule": record["cutoff_rule"],
                "transfer_strategy": record["transfer_strategy"],
                "selection_rank": rank,
                **{column: float(record[column]) for column in metric_columns},
            }
        )
    selected = {
        "cutoff_rule": str(aggregate.iloc[0]["cutoff_rule"]),
        "transfer_strategy": str(aggregate.iloc[0]["transfer_strategy"]),
    }
    return selected, rows


def run_task(
    data_path: Path,
    output_dir: Path,
    seed: int,
    feature_set: str,
    smoke: bool,
    force: bool,
    grid_jobs: int,
    model_threads: int,
) -> None:
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set}")
    task_name = f"{feature_set}_seed{seed}"
    task_dir = output_dir / ("smoke_tasks" if smoke else "tasks") / task_name
    complete_path = task_dir / "complete.json"
    if complete_path.exists() and not force:
        log(f"SKIP {task_name}: completion marker already exists")
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    task_started = time.perf_counter()
    artifacts = TaskArtifacts()
    features = FEATURE_SETS[feature_set]

    try:
        frame = load_dataset(data_path)
        row_indices = np.arange(len(frame))
        train_indices, test_indices = train_test_split(
            row_indices, test_size=0.30, random_state=seed, shuffle=True, stratify=None
        )
        train_indices = np.asarray(train_indices)
        test_indices = np.asarray(test_indices)
        context = task_context(seed, feature_set, train_indices, test_indices)
        X = frame[features]
        y = frame[TARGET].to_numpy(dtype=float)
        X_train = X.iloc[train_indices].copy()
        X_test = X.iloc[test_indices].copy()
        y_train = y[train_indices]
        y_test = y[test_indices]
        thresholds = calculate_thresholds(y_train)

        for row_index in train_indices:
            artifacts.split_indices.append(
                {**context, "row_index": int(row_index), "partition": "train"}
            )
        for row_index in test_indices:
            artifacts.split_indices.append(
                {**context, "row_index": int(row_index), "partition": "test"}
            )

        for cutoff_rule in CUTOFF_RULES:
            cutoff = thresholds[cutoff_rule]
            train_class = (y_train >= cutoff).astype(int)
            test_class = (y_test >= cutoff).astype(int)
            artifacts.thresholds.append(
                {
                    **context,
                    "cutoff_rule": cutoff_rule,
                    "training_derived_cutoff_value": cutoff,
                    "train_low_count": int((train_class == 0).sum()),
                    "train_high_count": int((train_class == 1).sum()),
                    "test_true_low_count": int((test_class == 0).sum()),
                    "test_true_high_count": int((test_class == 1).sum()),
                    "test_n_at_or_above_train_q3": int(
                        (y_test >= thresholds["Q3"]).sum()
                    ),
                    "test_n_at_or_above_train_q3_iqr2": int(
                        (y_test >= thresholds["Q3+IQR/2"]).sum()
                    ),
                }
            )

        cv_folds = 2 if smoke else 3
        max_epochs = SMOKE_MAX_EPOCHS if smoke else FULL_MAX_EPOCHS
        patience = (
            SMOKE_EARLY_STOPPING_PATIENCE
            if smoke
            else FULL_EARLY_STOPPING_PATIENCE
        )
        selection_folds = 2 if smoke else 3
        log(
            f"START {task_name}: train={len(train_indices)}, test={len(test_indices)}, "
            f"features={len(features)}, smoke={smoke}"
        )

        # Single-stage tree models: preprocessing is inside every CV fold.
        for model_index, model_name in enumerate(TREE_GRIDS):
            log(f"{task_name} | tuning {model_name}")
            estimator, best_params, search_rows, search_elapsed = tune_tree_model(
                model_name,
                X_train,
                y_train,
                features,
                seed + model_index,
                cv_folds,
                grid_jobs,
                model_threads,
                smoke,
            )
            for row in search_rows:
                artifacts.hyperparameter_search.append(
                    {**context, "model": model_name, **row}
                )
            prediction_started = time.perf_counter()
            prediction = estimator.predict(X_test)
            elapsed = search_elapsed + (time.perf_counter() - prediction_started)
            add_metric_row(
                artifacts,
                context,
                "single",
                model_name,
                "",
                "",
                y_test,
                prediction,
                thresholds,
                best_params,
                elapsed,
            )
            add_prediction_rows(
                artifacts,
                context,
                "single",
                model_name,
                "",
                "",
                test_indices,
                y_test,
                prediction,
                thresholds,
            )
            del estimator
            gc.collect()

        # Single-stage ANN tuning. The pipeline keeps both input preprocessing and
        # target standardization inside each training/validation fold.
        log(f"{task_name} | tuning single ANN")
        (
            ann_estimator,
            ann_best,
            ann_search_rows,
            ann_elapsed,
            ann_iterations,
        ) = tune_single_mlp(
            X_train,
            y_train,
            features,
            seed,
            cv_folds,
            grid_jobs,
            smoke,
        )
        for row in ann_search_rows:
            artifacts.hyperparameter_search.append(
                {**context, "model": "ANN", **row}
            )
        ann_prediction = ann_estimator.predict(X_test)
        add_metric_row(
            artifacts,
            context,
            "single",
            "ANN",
            "",
            "",
            y_test,
            ann_prediction,
            thresholds,
            ann_best,
            ann_elapsed,
            learning_rate=float(ann_best["learning_rate_init"]),
            training_epochs={"final_iterations": ann_iterations},
        )
        add_prediction_rows(
            artifacts,
            context,
            "single",
            "ANN",
            "",
            "",
            test_indices,
            y_test,
            ann_prediction,
            thresholds,
        )
        del ann_estimator
        gc.collect()

        # The two-stage Keras models share a training-only input transform and
        # target scaler across cost segments so transferred output weights have
        # the same target scale.
        final_preprocessor = make_preprocessor(features)
        X_train_processed = np.asarray(
            final_preprocessor.fit_transform(X_train), dtype=np.float32
        )
        X_test_processed = np.asarray(
            final_preprocessor.transform(X_test), dtype=np.float32
        )
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # Tune one SVM per training-derived cutoff using training-only CV.
        svm_estimators: dict[str, Pipeline] = {}
        svm_params: dict[str, dict[str, Any]] = {}
        for cutoff_index, cutoff_rule in enumerate(CUTOFF_RULES):
            log(f"{task_name} | tuning SVM ({cutoff_rule})")
            cutoff = thresholds[cutoff_rule]
            train_class = (y_train >= cutoff).astype(int)
            estimator, best_params, search_rows, best_score, elapsed = tune_svm(
                X_train,
                train_class,
                features,
                seed + cutoff_index,
                cv_folds,
                grid_jobs,
                smoke,
            )
            svm_estimators[cutoff_rule] = estimator
            svm_params[cutoff_rule] = best_params
            for row in search_rows:
                artifacts.hyperparameter_search.append(
                    {
                        **context,
                        "model": "SVM",
                        "cutoff_rule": cutoff_rule,
                        **row,
                    }
                )
            test_route = estimator.predict(X_test)
            test_class = (y_test >= cutoff).astype(int)
            artifacts.svm_results.append(
                {
                    **context,
                    "cutoff_rule": cutoff_rule,
                    "training_derived_cutoff_value": cutoff,
                    "selected_hyperparameters": json_dumps(best_params),
                    "cv_f1": best_score,
                    **classifier_scores(test_class, test_route),
                    "train_low_count": int((train_class == 0).sum()),
                    "train_high_count": int((train_class == 1).sum()),
                    "test_true_low_count": int((test_class == 0).sum()),
                    "test_true_high_count": int((test_class == 1).sum()),
                    "test_predicted_low_count": int((test_route == 0).sum()),
                    "test_predicted_high_count": int((test_route == 1).sum()),
                    "elapsed_seconds": elapsed,
                }
            )

        # Freeze the training-only selection rule before outer-test model evaluation.
        log(f"{task_name} | selecting cutoff/strategy inside outer training only")
        selected, selection_rows = select_transfer_configuration(
            X_train,
            y_train,
            features,
            seed,
            svm_params,
            selection_folds,
            max_epochs,
            patience,
            smoke,
        )
        for row in selection_rows:
            artifacts.selection_cv.append({**context, **row})
        artifacts.selected_configurations.append(
            {
                **context,
                **selected,
                "primary_selection_endpoint": "rmse_q3",
                "tie_breakers": "rmse,mape_pct,lexical",
                "selection_scope": "outer training only",
            }
        )
        log(
            f"{task_name} | selected {selected['cutoff_rule']} / "
            f"{selected['transfer_strategy']}"
        )

        # Fit and evaluate the no-transfer and all exploratory transfer configurations.
        selected_payload: dict[str, Any] | None = None
        for cutoff_index, cutoff_rule in enumerate(CUTOFF_RULES):
            cutoff = thresholds[cutoff_rule]
            train_class = (y_train >= cutoff).astype(int)
            test_true_class = (y_test >= cutoff).astype(int)
            test_route = svm_estimators[cutoff_rule].predict(X_test)
            class_metrics = classifier_scores(test_true_class, test_route)
            counts = {
                "train_low_count": int((train_class == 0).sum()),
                "train_high_count": int((train_class == 1).sum()),
                "test_true_low_count": int((test_true_class == 0).sum()),
                "test_true_high_count": int((test_true_class == 1).sum()),
                "test_predicted_low_count": int((test_route == 0).sum()),
                "test_predicted_high_count": int((test_route == 1).sum()),
            }
            low_mask = train_class == 0
            high_mask = train_class == 1
            cutoff_started = time.perf_counter()
            base_model, base_epochs = fit_ann_final(
                X_train_processed[low_mask],
                y_train_scaled[low_mask].astype(np.float32),
                TL_BASE_CONFIG,
                seed + 10000 + cutoff_index * 100,
                max_epochs=max_epochs,
                patience=patience,
            )
            high_scratch_model, high_scratch_epochs = fit_ann_final(
                X_train_processed[high_mask],
                y_train_scaled[high_mask].astype(np.float32),
                TL_BASE_CONFIG,
                seed + 10100 + cutoff_index * 100,
                max_epochs=max_epochs,
                patience=patience,
            )
            no_transfer_scaled = np.empty(len(test_route), dtype=float)
            low_test = test_route == 0
            high_test = test_route == 1
            if low_test.any():
                no_transfer_scaled[low_test] = base_model.predict(
                    X_test_processed[low_test], verbose=0
                ).reshape(-1)
            if high_test.any():
                no_transfer_scaled[high_test] = high_scratch_model.predict(
                    X_test_processed[high_test], verbose=0
                ).reshape(-1)
            no_transfer_prediction = inverse_target(target_scaler, no_transfer_scaled)
            no_transfer_elapsed = time.perf_counter() - cutoff_started
            add_metric_row(
                artifacts,
                context,
                "two_stage_no_tl",
                "Two-stage, no transfer",
                cutoff_rule,
                "",
                y_test,
                no_transfer_prediction,
                thresholds,
                {
                    "svm": svm_params[cutoff_rule],
                    "ann": TL_BASE_CONFIG,
                },
                no_transfer_elapsed,
                classifier=class_metrics,
                counts=counts,
                learning_rate=float(TL_BASE_CONFIG["lr"]),
                training_epochs={
                    "low": base_epochs,
                    "high": high_scratch_epochs,
                },
            )
            add_prediction_rows(
                artifacts,
                context,
                "two_stage_no_tl",
                "Two-stage, no transfer",
                cutoff_rule,
                "",
                test_indices,
                y_test,
                no_transfer_prediction,
                thresholds,
                test_true_class,
                test_route,
            )
            del high_scratch_model
            gc.collect()

            for strategy_index, strategy_name in enumerate(TRANSFER_STRATEGIES):
                strategy_started = time.perf_counter()
                transferred_model, high_epochs, frozen_names = fit_transfer_final(
                    base_model,
                    X_train_processed[high_mask],
                    y_train_scaled[high_mask].astype(np.float32),
                    strategy_name,
                    int(TL_BASE_CONFIG["batch_size"]),
                    seed + 11000 + cutoff_index * 100 + strategy_index,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                transfer_scaled = np.empty(len(test_route), dtype=float)
                if low_test.any():
                    transfer_scaled[low_test] = base_model.predict(
                        X_test_processed[low_test], verbose=0
                    ).reshape(-1)
                if high_test.any():
                    transfer_scaled[high_test] = transferred_model.predict(
                        X_test_processed[high_test], verbose=0
                    ).reshape(-1)
                transfer_prediction = inverse_target(target_scaler, transfer_scaled)
                strategy_elapsed = time.perf_counter() - strategy_started
                metric_row = add_metric_row(
                    artifacts,
                    context,
                    "two_stage_tl_exploratory",
                    "Two-stage, transfer",
                    cutoff_rule,
                    strategy_name,
                    y_test,
                    transfer_prediction,
                    thresholds,
                    {
                        "svm": svm_params[cutoff_rule],
                        "ann": TL_BASE_CONFIG,
                        "strategy": TRANSFER_STRATEGIES[strategy_name],
                    },
                    strategy_elapsed,
                    classifier=class_metrics,
                    counts=counts,
                    frozen_layers=frozen_names,
                    learning_rate=float(TRANSFER_STRATEGIES[strategy_name]["lr"]),
                    training_epochs={"low": base_epochs, "high": high_epochs},
                )
                add_prediction_rows(
                    artifacts,
                    context,
                    "two_stage_tl_exploratory",
                    "Two-stage, transfer",
                    cutoff_rule,
                    strategy_name,
                    test_indices,
                    y_test,
                    transfer_prediction,
                    thresholds,
                    test_true_class,
                    test_route,
                )
                if (
                    cutoff_rule == selected["cutoff_rule"]
                    and strategy_name == selected["transfer_strategy"]
                ):
                    selected_payload = {
                        "metric_row": metric_row,
                        "prediction": transfer_prediction.copy(),
                        "true_class": test_true_class.copy(),
                        "route": test_route.copy(),
                        "frozen_names": list(frozen_names),
                        "epochs": {"low": base_epochs, "high": high_epochs},
                        "counts": counts,
                        "classifier": class_metrics,
                    }
                del transferred_model
                gc.collect()
            del base_model
            tf.keras.backend.clear_session()
            gc.collect()

        if selected_payload is None:
            raise RuntimeError("Selected transfer configuration was not evaluated")
        add_metric_row(
            artifacts,
            context,
            "two_stage_tl_nested_selected",
            "Two-stage TL, training-selected",
            selected["cutoff_rule"],
            selected["transfer_strategy"],
            y_test,
            selected_payload["prediction"],
            thresholds,
            {
                "svm": svm_params[selected["cutoff_rule"]],
                "ann": TL_BASE_CONFIG,
                "strategy": TRANSFER_STRATEGIES[selected["transfer_strategy"]],
                "selection_endpoint": "rmse_q3",
            },
            float(selected_payload["metric_row"]["elapsed_seconds"]),
            classifier=selected_payload["classifier"],
            counts=selected_payload["counts"],
            frozen_layers=selected_payload["frozen_names"],
            learning_rate=float(
                TRANSFER_STRATEGIES[selected["transfer_strategy"]]["lr"]
            ),
            training_epochs=selected_payload["epochs"],
        )
        add_prediction_rows(
            artifacts,
            context,
            "two_stage_tl_nested_selected",
            "Two-stage TL, training-selected",
            selected["cutoff_rule"],
            selected["transfer_strategy"],
            test_indices,
            y_test,
            selected_payload["prediction"],
            thresholds,
            selected_payload["true_class"],
            selected_payload["route"],
        )

        tables = {
            "model_metrics.csv": artifacts.metrics,
            "test_predictions.csv": artifacts.predictions,
            "training_thresholds.csv": artifacts.thresholds,
            "split_indices.csv": artifacts.split_indices,
            "hyperparameter_search.csv": artifacts.hyperparameter_search,
            "svm_results.csv": artifacts.svm_results,
            "configuration_selection_cv.csv": artifacts.selection_cv,
            "selected_configuration.csv": artifacts.selected_configurations,
        }
        for filename, records in tables.items():
            atomic_write_dataframe(pd.DataFrame(records), task_dir / filename)

        completion = {
            "status": "complete",
            "task": task_name,
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - task_started,
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "row_count": len(frame),
            "feature_count": len(features),
            "features": features,
            "seed": seed,
            "smoke": smoke,
            "package_versions": package_versions(),
        }
        atomic_write_json(completion, complete_path)
        log(
            f"COMPLETE {task_name} in {completion['elapsed_seconds'] / 60.0:.1f} min"
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "task": task_name,
            "started_at": started_at,
            "failed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - task_started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(failure, task_dir / "failure.json")
        log(f"FAILED {task_name}: {type(error).__name__}: {error}")
        raise


def parse_csv_values(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled repeated-holdout construction-cost analysis"
    )
    parser.add_argument("--data", type=Path, default=Path("Data_ML2.xlsx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corrected_analysis_output"),
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--grid-jobs", type=int, default=1)
    parser.add_argument("--model-threads", type=int, default=2)
    args = parser.parse_args()

    configure_tensorflow(args.model_threads)
    seeds = parse_csv_values(args.seeds, int)
    feature_sets = parse_csv_values(args.feature_sets, str)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "analysis": "corrected construction-cost repeated-holdout evaluation",
        "protocol": "embedded leakage-controlled corrective protocol; see module docstring",
        "created_at": utc_now(),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "seeds": seeds,
        "feature_sets": {name: FEATURE_SETS[name] for name in feature_sets},
        "outer_test_size": 0.30,
        "outer_stratification": "none",
        "regression_cv_folds": 2 if args.smoke else 3,
        "classification_cv_folds": 2 if args.smoke else 3,
        "configuration_selection_folds": 2 if args.smoke else 3,
        "primary_selection_endpoint": "rmse_q3",
        "selection_tie_breakers": ["rmse", "mape_pct", "lexical"],
        "tree_grids": TREE_GRIDS,
        "svm_grid": SVM_GRID,
        "ann_grid": ANN_GRID,
        "tl_base_config": TL_BASE_CONFIG,
        "transfer_strategies": TRANSFER_STRATEGIES,
        "maximum_epochs": SMOKE_MAX_EPOCHS if args.smoke else FULL_MAX_EPOCHS,
        "early_stopping_patience": (
            SMOKE_EARLY_STOPPING_PATIENCE
            if args.smoke
            else FULL_EARLY_STOPPING_PATIENCE
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "smoke": args.smoke,
    }
    metadata_name = "smoke_run_metadata.json" if args.smoke else "run_metadata.json"
    atomic_write_json(run_metadata, args.output_dir / metadata_name)

    for feature_set in feature_sets:
        for seed in seeds:
            run_task(
                args.data,
                args.output_dir,
                seed,
                feature_set,
                args.smoke,
                args.force,
                args.grid_jobs,
                args.model_threads,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
