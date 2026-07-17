from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.corrected_cost_pipeline import (
    FEATURE_SETS,
    calculate_thresholds,
    evaluate_regression,
    hash_indices,
    load_dataset,
    make_preprocessor,
    monotone_vector,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast deterministic pipeline checks")
    parser.add_argument("--data", type=Path, default=Path("Data_ML2.xlsx"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = load_dataset(args.data)
    assert frame.shape == (419, 22)
    assert not frame.isna().any().any()
    assert not frame.duplicated().any()
    assert {name: len(features) for name, features in FEATURE_SETS.items()} == {
        "A_all_21": 21,
        "B_no_component_costs": 17,
        "C_physical_quantity": 16,
    }

    full_thresholds = calculate_thresholds(frame["O"].to_numpy())
    expected_thresholds = {
        "Mean": 148271.18986549918,
        "Q3": 180676.1142,
        "Q3+IQR/2": 228626.21862,
    }
    for name, expected in expected_thresholds.items():
        assert np.isclose(full_thresholds[name], expected, rtol=0, atol=1e-8)

    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.30, random_state=42, shuffle=True, stratify=None
    )
    assert len(train_indices) == 293
    assert len(test_indices) == 126
    assert set(train_indices).isdisjoint(set(test_indices))
    assert len(set(train_indices) | set(test_indices)) == 419

    transformed_shapes: dict[str, list[int]] = {}
    monotone_sums: dict[str, int] = {}
    for name, features in FEATURE_SETS.items():
        preprocessor = make_preprocessor(features)
        transformed = preprocessor.fit_transform(frame.iloc[train_indices][features])
        transformed_shapes[name] = list(transformed.shape)
        assert transformed.shape == (293, len(features))
        vector = monotone_vector(features)
        assert len(vector) == transformed.shape[1]
        monotone_sums[name] = int(sum(vector))
    assert monotone_sums == {
        "A_all_21": 6,
        "B_no_component_costs": 5,
        "C_physical_quantity": 5,
    }

    y_test = frame.iloc[test_indices]["O"].to_numpy()
    training_thresholds = calculate_thresholds(frame.iloc[train_indices]["O"].to_numpy())
    perfect = evaluate_regression(
        y_test,
        y_test,
        training_thresholds["Q3"],
        training_thresholds["Q3+IQR/2"],
    )
    assert np.isclose(perfect["r2"], 1.0)
    assert np.isclose(perfect["rmse"], 0.0)
    assert np.isclose(perfect["rmse_q3"], 0.0)
    assert np.isclose(perfect["rmse_q3_iqr2"], 0.0)
    assert perfect["n_rmse_q3"] > 0
    assert perfect["n_rmse_q3_iqr2"] > 0

    verification = {
        "status": "passed",
        "data_shape": list(frame.shape),
        "full_dataset_descriptive_thresholds": full_thresholds,
        "seed_42_train_hash": hash_indices(train_indices),
        "seed_42_test_hash": hash_indices(test_indices),
        "transformed_shapes": transformed_shapes,
        "monotone_constraint_counts": monotone_sums,
        "perfect_prediction_segment_counts": {
            "q3": perfect["n_rmse_q3"],
            "q3_iqr2": perfect["n_rmse_q3_iqr2"],
        },
    }
    text = json.dumps(verification, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
