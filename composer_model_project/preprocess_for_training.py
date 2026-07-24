"""Prepare leakage-safe train/validation/test arrays for LSTM and CNN training."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR.parent / "upload" / "train_features.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts"


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Unsupported value: {type(value)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # filename and relative_path are identifiers and can expose the target label.
    target = "composer"
    identifier_columns = ["filename", "relative_path"]
    X = df.drop(columns=[target, *identifier_columns]).select_dtypes(include="number")
    y_text = df[target].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    row_indices = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        row_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=SEED,
        stratify=y[train_val_idx],
    )

    # Learn every preprocessing decision from the training split only.
    train_variances = X.iloc[train_idx].var(axis=0)
    feature_columns = train_variances[train_variances > 0].index.tolist()
    removed_constant_columns = train_variances[train_variances == 0].index.tolist()
    X = X[feature_columns].astype("float32")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx]).astype("float32")
    X_val = scaler.transform(X.iloc[val_idx]).astype("float32")
    X_test = scaler.transform(X.iloc[test_idx]).astype("float32")

    y_train = y[train_idx].astype("int32")
    y_val = y[val_idx].astype("int32")
    y_test = y[test_idx].astype("int32")

    classes = np.arange(len(label_encoder.classes_))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {str(int(k)): float(v) for k, v in zip(classes, weights)}

    payload = {
        "seed": SEED,
        "feature_columns": feature_columns,
        "class_names": label_encoder.classes_.tolist(),
        "class_weights": class_weights,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    (OUTPUT_DIR / "preprocessed_data.json").write_text(
        json.dumps(payload, default=to_serializable), encoding="utf-8"
    )

    metadata = {
        "source_file": CSV_PATH.name,
        "rows": int(len(df)),
        "original_numeric_features": int(df.select_dtypes(include="number").shape[1]),
        "retained_features": int(len(feature_columns)),
        "removed_constant_columns": removed_constant_columns,
        "excluded_identifier_columns": identifier_columns,
        "class_names": label_encoder.classes_.tolist(),
        "class_counts": y_text.value_counts().sort_index().to_dict(),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_weights": class_weights,
        "feature_columns": feature_columns,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "seed": SEED,
    }
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
        },
        OUTPUT_DIR / "preprocessor.joblib",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
