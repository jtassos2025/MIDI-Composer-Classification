"""Create plots, a metrics table, README, and a reproducible Keras notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/composer_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CLASS_NAMES = ["Bach", "Beethoven", "Chopin", "Mozart"]


class _NotebookV4:
    @staticmethod
    def new_notebook() -> dict:
        return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}

    @staticmethod
    def new_markdown_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    @staticmethod
    def new_code_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }


class _NotebookWriter:
    v4 = _NotebookV4()

    @staticmethod
    def write(notebook: dict, path: Path) -> None:
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


nbf = _NotebookWriter()


def load_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def create_plots(cnn: dict, lstm: dict) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, (name, result, color) in enumerate(
        [("CNN", cnn, "#2563eb"), ("LSTM", lstm, "#7c3aed")]
    ):
        history = result["history"]
        epochs = np.arange(1, len(history["loss"]) + 1)
        axes[0, col].plot(epochs, history["loss"], label="Train", color=color)
        axes[0, col].plot(epochs, history["val_loss"], label="Validation", color="#f97316")
        axes[0, col].axvline(result["best_epoch"], linestyle="--", color="#64748b", alpha=0.8)
        axes[0, col].set(title=f"{name} Loss", xlabel="Epoch", ylabel="Loss")
        axes[0, col].legend()
        axes[1, col].plot(epochs, history["acc"], label="Train", color=color)
        axes[1, col].plot(epochs, history["val_acc"], label="Validation", color="#f97316")
        axes[1, col].axvline(result["best_epoch"], linestyle="--", color="#64748b", alpha=0.8)
        axes[1, col].set(title=f"{name} Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
        axes[1, col].legend()
    fig.suptitle("Composer Classification Training History", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, result) in zip(axes, [("CNN", cnn), ("LSTM", lstm)]):
        sns.heatmap(
            np.asarray(result["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set(title=f"{name} Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Model": ["CNN", "LSTM"],
            "Accuracy": [cnn["accuracy"], lstm["accuracy"]],
            "Macro F1": [cnn["macro_f1"], lstm["macro_f1"]],
        }
    ).melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=comparison, x="Model", y="Score", hue="Metric", palette=["#2563eb", "#f97316"], ax=ax)
    ax.set(title="Held-Out Test Performance", ylabel="Score", ylim=(0, 1.0))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_metrics_csv(cnn: dict, lstm: dict) -> None:
    rows = []
    for model, result in [("CNN", cnn), ("LSTM", lstm)]:
        rows.append(
            {
                "model": model,
                "test_accuracy": result["accuracy"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "macro_f1": result["macro_f1"],
                "epochs_trained": result["epochs_trained"],
                "best_epoch": result["best_epoch"],
                "parameters": result["parameter_count"],
            }
        )
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "model_results.csv", index=False)


def create_readme(cnn: dict, lstm: dict) -> None:
    text = f"""# LSTM and CNN Composer Classification

This project classifies 1,138 MIDI-derived musical scores as Bach, Beethoven, Chopin, or Mozart using the supplied `train_features.csv` data.

## Data and preprocessing

- Excluded `filename` and `relative_path` to prevent composer-name leakage.
- Removed three constant numeric columns, leaving 57 features.
- Used a stratified 64%/16%/20% train/validation/test split: 728/182/228 rows.
- Fit `StandardScaler` on training data only.
- Used balanced class weighting (moderated for CNN) because Bach accounts for 716 of 1,138 rows.
- Reshaped CNN input to 57 time steps × 1 channel.
- Grouped the LSTM input as 19 steps × 3 adjacent feature channels.

## Held-out test results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| CNN | {cnn['accuracy']:.3f} | {cnn['macro_precision']:.3f} | {cnn['macro_recall']:.3f} | {cnn['macro_f1']:.3f} |
| LSTM | {lstm['accuracy']:.3f} | {lstm['macro_precision']:.3f} | {lstm['macro_recall']:.3f} | {lstm['macro_f1']:.3f} |

The CNN is the recommended model for this feature table. The LSTM is weaker because these rows contain aggregate score-level features, not true chronological note sequences. An LSTM would be more naturally suited to raw note/event sequences extracted from MIDI files.

## Contents

- `composer_classification_lstm_cnn.ipynb`: reproducible Python/TensorFlow notebook.
- `artifacts/cnn_tfjs_model/` and `artifacts/lstm_tfjs_model/`: trained TensorFlow.js models.
- `artifacts/preprocessor.joblib`: fitted scaler, label encoder, and selected feature list.
- `artifacts/model_results.csv`: summary metrics.
- `artifacts/training_curves.png`, `confusion_matrices.png`, and `model_comparison.png`: evaluation plots.
- `artifacts/*_metrics.json`: detailed histories and per-class results.
- `preprocess_for_training.py` and `train_tfjs.cjs`: scripts used for the validation run.

## Run the Keras notebook

Place `train_features.csv` beside the notebook (or keep it under `upload/`) and run all cells. Install TensorFlow in that notebook environment if needed:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn joblib
```

The recorded metrics were produced by an equivalent TensorFlow.js CPU run. A fresh Keras run can vary slightly because of framework kernels and random initialization.
"""
    (PROJECT_DIR / "README.md").write_text(text, encoding="utf-8")


def create_notebook(cnn: dict, lstm: dict) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# Composer Classification with CNN and LSTM

This notebook builds and trains two deep-learning classifiers on `train_features.csv` to predict **Bach, Beethoven, Chopin, or Mozart**.

Validated held-out results from the supplied file:

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| 1D CNN | {cnn['accuracy']:.3f} | {cnn['macro_f1']:.3f} |
| LSTM | {lstm['accuracy']:.3f} | {lstm['macro_f1']:.3f} |

The identifier columns are deliberately excluded because their text contains composer information. The models use only numeric musical features."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """# If TensorFlow is missing, run this once, then restart the kernel:\n# %pip install tensorflow pandas scikit-learn matplotlib seaborn joblib\n\nimport os\nimport random\nfrom pathlib import Path\n\nimport joblib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nimport tensorflow as tf\nfrom sklearn.metrics import (\n    accuracy_score, classification_report, confusion_matrix,\n    precision_recall_fscore_support\n)\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder, StandardScaler\nfrom sklearn.utils.class_weight import compute_class_weight\nfrom tensorflow.keras import Sequential\nfrom tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom tensorflow.keras.layers import (\n    Conv1D, Dense, Dropout, Flatten, Input, LSTM, MaxPooling1D\n)\n\nSEED = 42\nos.environ['PYTHONHASHSEED'] = str(SEED)\nrandom.seed(SEED)\nnp.random.seed(SEED)\ntf.random.set_seed(SEED)\nprint('TensorFlow:', tf.__version__)"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load and inspect the data

The search below supports either placing the CSV beside the notebook or keeping it in an `upload` folder."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """candidates = [\n    Path('train_features.csv'),\n    Path('upload/train_features.csv'),\n    Path('../upload/train_features.csv'),\n]\nCSV_PATH = next((path for path in candidates if path.exists()), None)\nif CSV_PATH is None:\n    raise FileNotFoundError('Place train_features.csv beside this notebook or in an upload folder.')\n\ndf = pd.read_csv(CSV_PATH)\nprint('Shape:', df.shape)\nprint('Missing values:', int(df.isna().sum().sum()))\nprint('Duplicate rows:', int(df.duplicated().sum()))\ndisplay(df.head())\ndisplay(df['composer'].value_counts().rename('count').to_frame())"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Preprocess and split

- `composer` is the target.
- `filename` and `relative_path` are excluded to prevent label leakage.
- The split is stratified so each composer is represented proportionally.
- Constant-column removal and standardization are learned from training data only."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """TARGET = 'composer'\nIDENTIFIERS = ['filename', 'relative_path']\n\nX_df = df.drop(columns=[TARGET, *IDENTIFIERS]).select_dtypes(include='number')\ny_text = df[TARGET].astype(str)\n\nlabel_encoder = LabelEncoder()\ny = label_encoder.fit_transform(y_text)\nclass_names = label_encoder.classes_\nrow_indices = np.arange(len(df))\n\ntrain_val_idx, test_idx = train_test_split(\n    row_indices, test_size=0.20, random_state=SEED, stratify=y\n)\ntrain_idx, val_idx = train_test_split(\n    train_val_idx, test_size=0.20, random_state=SEED, stratify=y[train_val_idx]\n)\n\ntrain_variances = X_df.iloc[train_idx].var(axis=0)\nfeature_columns = train_variances[train_variances > 0].index.tolist()\nconstant_columns = train_variances[train_variances == 0].index.tolist()\nX_df = X_df[feature_columns].astype('float32')\n\nscaler = StandardScaler()\nX_train_2d = scaler.fit_transform(X_df.iloc[train_idx]).astype('float32')\nX_val_2d = scaler.transform(X_df.iloc[val_idx]).astype('float32')\nX_test_2d = scaler.transform(X_df.iloc[test_idx]).astype('float32')\ny_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]\n\nprint('Classes:', class_names.tolist())\nprint('Split sizes:', len(train_idx), len(val_idx), len(test_idx))\nprint('Retained features:', len(feature_columns))\nprint('Removed constants:', constant_columns)"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """# CNN: one feature per sequence position.\nX_train_cnn = X_train_2d[..., np.newaxis]\nX_val_cnn = X_val_2d[..., np.newaxis]\nX_test_cnn = X_test_2d[..., np.newaxis]\n\n# LSTM: 57 features grouped into 19 ordered steps with 3 channels.\nassert X_train_2d.shape[1] % 3 == 0\nlstm_steps = X_train_2d.shape[1] // 3\nX_train_lstm = X_train_2d.reshape(-1, lstm_steps, 3)\nX_val_lstm = X_val_2d.reshape(-1, lstm_steps, 3)\nX_test_lstm = X_test_2d.reshape(-1, lstm_steps, 3)\n\nclasses = np.arange(len(class_names))\nbalanced = compute_class_weight('balanced', classes=classes, y=y_train)\nlstm_class_weight = {int(i): float(w) for i, w in zip(classes, balanced)}\ncnn_class_weight = {int(i): float(np.sqrt(w)) for i, w in zip(classes, balanced)}\n\nprint('CNN shape:', X_train_cnn.shape)\nprint('LSTM shape:', X_train_lstm.shape)\nprint('LSTM class weights:', lstm_class_weight)"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 3. Build and train the 1D CNN"))
    cells.append(
        nbf.v4.new_code_cell(
            """n_classes = len(class_names)\nn_features = X_train_2d.shape[1]\n\ncnn_model = Sequential([\n    Input(shape=(n_features, 1)),\n    Conv1D(24, kernel_size=3, padding='same', activation='relu'),\n    Conv1D(32, kernel_size=3, padding='same', activation='relu'),\n    MaxPooling1D(pool_size=2),\n    Flatten(),\n    Dropout(0.30),\n    Dense(48, activation='relu'),\n    Dropout(0.20),\n    Dense(n_classes, activation='softmax'),\n], name='composer_cnn')\n\ncnn_model.compile(\n    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n    loss='sparse_categorical_crossentropy',\n    metrics=['accuracy'],\n)\ncnn_model.summary()"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """def callbacks():\n    return [\n        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),\n        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),\n    ]\n\ncnn_history = cnn_model.fit(\n    X_train_cnn, y_train,\n    validation_data=(X_val_cnn, y_val),\n    epochs=30,\n    batch_size=64,\n    class_weight=cnn_class_weight,\n    callbacks=callbacks(),\n    verbose=1,\n)"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 4. Build and train the LSTM"))
    cells.append(
        nbf.v4.new_code_cell(
            """lstm_model = Sequential([\n    Input(shape=(lstm_steps, 3)),\n    LSTM(32),\n    Dropout(0.25),\n    Dense(48, activation='relu'),\n    Dropout(0.20),\n    Dense(n_classes, activation='softmax'),\n], name='composer_lstm')\n\nlstm_model.compile(\n    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n    loss='sparse_categorical_crossentropy',\n    metrics=['accuracy'],\n)\nlstm_model.summary()\n\nlstm_history = lstm_model.fit(\n    X_train_lstm, y_train,\n    validation_data=(X_val_lstm, y_val),\n    epochs=40,\n    batch_size=64,\n    class_weight=lstm_class_weight,\n    callbacks=callbacks(),\n    verbose=1,\n)"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 5. Evaluate both models on the held-out test set"))
    cells.append(
        nbf.v4.new_code_cell(
            """def evaluate_model(name, model, X_test):\n    probabilities = model.predict(X_test, verbose=0)\n    predictions = probabilities.argmax(axis=1)\n    precision, recall, f1, _ = precision_recall_fscore_support(\n        y_test, predictions, average='macro', zero_division=0\n    )\n    result = {\n        'model': name,\n        'accuracy': accuracy_score(y_test, predictions),\n        'macro_precision': precision,\n        'macro_recall': recall,\n        'macro_f1': f1,\n    }\n    print(f'\\n{name}')\n    print(classification_report(y_test, predictions, target_names=class_names, zero_division=0))\n    return result, predictions\n\ncnn_result, cnn_pred = evaluate_model('CNN', cnn_model, X_test_cnn)\nlstm_result, lstm_pred = evaluate_model('LSTM', lstm_model, X_test_lstm)\nresults_df = pd.DataFrame([cnn_result, lstm_result]).set_index('model')\ndisplay(results_df.style.format('{:.3f}'))"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 5))\nfor ax, name, pred in zip(axes, ['CNN', 'LSTM'], [cnn_pred, lstm_pred]):\n    sns.heatmap(\n        confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues', cbar=False,\n        xticklabels=class_names, yticklabels=class_names, ax=ax\n    )\n    ax.set(title=f'{name} Confusion Matrix', xlabel='Predicted', ylabel='Actual')\n    ax.tick_params(axis='x', rotation=30)\nplt.tight_layout()\nplt.show()"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(13, 8))\nfor col, (name, history) in enumerate([('CNN', cnn_history), ('LSTM', lstm_history)]):\n    axes[0, col].plot(history.history['loss'], label='Train')\n    axes[0, col].plot(history.history['val_loss'], label='Validation')\n    axes[0, col].set_title(f'{name} Loss')\n    axes[0, col].legend()\n    axes[1, col].plot(history.history['accuracy'], label='Train')\n    axes[1, col].plot(history.history['val_accuracy'], label='Validation')\n    axes[1, col].set_title(f'{name} Accuracy')\n    axes[1, col].legend()\nplt.tight_layout()\nplt.show()"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 6. Save the trained models and preprocessing objects"))
    cells.append(
        nbf.v4.new_code_cell(
            """OUTPUT_DIR = Path('trained_models')\nOUTPUT_DIR.mkdir(exist_ok=True)\ncnn_model.save(OUTPUT_DIR / 'composer_cnn.keras')\nlstm_model.save(OUTPUT_DIR / 'composer_lstm.keras')\njoblib.dump(\n    {'scaler': scaler, 'label_encoder': label_encoder, 'feature_columns': feature_columns},\n    OUTPUT_DIR / 'composer_preprocessor.joblib',\n)\nresults_df.to_csv(OUTPUT_DIR / 'model_results.csv')\nprint('Saved to:', OUTPUT_DIR.resolve())"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Interpretation

The CNN is expected to outperform the LSTM on this dataset because the CSV contains fixed, aggregate features rather than chronological note events. The convolution layers can learn local patterns across related feature groups while the flattened representation preserves each feature's position. For a more natural LSTM experiment, extract ordered MIDI note sequences—such as pitch, duration, velocity, and inter-onset interval per event—and pad or window those sequences before training."""
        )
    )
    nb["cells"] = cells
    nbf.write(nb, PROJECT_DIR / "composer_classification_lstm_cnn.ipynb")


def main() -> None:
    cnn = load_json("cnn_metrics.json")
    lstm = load_json("lstm_metrics.json")
    create_plots(cnn, lstm)
    create_metrics_csv(cnn, lstm)
    create_readme(cnn, lstm)
    create_notebook(cnn, lstm)
    print("Generated notebook, README, plots, and model summary.")


if __name__ == "__main__":
    main()
