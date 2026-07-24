# LSTM and CNN Composer Classification

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
| CNN | 0.886 | 0.810 | 0.815 | 0.809 |
| LSTM | 0.719 | 0.604 | 0.657 | 0.625 |

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
