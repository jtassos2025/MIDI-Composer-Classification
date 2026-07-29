# Composer Classification Feature Dataset

## Overview

This project classifies MIDI files by composer: Bach, Beethoven, Chopin, and Mozart. Each feature row represents one MIDI file and contains three identifiers plus 60 numeric musical descriptors.

The project keeps the original `midiclassics` dataset intact, adds newly found MIDI sources separately, filters feature-level duplicates, and produces train/dev/test CSVs for modeling.

## Data layout

```text
data/
|-- midiclassics/                  # Preferred original MIDI library by composer
|-- new_midi_files/                # Add each new MIDI source as a direct subfolder here
|-- features/
|   |-- original features/         # Original train/dev/test feature CSVs
|   |-- new_train_features.csv     # Incremental features for accepted new MIDI files
|   |-- new_dev_features.csv
|   |-- new_test_features.csv
|   |-- combined_train_features.csv
|   |-- combined_dev_features.csv
|   `-- combined_test_features.csv
`-- split_plans/
    `-- candidate_split_plan.csv   # Duplicate decisions and deterministic split assignments
```

Run notebooks from the `notebooks/` directory so their `../data/...` paths resolve correctly:

```powershell
Set-Location notebooks
jupyter notebook
```

Install dependencies first:

```powershell
pip install -r requirements.txt
```

## Adding new MIDI data

1. Create or download a source folder under `data/new_midi_files/`.
2. Ensure MIDI filenames or enclosing folders identify Beethoven, Chopin, or Mozart.
3. Run [`MIDI_Ingest.ipynb`](notebooks/MIDI_Ingest.ipynb).
4. Run [`Generate_New_MIDI_Features.ipynb`](notebooks/Generate_New_MIDI_Features.ipynb).
5. Run [`Merge_Feature_CSVs.ipynb`](notebooks/Merge_Feature_CSVs.ipynb).

`MIDI_Ingest.ipynb` automatically discovers every direct source folder under `data/new_midi_files`. It compares duration, estimated tempo, note count, and notes per second within each composer. `midiclassics` remains the preferred copy when a duplicate is found.

The split plan assigns one of four statuses:

- `duplicate_of_midiclassics`: exclude; retain the preferred library copy.
- `duplicate_of_candidate`: exclude; another new-source file represents the group.
- `keep_candidate_representative`: retain one representative of a new duplicate group.
- `keep_unique`: retain; no matching feature signature was found.

Only `keep_*` rows are eligible for feature generation. `recommended_split` is deterministic from the composer and feature signature, using a 70%/15%/15% train/dev/test target.

## Feature generation and merging

[`Generate_New_MIDI_Features.ipynb`](notebooks/Generate_New_MIDI_Features.ipynb) reads the split plan and maintains only three new-only files:

- `new_train_features.csv`
- `new_dev_features.csv`
- `new_test_features.csv`

It uses the same 63-column schema as the original dataset. On later runs it reuses rows already present in those files and calculates expensive features only for newly accepted MIDI files.

[`Merge_Feature_CSVs.ipynb`](notebooks/Merge_Feature_CSVs.ipynb) validates matching schemas, rejects duplicate `(composer, relative_path)` rows, sorts by composer, and writes the three `combined_*_features.csv` files. It does not modify either input set.

## Feature columns

Identifier columns:

- `composer`: classification target; exclude from model inputs.
- `filename`: source filename; traceability only.
- `relative_path`: source path under `data`; traceability and duplicate checks only.

Musical features include tempo, note and chord counts, pitch-class statistics, duration and velocity statistics, rhythmic density, onset intervals, polyphony, MIDI-track counts, and General MIDI instrument-family track counts. The full schema contains 63 columns.

Some MIDI files contain malformed tempo, key, or time-signature events. Parsing continues when possible, but source metadata may be imperfect.

## Analysis notebooks

- [`MIDI_EDA.ipynb`](notebooks/MIDI_EDA.ipynb) analyzes the original feature splits.
- [`Combined_MIDI_EDA.ipynb`](notebooks/Combined_MIDI_EDA.ipynb) has the same EDA structure but uses the three combined CSVs.

For modeling, exclude `composer`, `filename`, `relative_path`, and the EDA-only `split` column from model inputs. Evaluate class-imbalanced models with macro F1, balanced accuracy, per-composer recall, and confusion matrices.

## Model project

`composer_model_project/` contains the CNN/LSTM experiment and its current artifacts. Before re-evaluating it on the expanded data, update its preprocessing and training workflow to consume the three `combined_*_features.csv` files directly, preserving their train/dev/test assignments rather than creating a new random split.
