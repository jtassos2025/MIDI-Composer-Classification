# Composer Classification Feature Dataset

## Overview

This project classifies MIDI files from four classical composers: Bach, Beethoven, Chopin, and Mozart. Each row in the feature datasets represents one MIDI file.

Generated feature splits:

- `data/features/train_features.csv`
- `data/features/dev_features.csv`
- `data/features/test_features.csv`

The current datasets contain 63 columns: three identifier columns and 60 numeric descriptors suitable for model input.

## Repository Layout

```text
.
|-- data/
|   |-- midiclassics/       # Source MIDI files, organized by composer
|   `-- features/           # train, dev, and test feature CSVs
|       |-- dev_features
|       `-- test_features
|       `-- train_features
|-- notebooks/
|   |-- MIDI_Extract.ipynb  # Feature extraction and idempotent enrichment
|   `-- MIDI_EDA.ipynb      # Exploratory data analysis
|-- requirements.txt
`-- README.md
```

Run the notebooks from the `notebooks/` directory. Their paths intentionally use `../data/...` so they resolve to the dataset folders at the repository root:

```powershell
Set-Location notebooks
jupyter notebook
```

Install the project dependencies first:

```powershell
pip install -r requirements.txt
```


## Dataset Splits and Balancing

The training split is imbalanced:

| Composer | MIDI files |
| --- | ---: |
| Bach | 716 |
| Mozart | 179 |
| Beethoven | 148 |
| Chopin | 95 |

For training, retain every original MIDI file, use class-weighted loss and balanced sampling, and evaluate with macro F1, balanced accuracy, and per-composer metrics. Do not randomly delete Bach files; add legitimate, varied MIDI files for Chopin first, then Beethoven and Mozart.


## Identifier Columns

- `composer`: classification target. Do not include it as a model input.
- `filename`: original MIDI filename. Use for traceability only, not model input.
- `relative_path`: source path within `data/midiclassics`. Use for traceability and duplicate checks only, not model input.


## Existing Musical Features

### Global note and timing descriptors

- `tempo`, `num_notes`, `num_chords`, `avg_pitch`, `pitch_range`
- `avg_duration`, `avg_velocity`, `note_density`
- `total_duration`, `num_midi_tracks`, `num_note_tracks`
- `num_unique_programs`, `drum_track_count`, `has_drums`

`num_midi_tracks` includes every raw MIDI track, including metadata tracks. `num_note_tracks` counts only instrument tracks containing notes.

### Pitch and dynamics descriptors

- `pitch_class_0` through `pitch_class_11`: normalized pitch-class histogram.
- `pitch_entropy`, `pitch_class_variance`, `range_normalized`
- `pitch_std`, `pitch_median`
- `velocity_std`, `velocity_range`
- `duration_std`, `duration_median`

### Rhythm and texture descriptors

- `notes_per_chord`, `chord_density`
- `velocity_variation`, `tempo_note_ratio`, `chromatic_ratio`
- `onset_interval_mean`, `onset_interval_std`
- `max_polyphony`, `avg_polyphony`

`velocity_variation` and `tempo_note_ratio` are legacy ratio features. `velocity_std`, onset intervals, and polyphony are usually more interpretable rhythm and texture measures. `chromatic_ratio` is the proportion of black-key pitch classes, not a key-aware measure of chromaticism.

### Instrumentation descriptors

The following General MIDI program-family columns count note tracks assigned to each family:

- `gm_piano_track_count`, `gm_chromatic_percussion_track_count`, `gm_organ_track_count`, `gm_guitar_track_count`
- `gm_bass_track_count`, `gm_strings_track_count`, `gm_ensemble_track_count`, `gm_brass_track_count`
- `gm_reed_track_count`, `gm_pipe_track_count`, `gm_synth_lead_track_count`, `gm_synth_pad_track_count`
- `gm_synth_effects_track_count`, `gm_ethnic_track_count`, `gm_percussive_track_count`, `gm_sound_effects_track_count`

These metadata can be useful but depend on the quality of the MIDI arrangement and program assignments.


## Extraction Workflow

[`notebooks/MIDI_Extract.ipynb`](notebooks/MIDI_Extract.ipynb) has two stages:

1. Create the base feature CSVs only when one or more of the three split files is absent. This slow stage performs MIDI parsing and `music21` chord analysis.
2. Enrich the existing CSVs in a single idempotent cell. For each feature column, it skips work when that column already exists and is populated for every row. When new features are added later, only missing or incomplete columns are written; the existing train/dev/test assignments are preserved.

Some MIDI files have malformed tempo, key, or time-signature events. Feature extraction continues, but those metadata values may be imperfect for those files.


## Exploratory Data Analysis

[`notebooks/MIDI_EDA.ipynb`](notebooks/MIDI_EDA.ipynb) provides a clean, split-aware review of the 63-column dataset. It includes data-quality checks, class-balance plots, pitch/rhythm/texture comparisons, MIDI instrumentation summaries, composer feature profiles, correlation checks, and train-only univariate feature-ranking scores.

For modeling, exclude `composer`, `filename`, `relative_path`, and the EDA-only `split` column from inputs. Treat file-level velocity and MIDI program features carefully: they may reflect source encoding or arrangement conventions as well as composer style.
