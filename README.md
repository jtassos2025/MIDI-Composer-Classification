## Overview

This dataset contains numerical features extracted from MIDI files from nine classical composers:

- Bach
- Bartók
- Byrd
- Chopin
- Handel
- Hummel
- Mendelssohn
- Mozart
- Schumann

Each row represents one MIDI file. The `composer` column is the target label used for classification.

Generated files:

- data/train_features.csv
- data/dev_features.csv
- data/test_features.csv


## Feature Descriptions

### Metadata Features

composer  
- Composer label (classification target).

filename  
- Original MIDI filename.  
- Used for reference only and should not be included as a model input feature.


## General Musical Features

**tempo**  
- Estimated tempo of the MIDI file in beats per minute (BPM).
- Represents the speed of the musical performance.

**num_notes**  
- Total number of notes in the MIDI file.
- Represents the overall amount of musical activity.

**num_chords**  
- Number of detected chords using `music21` chordification.
- Represents harmonic activity.

**avg_pitch**  
- Average MIDI pitch value of all notes.
- Represents the overall register of the composition.

**pitch_range**  
- Difference between the highest and lowest note.
- Represents the total pitch span of the composition.

**avg_duration**  
- Average note duration in seconds.
- Represents rhythmic characteristics and note length tendencies.

**avg_velocity**  
- Average MIDI velocity value.
- Represents average note intensity/dynamics.

**note_density**  
- Number of notes divided by total MIDI duration.
- Represents how musically dense or active a piece is.


## Pitch Distribution Features

The MIDI pitch classes represent the 12 chromatic notes:

pitch_class_0 = C  
pitch_class_1 = C#/Db  
pitch_class_2 = D  
pitch_class_3 = D#/Eb  
pitch_class_4 = E  
pitch_class_5 = F  
pitch_class_6 = F#/Gb  
pitch_class_7 = G  
pitch_class_8 = G#/Ab  
pitch_class_9 = A  
pitch_class_10 = A#/Bb  
pitch_class_11 = B  

- Normalized histogram of pitch usage.
- Each value represents the proportion of notes belonging to that pitch class.
- Captures tonal and harmonic tendencies of each composition.


## Derived Features

**range_normalized**

- Formula:
  pitch_range / avg_pitch

- Normalizes pitch range relative to the average register.


**notes_per_chord**

- Formula:
  num_notes / num_chords

- Represents the amount of note activity occurring per harmonic event.


**chord_density**

- Formula:
  num_chords / num_notes

- Represents the frequency of harmonic changes relative to note activity.


**velocity_variation**

- Formula:
  avg_velocity / tempo

- Represents the relationship between dynamics and tempo.


**tempo_note_ratio**

- Formula:
  tempo / num_notes

- Represents the relationship between performance speed and note activity.


**chromatic_ratio**

- Percentage of notes belonging to chromatic pitch classes.
- Measures the amount of chromatic pitch usage and harmonic complexity.


**pitch_entropy**

- Measures how evenly distributed the pitch classes are.
- Higher values indicate more varied pitch usage.
- Lower values indicate stronger concentration around certain pitches.


**pitch_class_variance**

- Measures the variance of pitch class usage.
- Higher values indicate stronger differences between frequently and rarely used pitch classes.


## Pipeline

MIDI Files

↓

Feature extraction using `pretty_midi`, Chord analysis using `music21`, Feature calculation, & CSV feature datasets

↓

Machine learning composer classification