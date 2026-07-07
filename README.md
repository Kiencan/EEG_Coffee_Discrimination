# EEG Coffee Discrimination

Classifying **Arabica vs Robusta coffee smell** from EEG signals recorded
while sniffing (binary classification). Classification unit = one 3-second
sniff epoch.

- **Input:** one 3-second EEG epoch (16 channels × 300 samples).
- **Output:** `Arabica` or `Robusta` label.

## Data

The raw data (`data/P0XX_KT88_with_times.csv`, 18 subjects, ~1.1GB) is not
stored in this repo (exceeds GitHub's file size limits). Download it here:

**[Google Drive — data/](https://drive.google.com/file/d/1UA3_y2SPQtY4IkyRKGciI1wVllG4WEiG/view?usp=sharing)**

After downloading, extract it into a `data/` folder at the project root
(same level as `src/`, `scripts/`).

For data format details, stimulus codes, and quality notes (clipping,
`EXCLUDE_SUBJECTS`), see [CLAUDE.md](CLAUDE.md).

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

From the project root:

```bash
python -m pytest -q                       # run the full test suite
python scripts/01_data_quality.py         # data quality report
python scripts/02_preprocess_epochs.py    # export cleaned epochs
python scripts/03_features.py             # build feature matrix
python scripts/04_train_loso.py           # train + evaluate with LOSO
```

`outputs/` (npz files, figures, result CSVs) is gitignored — it's a
reproducible artifact of the data, not committed.

## Project structure

```
src/config.py     # stimulus codes, frequency bands, thresholds, EXCLUDE_SUBJECTS, paths
src/eeg_io.py     # CSV loading, run-length detection, epoch cutting
src/quality.py    # quality checks, artifact epoch rejection
src/preprocess.py # zero-phase 1-45 Hz bandpass + CAR
src/features.py   # Welch band power (delta/theta/alpha/beta/gamma)
src/evaluate.py   # LOSO CV (LeaveOneGroupOut by subject) + metrics

scripts/          # sequential pipeline, 01 -> 22
tests/            # tests for each module in src/
docs/              # design spec and plan
protocol/          # original experiment protocol + sensory data
paper/             # paper draft
```

## License

[Apache License 2.0](LICENSE)
