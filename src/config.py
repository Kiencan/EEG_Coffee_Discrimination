"""Central configuration for the EEG coffee classification pipeline."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Sampling
FS = 100  # Hz (confirmed: timestamp diff ~0.01s)
EPOCH_LEN = 300  # samples = 3 s sniff window

# Channels (KT88): 16 EEG + 2 ECG dropped
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
ECG_CHANNELS = ["ECG1", "ECG2"]

# Stimulus codes (from protocol)
ARABICA_CODES = {981, 633, 902, 598, 733}   # C1
ROBUSTA_CODES = {585, 597, 200, 558, 692}   # C2
CONTROL_CODES = {712, 238, 759, 869, 562}   # C0 (ignored)

# Frequency bands (Hz); high capped < Nyquist (50 Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Preprocessing
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
FILTER_ORDER = 4
USE_CAR = True  # common average reference

# Quality thresholds
SATURATION_UV = 204.0      # KT88 appears to clip near +/-204.8 uV
FLAT_STD_UV = 0.5          # channel/epoch with std below this is "flat"
# Epoch artifact threshold calibrated from data; fallback fixed value:
EPOCH_PTP_PERCENTILE = 99  # reject epochs above this ptp percentile

# Subjects excluded from the training set.
# P003-P013 are hardware-clipped (over-amplified): 2.3-17% of samples railed at
# +/-204.8 uV, vs <0.4% for the clean group -- a 1% saturation threshold cleanly
# separates the two. Clipping is confounded with subject, so these are excluded
# to avoid corrupting LOSO. (P014's only defect was 2 trailing all-NaN rows,
# which load_subject drops automatically; it stays in the clean group.)
EXCLUDE_SUBJECTS = {"P003", "P004", "P005", "P006", "P007", "P009",
                    "P010", "P011", "P012", "P013"}
