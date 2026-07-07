"""Raw signal and power spectrum: clean vs clipped subject.

Demonstrates that hardware clipping is not a cosmetic amplitude issue but a
signature of contamination by noise far larger than physiological EEG. The
clipped recording shows broadband elevation of spectral power and harmonic
distortion typical of non-linear saturation, while the clean recording stays
within physiological amplitude (~50 uV) and shows the expected 1/f shape.

Output:
    outputs/report/fig_4_5_raw_psd.png

Run from project root:
    python scripts/22_signal_quality_figure.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, OUTPUT_DIR, FS, EEG_CHANNELS, SATURATION_UV
from src.eeg_io import load_subject


def welch_psd(x, fs, nperseg, noverlap=None):
    """Welch PSD with Hann window, in-house to avoid the scipy import chain."""
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    win = np.hanning(nperseg)
    norm = (win * win).sum() * fs
    n = (len(x) - nperseg) // step + 1
    psd = np.zeros(nperseg // 2 + 1)
    for i in range(n):
        seg = x[i * step:i * step + nperseg]
        seg = seg - seg.mean()
        spec = np.fft.rfft(seg * win)
        psd += (spec.conj() * spec).real
    psd /= n * norm
    psd[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd

FIG_DIR = OUTPUT_DIR / "report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_ID = "P001"
CLIP_ID = "P006"
CHANNEL = "F3"
WIN_SEC = 8            # seconds of raw trace to display

COLOR_CLEAN = "#2a6fb3"
COLOR_CLIP = "#b3331b"


def _welch_psd(x, fs=FS):
    nperseg = min(int(4 * fs), len(x))   # 4 s windows
    return welch_psd(x, fs=fs, nperseg=nperseg)


def _quietest_window(x, win):
    """Return the start index of the win-sample window with the smallest std."""
    if len(x) <= win:
        return 0
    stride = max(1, win // 4)
    best_i, best_s = 0, np.inf
    for i in range(0, len(x) - win, stride):
        s = np.std(x[i:i + win])
        if s < best_s:
            best_s = s
            best_i = i
    return best_i


def main():
    sig_clean, _, _ = load_subject(DATA_DIR / f"{CLEAN_ID}_KT88_with_times.csv")
    sig_clip, _, _ = load_subject(DATA_DIR / f"{CLIP_ID}_KT88_with_times.csv")

    ch = EEG_CHANNELS.index(CHANNEL)
    win = WIN_SEC * FS
    s_cln = _quietest_window(sig_clean[:, ch], win)
    s_clp = _quietest_window(sig_clip[:, ch], win)
    trace_clean = sig_clean[s_cln:s_cln + win, ch]
    trace_clip = sig_clip[s_clp:s_clp + win, ch]
    t = np.arange(win) / FS

    f_cln, p_cln = _welch_psd(sig_clean[:, ch])
    f_clp, p_clp = _welch_psd(sig_clip[:, ch])

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })
    y_lim = (-260, 260)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2),
                             gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    ax.plot(t, trace_clean, color=COLOR_CLEAN, lw=0.9)
    ax.axhline(SATURATION_UV, color="k", ls="--", lw=0.7)
    ax.axhline(-SATURATION_UV, color="k", ls="--", lw=0.7)
    ax.set_ylim(*y_lim)
    ax.set_ylabel(f"{CHANNEL} ({chr(956)}V)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"(a) {CLEAN_ID} raw (clean)")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, trace_clip, color=COLOR_CLIP, lw=0.9)
    ax.axhline(SATURATION_UV, color="k", ls="--", lw=0.7)
    ax.axhline(-SATURATION_UV, color="k", ls="--", lw=0.7)
    ax.set_ylim(*y_lim)
    ax.set_ylabel(f"{CHANNEL} ({chr(956)}V)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"(b) {CLIP_ID} raw (clipped)")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.loglog(f_cln, p_cln, color=COLOR_CLEAN, lw=1.8, label=f"{CLEAN_ID} clean")
    ax.loglog(f_clp, p_clp, color=COLOR_CLIP, lw=1.8, alpha=0.9,
              label=f"{CLIP_ID} clipped")
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        ax.axvspan(lo, hi, color="k", alpha=0.04)
    ax.set_xlim(0.5, 50)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD ({chr(956)}V$^2$/Hz)")
    ax.set_title("(c) Welch power spectrum")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left")

    out = FIG_DIR / "fig_4_5_raw_psd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    ratio_band = {}
    for lo, hi, name in [(1, 4, "delta"), (4, 8, "theta"), (8, 13, "alpha"),
                         (13, 30, "beta"), (30, 45, "gamma")]:
        m = (f_cln >= lo) & (f_cln < hi)
        ratio_band[name] = float(
            np.trapezoid(p_clp[m], f_cln[m])
            / max(np.trapezoid(p_cln[m], f_cln[m]), 1e-12))
    print("PSD ratio (clipped / clean) per band:")
    for k, v in ratio_band.items():
        print(f"  {k:6s}: x{v:6.1f}")


if __name__ == "__main__":
    main()
