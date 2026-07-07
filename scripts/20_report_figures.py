"""Generate figures for the report (section 4).

Outputs (saved to outputs/report/):
    fig_4_1_epoch_cutting.png       -- raw signal + colored marker strip
    fig_4_2_preprocess_stages.png   -- raw vs bandpass vs CAR for one epoch
    fig_4_3_clipping_evidence.png   -- hist + clipped trace + per-subject bar
    fig_4_4_marker_order.png        -- canonical vs EEG-marker order, P001 vs P019

Run from project root:
    python scripts/20_report_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, EEG_CHANNELS,
                        ARABICA_CODES, ROBUSTA_CODES, CONTROL_CODES,
                        SATURATION_UV, EXCLUDE_SUBJECTS)
from src.eeg_io import load_subject, find_code_runs
from src.preprocess import bandpass_filter, common_average_reference
from src.sensory import load_trial_order

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = OUTPUT_DIR / "report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOR_ARABICA = "#2a7f3f"
COLOR_ROBUSTA = "#b3331b"
COLOR_CONTROL = "#888888"


def label_for_code(code):
    if code in ARABICA_CODES:
        return "Arabica", COLOR_ARABICA
    if code in ROBUSTA_CODES:
        return "Robusta", COLOR_ROBUSTA
    if code in CONTROL_CODES:
        return "Control", COLOR_CONTROL
    return None, None


def fig_4_1():
    """Raw EEG segment with marker strip below."""
    sig, codes, sid = load_subject(DATA_DIR / "P001_KT88_with_times.csv")
    runs = find_code_runs(codes)
    coffee_starts = [s for c, s, L in runs
                     if (c in ARABICA_CODES or c in ROBUSTA_CODES
                         or c in CONTROL_CODES) and L >= EPOCH_LEN]
    if not coffee_starts:
        raise RuntimeError("no labeled runs found")
    center = coffee_starts[len(coffee_starts) // 4]
    win_start = max(0, center - 10 * FS)
    win_end = min(len(sig), win_start + 60 * FS)
    t = np.arange(win_start, win_end) / FS

    ch_idx = EEG_CHANNELS.index("C3")
    trace = sig[win_start:win_end, ch_idx]

    fig, (ax_sig, ax_mark) = plt.subplots(
        2, 1, figsize=(11, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax_sig.plot(t, trace, color="#222", lw=0.6)
    ax_sig.set_ylabel(f"{EEG_CHANNELS[ch_idx]} (µV)")
    ax_sig.set_title(f"{sid} -- 60 s segment with stimulus markers")
    ax_sig.grid(alpha=0.25)

    for code, start, length in runs:
        if start + length < win_start or start > win_end:
            continue
        lab, col = label_for_code(code)
        if lab is None:
            continue
        s = max(start, win_start) / FS
        e = min(start + length, win_end) / FS
        ax_sig.axvspan(s, e, color=col, alpha=0.18, lw=0)
        ax_mark.axvspan(s, e, color=col, alpha=0.9, lw=0)
        ax_mark.text((s + e) / 2, 0.5, str(code), ha="center", va="center",
                     fontsize=7, color="white")

    ax_mark.set_ylim(0, 1)
    ax_mark.set_yticks([])
    ax_mark.set_xlabel("Time (s)")
    ax_mark.set_ylabel("code", rotation=0, ha="right", va="center")

    handles = [
        mpatches.Patch(color=COLOR_ARABICA, alpha=0.6, label="Arabica (C1)"),
        mpatches.Patch(color=COLOR_ROBUSTA, alpha=0.6, label="Robusta (C2)"),
        mpatches.Patch(color=COLOR_CONTROL, alpha=0.6, label="Control (C0)"),
    ]
    ax_sig.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)

    out = FIG_DIR / "fig_4_1_epoch_cutting.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_4_2():
    """One Arabica epoch: raw -> bandpass -> CAR, 16 channels stacked."""
    sig, codes, sid = load_subject(DATA_DIR / "P001_KT88_with_times.csv")
    runs = find_code_runs(codes)
    target = next(((c, s) for c, s, L in runs
                   if c in ARABICA_CODES and L >= EPOCH_LEN), None)
    if target is None:
        raise RuntimeError("no Arabica epoch found")
    code, start = target

    raw_ep = sig[start:start + EPOCH_LEN].T  # (16, 300)
    filt_full = bandpass_filter(sig.T, fs=FS, low=BANDPASS_LOW,
                                high=BANDPASS_HIGH, order=FILTER_ORDER)
    bp_ep = filt_full[:, start:start + EPOCH_LEN]
    car_ep = common_average_reference(bp_ep)

    t = np.arange(EPOCH_LEN) / FS

    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    titles = ["(a) Raw", "(b) Bandpass 1-45 Hz", "(c) Bandpass + CAR"]
    data = [raw_ep, bp_ep, car_ep]
    offset = 80.0
    for ax, dat, ttl in zip(axes, data, titles):
        for ci, ch in enumerate(EEG_CHANNELS):
            ax.plot(t, dat[ci] + ci * offset, color="#222", lw=0.55)
        ax.set_title(ttl)
        ax.set_xlabel("Time in epoch (s)")
        ax.grid(alpha=0.2)
    axes[0].set_yticks([i * offset for i in range(len(EEG_CHANNELS))])
    axes[0].set_yticklabels(EEG_CHANNELS)
    axes[0].set_ylabel("Channel (offset for display)")
    fig.suptitle(f"{sid} -- one Arabica epoch (code={code}) "
                 f"through preprocessing stages",
                 y=0.995, fontsize=11)
    out = FIG_DIR / "fig_4_2_preprocess_stages.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _saturation_fraction(sig, thresh=SATURATION_UV):
    arr = np.asarray(sig, dtype=float)
    return float(np.mean(np.abs(arr) >= thresh))


def fig_4_3():
    """Clipping evidence: histogram + clipped trace + per-subject bar."""
    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 16, "legend.fontsize": 15,
    })
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))
    per_subj = []
    sig_clean, sig_clip = None, None
    sid_clean, sid_clip = "P001", "P006"
    for f in files:
        sig, codes, sid = load_subject(f)
        per_subj.append((sid, _saturation_fraction(sig) * 100.0))
        if sid == sid_clean:
            sig_clean = sig
        if sid == sid_clip:
            sig_clip = sig

    per_subj.sort(key=lambda x: x[0])
    subj_ids = [s for s, _ in per_subj]
    sat_pct = [v for _, v in per_subj]

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-250, 250, 121)
    ch_idx = EEG_CHANNELS.index("F3")
    ax_a.hist(sig_clean[:, ch_idx], bins=bins, color="#2a6fb3",
              alpha=0.7, label=f"{sid_clean} (clean)", density=True)
    ax_a.hist(sig_clip[:, ch_idx], bins=bins, color="#b3331b",
              alpha=0.6, label=f"{sid_clip} (clipped)", density=True)
    ax_a.axvline(SATURATION_UV, color="k", ls="--", lw=0.8)
    ax_a.axvline(-SATURATION_UV, color="k", ls="--", lw=0.8)
    ax_a.set_title(f"(a) Amplitude distribution, channel {EEG_CHANNELS[ch_idx]}")
    ax_a.set_xlabel("Amplitude (µV)")
    ax_a.set_ylabel("Density")
    ax_a.legend()
    ax_a.grid(alpha=0.25)

    ax_b = fig.add_subplot(gs[0, 1])
    seg = sig_clip[5 * FS:10 * FS, ch_idx]
    t = np.arange(seg.size) / FS
    ax_b.plot(t, seg, color="#b3331b", lw=0.7)
    ax_b.axhline(SATURATION_UV, color="k", ls="--", lw=0.7)
    ax_b.axhline(-SATURATION_UV, color="k", ls="--", lw=0.7)
    ax_b.set_title(f"(b) {sid_clip} raw, {EEG_CHANNELS[ch_idx]} (clipped)")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Amplitude (µV)")
    ax_b.grid(alpha=0.25)

    ax_c = fig.add_subplot(gs[1, :])
    colors = ["#b3331b" if s in EXCLUDE_SUBJECTS else "#2a6fb3"
              for s in subj_ids]
    bars = ax_c.bar(subj_ids, sat_pct, color=colors, edgecolor="black", lw=0.4)
    ax_c.axhline(1.0, color="k", ls="--", lw=0.8, label="1% threshold")
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Saturated samples (%)")
    ax_c.set_title("(c) Per-subject saturation rate (log scale)")
    ax_c.legend(loc="upper right")
    ax_c.grid(alpha=0.25, axis="y", which="both")
    plt.setp(ax_c.get_xticklabels(), rotation=45, ha="right")

    out = FIG_DIR / "fig_4_3_clipping_evidence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _eeg_order(codes):
    """All Arabica/Robusta/Control codes in run-length order."""
    out = []
    for code, start, length in find_code_runs(codes):
        if (code in ARABICA_CODES or code in ROBUSTA_CODES
                or code in CONTROL_CODES) and length >= EPOCH_LEN:
            out.append(code)
    return out


def _strip(ax, codes, row_label):
    n = len(codes)
    for i, c in enumerate(codes):
        _, col = label_for_code(c)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                   facecolor=col or "white",
                                   edgecolor="white", lw=0.4))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(row_label, rotation=0, ha="right", va="center",
                  fontsize=9, labelpad=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def fig_4_4():
    """Canonical vs EEG-marker order: P001 (match) vs P019 (mismatch)."""
    order = load_trial_order(ROOT / "protocol" / "experiment_sequences.xlsx")

    fig, axes = plt.subplots(4, 1, figsize=(12, 4.2), sharex=True,
                             gridspec_kw={"hspace": 0.25})

    subjects = ["P001", "P019"]
    for row, sid in enumerate(subjects):
        sig, codes, _ = load_subject(DATA_DIR / f"{sid}_KT88_with_times.csv")
        eeg_seq = _eeg_order(codes)
        canon = order[sid][:len(eeg_seq)]
        ax_top = axes[2 * row]
        ax_bot = axes[2 * row + 1]
        _strip(ax_top, canon, "canonical")
        _strip(ax_bot, eeg_seq, "EEG marker")
        # Subject label to the far left, spanning the two rows
        ax_top.annotate(sid, xy=(-0.11, -0.05), xycoords="axes fraction",
                        fontsize=12, fontweight="bold", ha="right",
                        va="center")

    axes[-1].set_xlabel("Trial index (time order)")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    handles = [
        mpatches.Patch(color=COLOR_ARABICA, label="Arabica"),
        mpatches.Patch(color=COLOR_ROBUSTA, label="Robusta"),
        mpatches.Patch(color=COLOR_CONTROL, label="Control"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), fontsize=9, frameon=False)

    fig.suptitle("Canonical trial order vs EEG `code` markers",
                 y=1.09, fontsize=11)
    out = FIG_DIR / "fig_4_4_marker_order.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()


if __name__ == "__main__":
    main()
