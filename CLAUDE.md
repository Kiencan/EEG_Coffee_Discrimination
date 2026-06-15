# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong dự án này.

## Dự án

Phân loại **mùi cà phê Arabica vs Robusta** từ tín hiệu EEG khi ngửi (phân loại
nhị phân). Đơn vị phân loại = một epoch ngửi 3 giây. Bỏ qua tín hiệu control
(không khí) và baseline — chỉ dùng epoch ngửi cà phê.

- **Input:** một epoch EEG 3 giây (16 kênh × 300 mẫu).
- **Output:** nhãn `Arabica` hoặc `Robusta`.

## Dữ liệu

- File: `data/P0XX_KT88_with_times.csv` — **18 người** (P001–P020, **thiếu P002
  và P008** vì không thu).
- Máy KT88, **100 Hz**, ADC 12-bit, full-scale **±204.8 µV**.
- Cột: `times, timestamp, Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3,
  T4, T5, T6, ECG1, ECG2, code`. → 16 kênh EEG + 2 kênh ECG (bỏ) + marker.
- Đơn vị µV.
- Cột `code` đánh dấu kích thích:
  - `0` = nghỉ/ISI; `1` = baseline 2000 ms trước trial (đều bỏ qua).
  - **Arabica (C1):** 981, 633, 902, 598, 733
  - **Robusta (C2):** 585, 597, 200, 558, 692
  - **Control (C0, bỏ qua):** 712, 238, 759, 869, 562
- Mỗi mã odor xuất hiện thành đoạn liên tục đúng **300 mẫu (3 s)** = 1 trial; ~15
  Arabica + 15 Robusta mỗi người (P014 có 16 Arabica — 1 trial dư hợp lệ).

### Lưu ý chất lượng dữ liệu (quan trọng)

- **P003–P013 bị clipping phần cứng** (over-amplified): 2.3–17% mẫu chạm rail
  ±204.8 µV, so với <0.4% ở nhóm sạch. Clipping trùng theo từng người →
  **confound cho LOSO**. Mặc định các người này bị loại qua
  `EXCLUDE_SUBJECTS` trong `src/config.py`. Nhóm sạch = P001, P014–P020 (8 người).
- **P014** từng có 2 hàng rỗng (toàn NaN) ở cuối file → `load_subject` tự bỏ
  hàng rỗng nên P014 được giữ lại.
- Ngưỡng `SATURATION_UV = 204.0` là CỐ Ý (sát rail) — đừng đổi.

## Kiến trúc

Module nhập được trong `src/` + script chạy trong `scripts/`. Mỗi module một
trách nhiệm, có test riêng trong `tests/`.

```
src/config.py     # mã code, dải tần, ngưỡng, EXCLUDE_SUBJECTS, đường dẫn
src/eeg_io.py     # load CSV (bỏ ECG + hàng rỗng), run-length, cắt epoch
src/quality.py    # độ đều lấy mẫu, kênh chết, loại epoch artifact (ptp + NaN/inf)
src/preprocess.py # bandpass 1–45 Hz zero-phase + common average reference (CAR)
src/features.py   # band power Welch (delta/theta/alpha/beta/gamma), abs + rel
src/evaluate.py   # LOSO CV (LeaveOneGroupOut theo subject) + metrics

scripts/01_data_quality.py      # báo cáo chất lượng -> outputs/quality_report.csv
scripts/02_preprocess_epochs.py # lọc + CAR + loại artifact -> outputs/epochs.npz
scripts/03_features.py          # ma trận đặc trưng -> outputs/features.npz
scripts/04_train_loso.py        # LOSO -> outputs/loso_results.csv, loso_figures.png
```

Luồng chạy tuần tự: `01 → 02 → 03 → 04`. Mỗi script tự thêm project root vào
`sys.path` để import `src`.

## Quy ước & cách làm

- **Đánh giá = LOSO (subject-independent)**: train trên N-1 người, test người còn
  lại. Mọi bước fit (scaler, model) chỉ học trên train fold — KHÔNG để rò rỉ dữ
  liệu giữa các trial cùng người. Đừng đổi sang chia ngẫu nhiên theo trial.
- **TDD**: viết test trước (đỏ) → code tối thiểu (xanh) → commit. Test dùng
  fixture tổng hợp; script thì kiểm trên dữ liệu thật.
- Filter áp lên TOÀN session trước khi cắt epoch (tránh artifact mép trong epoch).
- Báo cáo kết quả TRUNG THỰC, kể cả khi ở mức ngẫu nhiên — không tô vẽ accuracy.

## Lệnh thường dùng

Chạy từ thư mục gốc dự án (`machine_learning/`):

```bash
pip install -r requirements.txt
python -m pytest -q                       # chạy toàn bộ test (KHÔNG chạy trong tests/)
python scripts/01_data_quality.py         # báo cáo chất lượng
python scripts/02_preprocess_epochs.py    # xuất epoch sạch
python scripts/03_features.py             # build đặc trưng
python scripts/04_train_loso.py           # train + đánh giá LOSO
```

`outputs/` bị gitignore (chứa .npz, hình, CSV kết quả) — coi là sản phẩm tái tạo
được, không commit.

## Môi trường

- Python 3.14, **NumPy 2.x** — dùng `np.trapezoid` (KHÔNG phải `np.trapz`, đã bị
  gỡ). Dùng `np.nanpercentile`/`np.isfinite` để bền với NaN/inf.
- scikit-learn ≥1.9: `SVC(probability=True)` đang phát cảnh báo deprecation (sẽ
  bị bỏ ở 1.11) — chỉ là warning, chưa ảnh hưởng kết quả.
- Windows; chạy lệnh `python -m pytest` từ project root để import `src` được.

## Kết quả hiện tại (tham khảo)

LOSO trên 8 người sạch (238 epoch, cân bằng): Random Forest acc≈0.55, SVM≈0.51,
LogReg≈0.47 — **về cơ bản ở mức ngẫu nhiên**. Band power + ML cổ điển chưa cho
tín hiệu tổng quát hóa đáng tin giữa các người. Hướng tiếp: within-subject,
đặc trưng ERP/connectivity, hoặc deep learning (EEGNet).

## Tài liệu thiết kế

- Spec: `docs/superpowers/specs/2026-06-11-eeg-coffee-arabica-robusta-design.md`
- Plan: `docs/superpowers/plans/2026-06-11-eeg-coffee-arabica-robusta.md`
- Protocol gốc: `protocol/EEG_Coffee_Protocol.md`
