# Control (C0) vs Coffee (C1∪C2) — Feature-comparison Experiment Design

**Ngày:** 2026-06-16
**Trạng thái:** Đã duyệt thiết kế, chuẩn bị lập kế hoạch triển khai

## 1. Mục tiêu

Phân loại nhị phân **Control (không khí, C0)** vs **Coffee (gộp Arabica + Robusta,
C1∪C2)** từ một epoch EEG 3 giây. So sánh nhiều phương pháp biểu diễn — **raw
data** và **các họ đặc trưng trích xuất** — bằng đánh giá LOSO để tìm ra biểu
diễn tốt nhất cho bài toán này.

- **Input:** một epoch EEG 3 giây (16 kênh × 300 mẫu).
- **Output:** nhãn `Control` hoặc `Coffee`.

## 2. Dữ liệu

- Dùng **8 người sạch:** P001, P014–P020 (loại P003–P013 do clipping phần cứng,
  theo `EXCLUDE_SUBJECTS` trong `src/config.py`).
- Mỗi người: 15 trial C0 (Control) + 30 trial Coffee (15 Arabica + 15 Robusta;
  P014 có 16 Arabica). → mất cân bằng lớp ~1:2 (Control:Coffee).
- Mã `code` (đã có trong config): C0 = {712, 238, 759, 869, 562}; Coffee =
  ARABICA_CODES ∪ ROBUSTA_CODES. Bỏ qua `code` 0 (ISI) và 1 (baseline).
- Mỗi trial = đoạn liên tục đúng 300 mẫu (3 s) ở 100 Hz.

## 3. Quyết định thiết kế (đã chốt)

- **Đơn vị phân loại:** 1 trial = 1 epoch 3 giây.
- **Subjects:** chỉ 8 người sạch.
- **Mất cân bằng:** giữ toàn bộ trial; classifier dùng `class_weight='balanced'`;
  báo cáo **balanced-accuracy, ROC-AUC, macro-F1** (KHÔNG dùng accuracy thô làm
  tiêu chí chính). Mốc ngẫu nhiên (balanced-accuracy) = 0.5.
- **Đánh giá:** LOSO (LeaveOneGroupOut theo subject), mọi bước fit chỉ học trên
  train fold (scaler, PCA, SelectKBest, model) — không rò rỉ.
- **Code:** merge nhánh `codex/eeg-feature-experiments` vào `master` để tái dùng
  `raw_features.py`, `engineered_features.py` và hạ tầng search; không viết lại.
- **Dùng SelectKBest** trong nhánh "selected_*". **Không** thêm deep learning ở
  bước này (YAGNI).

## 4. Merge codex vào master

- Merge `codex/eeg-feature-experiments` (đỉnh `26ebd60`) vào `master`. Hai nhánh
  rẽ từ `70cf535`; master thêm CLAUDE.md, codex thêm 13 file đặc trưng.
- Xung đột dự kiến: `.gitignore` (cả hai sửa). Giải quyết bằng hợp nhất cả hai
  thay đổi. Các file còn lại không chồng lấn.
- Sau merge: chạy `python -m pytest -q` xác nhận toàn bộ test (cũ + của codex)
  xanh trước khi xây phần mới.

## 5. Thành phần mới (nhỏ, tách bạch)

### 5.1 Cắt epoch có Control (`src/eeg_io.py`)
- Thêm `label_for_task(code, task="coffee")`:
  - `task="coffee"` (mặc định, hành vi cũ): C1→`Arabica`, C2→`Robusta`, khác→None.
  - `task="control_vs_coffee"`: C0→`Control`, C1∪C2→`Coffee`, khác (0/1)→None.
- Thêm `extract_epochs_task(signals, codes, task, expected_len)` dùng
  `label_for_task`. **Giữ nguyên** `label_for_code`/`extract_epochs` cũ để pipeline
  Arabica-vs-Robusta không đổi.

### 5.2 Dataset builder (`scripts/11_build_c0_vs_coffee.py`)
- Lặp các file `data/P*`, **bỏ subject trong `EXCLUDE_SUBJECTS`**.
- Bandpass 1–45 Hz (toàn session) + CAR mỗi epoch (tái dùng `src/preprocess.py`).
- Cắt epoch task `control_vs_coffee`; loại artifact bằng `epoch_is_artifact`
  (NaN/inf + ptp percentile, tái dùng `src/quality.py`).
- Lưu `outputs/epochs_c0_vs_coffee.npz`: `X` (n×16×300), `y` (`Control`/`Coffee`),
  `subjects`. In số epoch mỗi lớp và class balance.

### 5.3 Mở rộng đánh giá (`src/evaluate.py`)
- Thêm khóa `balanced_accuracy` (sklearn `balanced_accuracy_score`) vào dict trả
  về của `run_loso`. Không đổi các khóa hiện có (API tương thích ngược).
- Thêm factory phân loại có `class_weight='balanced'` dùng cho thí nghiệm này
  (logreg/linear-SVM/RF), đặt trong module search mới (5.5) hoặc evaluate.

## 6. Hai thí nghiệm

### 6.1 Thí nghiệm A — Raw data (`scripts/12_raw_loso_c0.py`)
- Đọc `epochs_c0_vs_coffee.npz`, dùng `raw_features.flatten_epochs` →
  `raw_features.make_raw_pca_classifiers` (PCA trong pipeline, fit trong fold).
- LOSO 8 fold; in balanced-acc/AUC/macro-F1 mỗi mô hình.

### 6.2 Thí nghiệm B — So sánh họ đặc trưng (`scripts/13_search_features_c0.py`)
- Các họ: `{bandpower, temporal_bandpower, narrow_bandpower, time, hjorth,
  engineered}` (qua `engineered_features.build_feature_family_matrix`).
- Mỗi họ × {logreg, linear-SVM, RF} với `class_weight='balanced'`, và biến thể
  **SelectKBest** (`make_selected_feature_classifiers`).
- LOSO mỗi (họ × mô hình); **xếp hạng theo balanced-accuracy rồi ROC-AUC**.
- Đầu ra: `outputs/c0_vs_coffee_results.csv` (mọi tổ hợp), và
  `outputs/c0_vs_coffee_best.png` (confusion + accuracy từng người của tổ hợp
  tốt nhất). In bảng xếp hạng + “best feature/model”.

## 7. Testing (TDD)

- `label_for_task`: C0→Control, C1/C2→Coffee với `control_vs_coffee`; hành vi cũ
  với `coffee`.
- `extract_epochs_task`: trên fixture tổng hợp đếm đúng số epoch Control/Coffee.
- `run_loso`: dict trả về có `balanced_accuracy` trong [0,1] (test tổng hợp).
- Module `raw_features`/`engineered_features`: đã có test từ codex (chạy lại đảm
  bảo không hồi quy).
- Script chạy trên dữ liệu thật để kiểm thực nghiệm (không phải unit test).

## 8. Ngoài phạm vi (YAGNI)

- Deep learning (EEGNet/CNN).
- Multi-class 3 lớp (C0/C1/C2) — bài này gộp Coffee.
- Tự động tinh chỉnh k của SelectKBest (dùng k cố định mặc định, ví dụ 80).
- Within-subject CV.
