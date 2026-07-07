# Phân loại Arabica vs Robusta từ tín hiệu EEG khi ngửi cà phê — Thiết kế

**Ngày:** 2026-06-11
**Trạng thái:** Đã duyệt thiết kế, chuẩn bị lập kế hoạch triển khai

## 1. Mục tiêu

Xây dựng pipeline machine learning phân loại nhị phân **Arabica vs Robusta** từ
một epoch tín hiệu EEG thu được khi người tham gia ngửi cà phê (cửa sổ ngửi 3
giây). Bỏ qua tín hiệu control (không khí C0) và baseline — chỉ tập trung vào
epoch ngửi cà phê.

- **Input:** một epoch EEG 3 giây (16 kênh × 300 mẫu).
- **Output:** nhãn `Arabica` hoặc `Robusta`.

## 2. Bối cảnh dữ liệu

- **18 người tham gia:** P001–P020, thiếu P002 và P008.
- File: `data/P0XX_KT88_with_times.csv`, ~390k mẫu/file.
- **Tần số lấy mẫu:** 100 Hz (chênh lệch `timestamp` ≈ 0.01 s). Nyquist = 50 Hz.
- **Cột:** `times, timestamp, Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8,
  T3, T4, T5, T6, ECG1, ECG2, code` (16 kênh EEG + 2 ECG + marker).
- **Đơn vị:** microvolt (giá trị thô tới ~±200 µV).
- **Mã kích thích (`code`):**
  - `0` = nghỉ/ISI; `1` = baseline 2000 ms trước mỗi trial (bỏ qua).
  - **Arabica (C1):** 981, 633, 902, 598, 733
  - **Robusta (C2):** 585, 597, 200, 558, 692
  - **Control (C0, bỏ qua):** 712, 238, 759, 869, 562
- Mỗi điều kiện 15 trial × 300 mẫu (3 s) cho mỗi người.
- Dữ liệu mục tiêu lý thuyết: 18 người × (15 Arabica + 15 Robusta) ≈ **540 epoch**
  trước khi loại artifact.

## 3. Quyết định thiết kế (đã chốt)

- **Đơn vị phân loại:** 1 trial = 1 epoch 3 giây.
- **Đặc trưng + mô hình:** band power + ML cổ điển (không deep learning ở bước này).
- **Đánh giá:** Leave-One-Subject-Out (subject-independent) — mô hình phải tổng
  quát hóa cho người MỚI, tránh rò rỉ dữ liệu giữa các trial cùng người.

## 4. Pipeline

### 4.1 Kiểm tra & lọc chất lượng dữ liệu

**Tầng file/người tham gia:**
- Xác nhận 18 file, đồng nhất 21 cột, tần số ~100 Hz (kiểm tra độ đều `timestamp`,
  phát hiện mất mẫu/gián đoạn).
- Đếm số trial mỗi điều kiện/người (kỳ vọng 15 Arabica + 15 Robusta); báo cáo
  người thiếu/thừa.

**Tầng kênh:**
- Phát hiện kênh "chết": flat-line (phương sai ≈ 0), giá trị hằng, saturation/railing.
- Thống kê biên độ mỗi kênh; gắn cờ kênh nhiễu bất thường.

**Tầng epoch:**
- Trích epoch từ cột `code`: gom đoạn liên tục cùng mã odor → mỗi đoạn phải đúng
  300 mẫu. Cắt/loại đoạn không đủ độ dài.
- Loại artifact theo epoch:
  - Ngưỡng biên độ peak-to-peak (nghi blink/cử động).
  - Epoch phương sai quá thấp (mất tiếp xúc điện cực).
  - Ngưỡng **calibrate dựa trên phân bố biên độ thực tế** (percentile), không cứng nhắc.
- ECG1/ECG2: bỏ qua (chỉ dùng tùy chọn để kiểm tra, không đưa vào đặc trưng).

**Đầu ra:** báo cáo chất lượng (CSV + hình) — số epoch hợp lệ/bị loại mỗi người,
kênh xấu, lý do loại. Đây là bộ "lọc dữ liệu sạch" trước khi train.

### 4.2 Tiền xử lý tín hiệu

- Bỏ 2 kênh ECG, giữ 16 kênh EEG.
- Mỗi kênh: detrend → **bandpass 1–45 Hz** (filtfilt, zero-phase).
- **Re-reference common average (CAR):** bật (giảm nhiễu chung).
- Cắt epoch 3 giây cho mỗi trial Arabica/Robusta đã qua kiểm tra chất lượng.
- Lưu `epochs.npz`: `X` (n_epoch × 16 × 300), `y` (nhãn), `subject_id`.

### 4.3 Trích đặc trưng (band power)

Mỗi epoch → vector đặc trưng:
- PSD bằng Welch; công suất 5 dải mỗi kênh: **delta (1–4), theta (4–8),
  alpha (8–13), beta (13–30), gamma (30–45) Hz** → 16 × 5 = 80 đặc trưng.
- Thêm **relative band power** (chuẩn hóa theo tổng công suất) → ~160 đặc trưng.
- (Mở rộng sau, chưa vào baseline: Hjorth, spectral entropy.)

### 4.4 Mô hình & đánh giá

- Pipeline: `StandardScaler` (fit chỉ trên train fold) → classifier.
- So sánh **Logistic Regression, SVM-RBF, Random Forest** (XGBoost tùy chọn).
- **LOSO CV: 18 fold** — train 17 người, test người còn lại. Mọi bước fit
  (scaler, model) chỉ học trên train fold.
- Chỉ số: **accuracy, macro-F1, ROC-AUC, confusion matrix, accuracy từng người**.
  Mốc ngẫu nhiên = 50%.

## 5. Cấu trúc code & sản phẩm

Python, module hóa:
- `config.py` — mã code, dải tần, ngưỡng, đường dẫn.
- `01_data_quality.py` — báo cáo chất lượng + danh sách epoch sạch.
- `02_preprocess_epochs.py` — lọc, CAR, cắt epoch → `epochs.npz`.
- `03_features.py` — ma trận đặc trưng.
- `04_train_loso.py` — CV LOSO, xuất bảng kết quả + hình.
- `requirements.txt`.

**Thư viện:** `numpy, scipy, pandas, scikit-learn, matplotlib` (`mne` tùy chọn).

## 6. Ngoài phạm vi (YAGNI)

- Deep learning (EEGNet/CNN) — để pha sau.
- Phân loại Control hay multi-class 3 lớp.
- Cửa sổ con < 3 s.
- Loại artifact bằng ICA/EOG nâng cao.
