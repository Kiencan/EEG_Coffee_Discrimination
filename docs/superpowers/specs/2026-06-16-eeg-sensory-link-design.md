# EEG ↔ Sensory-rating link — Design

**Ngày:** 2026-06-16
**Trạng thái:** Đã duyệt thiết kế

## 1. Mục tiêu

Khám phá mối liên kết giữa tín hiệu EEG (khi ngửi cà phê) và đánh giá cảm quan
chủ quan của người tham gia, trên 3 thang: **Valence, Intensity, Favourite**
(1–7). Dùng cả tương quan, hồi quy (dự đoán điểm liên tục), và phân loại 3 mức;
cả within-subject lẫn cross-subject.

## 2. Dữ liệu

- Cảm quan: `protocol/sensory_data.csv` — dạng wide, 20 người × 45 trial, mỗi
  trial có mã kích thích + 3 điểm (Valence/Intensity/Favourite, 1–7).
- EEG: epoch cà phê (Arabica+Robusta) của **8 người sạch** (P001, P014–P020),
  ~238 epoch (loại P003–P013 clipping, theo `EXCLUDE_SUBJECTS`).
- **Ghép per-trial:** thứ tự mã kích thích trong EEG (run-length theo thời gian)
  khớp 100% với thứ tự trial trong cảm quan (đã xác minh P001: 45/45). Ghép theo
  vị trí, có assert khớp mã.
- Quan sát: Intensity tách điều kiện mạnh (Control 1.9 vs cà phê ~5.1); Valence/
  Favourite gần nhau giữa điều kiện.
- Phạm vi: chỉ epoch **cà phê** (bỏ Control), đặc trưng **band-power 160**.

## 3. Thành phần

### 3.1 `src/sensory.py` (có test)
- `load_sensory_long(csv_path)` → DataFrame cột `[subject, trial, code, valence,
  intensity, favourite]`. Map `Person_N` → `P0{N:02d}` (P001..P020). Bỏ hàng NaN.
- `align_ratings(ordered_codes, subject_sensory_df)` → mảng (valence, intensity,
  favourite) khớp theo vị trí với `ordered_codes`; **raise ValueError** nếu chuỗi
  mã không khớp (an toàn, không ghép nhầm).

### 3.2 `scripts/16_build_sensory_dataset.py`
- Mỗi người sạch: `load_subject` → bandpass 1–45 Hz + CAR; lấy run-length của mọi
  mã kích thích (coffee+control) THEO THỨ TỰ để ghép vị trí với cảm quan; assert
  khớp; gắn 3 điểm cho mỗi run.
- Giữ epoch cà phê, loại artifact (`epoch_is_artifact`).
- Lưu `outputs/epochs_sensory.npz`: `X` (n×16×300), `subjects`, `codes`,
  `condition` (Arabica/Robusta), `valence`, `intensity`, `favourite`.

### 3.3 `scripts/17_sensory_correlation.py` (within-subject)
- Build band-power features (`src/features.build_feature_matrix`).
- Trong mỗi người: Spearman giữa từng đặc trưng (160) và từng thang; trung bình
  hệ số qua người (Fisher-z trung bình hoặc trung bình thường).
- Lưu `outputs/sensory_correlations.csv` (đặc trưng × thang: mean_corr, |mean|,
  nhất quán dấu) + heatmap `sensory_correlation_heatmap.png`. In top liên hệ.

### 3.4 `scripts/18_sensory_regression.py`
- Mỗi thang: pipeline `StandardScaler → RidgeCV` dự đoán điểm liên tục.
- **Within-subject:** K-fold (k=5) trong mỗi người; gộp dự đoán; báo cáo mean R²
  (qua người) và Pearson corr(pred, true).
- **Cross-subject:** LOSO (8 fold); R² và corr.
- Lưu `outputs/sensory_regression.csv` (thang × {within, cross}: R², corr, n).
- Lưu ý trung thực: 30 trial/người là nhỏ → R² có thể thấp/âm; báo cáo nguyên.

### 3.5 `scripts/19_sensory_classification.py`
- Chia mỗi thang thành **thấp/vừa/cao**: within = tertile theo từng người (khử
  baseline cá nhân); cross = tertile toàn cục.
- Phân loại RandomForest + LogReg (`class_weight='balanced'`).
- **Within-subject** (k-fold mỗi người) + **cross-subject** (LOSO).
- Báo cáo **balanced-accuracy** (mốc 3 lớp = 0.333). Lưu
  `outputs/sensory_classification.csv`.

## 4. Quy ước & metric

- Within-subject là phép kiểm chính cho điểm cảm quan (khử baseline cá nhân);
  cross-subject để đối chiếu. Mọi bước fit chỉ học trên train fold.
- Hồi quy: R² + Pearson corr(pred, true). Phân loại 3 mức: balanced-accuracy.
- Báo cáo trung thực mọi con số kể cả mức ngẫu nhiên.

## 5. Ngoài phạm vi (YAGNI)

- Đặc trưng engineered/raw (giữ band-power cho lần đầu).
- Thêm epoch Control vào Intensity (có thể làm sau).
- Deep learning; mô hình hỗn hợp (mixed-effects).
