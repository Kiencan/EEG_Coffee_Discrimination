# Pseudo-subject blocks for Control vs Coffee — Design

**Ngày:** 2026-06-16
**Trạng thái:** Đã duyệt thiết kế

## 1. Mục tiêu

Áp dụng kỹ thuật **class-wise pseudo-subject** (đã có trong `src/pseudo_subjects.py`)
lên bộ Control(C0) vs Coffee(C1∪C2). Chia mỗi người thật thành các pseudo-subject
theo block 5 mẫu liên tiếp trong mỗi lớp, rồi đánh giá LOSO theo 2 cách gom nhóm
để xem cách "nới lỏng" có lộ tín hiệu phân biệt không.

## 2. Bối cảnh & dữ liệu

- Đầu vào: `outputs/epochs_c0_vs_coffee.npz` (356 epoch, 8 người sạch,
  Coffee 238 / Control 118).
- Hàm có sẵn (đã test trong `tests/test_pseudo_subjects.py`):
  `make_classwise_pseudo_subjects(y, subjects, block_size=5)` →
  pseudo ID `{subject}_{label}_B{block}`, giữ block lẻ cuối; và
  `pseudo_subject_report(...)`.
- block_size=5: Control ~15/người → 3 block; Coffee ~30/người → 6 block →
  ~9 pseudo/người × 8 ≈ ~72 nhóm pseudo (so với 8 người thật).

## 3. Lưu ý phương pháp (ghi rõ trong báo cáo)

Nhóm `pseudo_block` có ID đặc thù theo lớp, nên khi LOSO bỏ ra 1 pseudo-subject,
dữ liệu của *cùng người thật* (block khác/lớp khác) vẫn nằm trong train →
**rò rỉ trong-người, kết quả lạc quan**, KHÔNG phải bằng chứng tổng quát hóa.
So sánh với `real_subject` (LOSO thật) để thấy khoảng cách.

## 4. Thành phần (2 script mới, tái dùng module)

### 4.1 `scripts/14_make_pseudo_c0.py`
- Load `epochs_c0_vs_coffee.npz`.
- `make_classwise_pseudo_subjects(y, subjects, block_size=5)`.
- Lưu `outputs/epochs_pseudo_c0.npz` (X, y, real_subjects, pseudo_subjects,
  pseudo_labels, block_index) + `outputs/pseudo_c0_report.csv`.
- In số pseudo-subject và phân bố block theo (người, lớp).

### 4.2 `scripts/15_evaluate_pseudo_c0.py`
- Load `epochs_pseudo_c0.npz`.
- Với MỖI cách gom nhóm trong {`pseudo_block`, `real_subject`}: chạy `run_loso`
  (`positive_label="Coffee"`) trên các biểu diễn đại diện:
  - bandpower (`build_feature_matrix`) × {random_forest, svm_rbf}.
  - engineered (`build_feature_family_matrix(family="engineered")`) ×
    {selected_random_forest, selected_logreg} với k="all".
  - raw (`flatten_epochs`) × {raw_pca_logreg}.
  - Tất cả classifier dùng `class_weight="balanced"`.
- Báo cáo **balanced_accuracy, macro_f1, roc_auc** + số nhóm.
- Lưu `outputs/c0_pseudo_evaluation.csv`, in bảng so sánh 2 cách gom nhóm.

## 5. Metric & mất cân bằng

- Mất cân bằng ~1:2 → `class_weight="balanced"`, metric chính =
  **balanced_accuracy** (mốc 0.5). `run_loso` đã hỗ trợ balanced_accuracy +
  AUC fallback qua `decision_function`.

## 6. Ngoài phạm vi (YAGNI)

- Averaged-block (`average_classwise_blocks`, scripts 09/10 cho A-vs-R) — không
  làm ở bài này.
- Deep learning, within-subject CV, multi-class.
