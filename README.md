# EEG Coffee Discrimination

Phân loại **mùi cà phê Arabica vs Robusta** từ tín hiệu EEG khi ngửi (phân
loại nhị phân). Đơn vị phân loại = một epoch ngửi 3 giây.

- **Input:** một epoch EEG 3 giây (16 kênh × 300 mẫu).
- **Output:** nhãn `Arabica` hoặc `Robusta`.

## Dữ liệu

Dữ liệu thô (`data/P0XX_KT88_with_times.csv`, 18 người, ~1.1GB) không được
lưu trong repo (vượt giới hạn kích thước file của GitHub). Tải về tại:

**[Google Drive — data/](https://drive.google.com/file/d/1UA3_y2SPQtY4IkyRKGciI1wVllG4WEiG/view?usp=sharing)**

Sau khi tải, giải nén vào thư mục `data/` ở gốc dự án (cùng cấp với `src/`,
`scripts/`).

Chi tiết định dạng dữ liệu, mã kích thích, và lưu ý chất lượng (clipping,
`EXCLUDE_SUBJECTS`) xem trong [CLAUDE.md](CLAUDE.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy pipeline

Từ thư mục gốc dự án:

```bash
python -m pytest -q                       # chạy toàn bộ test
python scripts/01_data_quality.py         # báo cáo chất lượng
python scripts/02_preprocess_epochs.py    # xuất epoch sạch
python scripts/03_features.py             # build đặc trưng
python scripts/04_train_loso.py           # train + đánh giá LOSO
```

`outputs/` (npz, hình, CSV kết quả) bị gitignore — là sản phẩm tái tạo được
từ dữ liệu, không commit.

## Cấu trúc dự án

```
src/config.py     # mã code, dải tần, ngưỡng, EXCLUDE_SUBJECTS, đường dẫn
src/eeg_io.py     # load CSV, run-length, cắt epoch
src/quality.py    # kiểm tra chất lượng, loại epoch artifact
src/preprocess.py # bandpass 1-45 Hz zero-phase + CAR
src/features.py   # band power Welch (delta/theta/alpha/beta/gamma)
src/evaluate.py   # LOSO CV (LeaveOneGroupOut theo subject) + metrics

scripts/          # pipeline chạy tuần tự 01 -> 22
tests/            # test cho từng module trong src/
docs/              # spec, plan thiết kế
protocol/          # protocol thí nghiệm gốc + dữ liệu cảm quan
paper/             # bản thảo bài báo
```

## Giấy phép

[Apache License 2.0](LICENSE)
