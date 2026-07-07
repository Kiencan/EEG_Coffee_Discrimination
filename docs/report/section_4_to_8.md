# Báo cáo cuối kỳ — Phân loại mùi cà phê Arabica vs Robusta từ EEG

Các phần 4–8.

---

## 4. Tiền xử lý và kiểm tra chất lượng dữ liệu

Trước khi chạy bất kỳ mô hình phân loại nào, toàn bộ dữ liệu thô được đưa qua một quy trình tiền xử lý nhiều bước với mục tiêu kép: chuyển tín hiệu EEG liên tục thành các epoch ngửi có nhãn để học máy có thể tiêu thụ trực tiếp, và phát hiện sớm những bất thường về kỹ thuật thu nhận có thể bóp méo kết quả ở pha đánh giá. Quy trình được tổ chức trong các module riêng (`src/eeg_io.py`, `src/preprocess.py`, `src/quality.py`) và được kiểm soát bằng test đơn vị nhằm đảm bảo tính tái lập.

### 4.1. Định dạng dữ liệu và cắt epoch theo marker

Mỗi người tham gia được lưu trong một tệp CSV duy nhất theo định dạng `P0XX_KT88_with_times.csv`, chứa hai mốc thời gian (`times`, `timestamp`), 16 kênh EEG thuộc hệ 10–20 (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6), hai kênh ECG (ECG1, ECG2) và cột `code` đánh dấu kích thích. Hai kênh ECG bị loại ngay ở bước đọc dữ liệu vì không phục vụ mục tiêu phân loại mùi và có biên độ khác hẳn EEG, dễ gây nhiễu cho bước common average reference ở sau. Tốc độ lấy mẫu của máy KT88 được kiểm chứng bằng cách lấy hiệu các mốc `timestamp`: trung vị ~10 ms tương đương 100 Hz, độ lệch cực đại nhỏ hơn 0.5% trên toàn bộ 18 người, nên có thể coi tần số lấy mẫu là 100 Hz và một epoch ngửi dài 3 giây tương ứng với đúng 300 mẫu.

Cột `code` mã hóa thông tin kích thích theo một quy ước riêng cho từng loại mùi: năm mã số nguyên (981, 633, 902, 598, 733) tương ứng với Arabica (lớp C1), năm mã (585, 597, 200, 558, 692) tương ứng với Robusta (C2), năm mã (712, 238, 759, 869, 562) tương ứng với mẫu control là không khí (C0), giá trị 0 đại diện cho khoảng nghỉ/ISI, và giá trị 1 đại diện cho 2 giây baseline ngay trước mỗi trial. Một trial được biểu diễn bằng một đoạn liên tục các mẫu cùng mã, nên việc cắt epoch được thực hiện theo thuật toán run-length: ta lấy hiệu cấp một của cột `code`, đánh dấu các vị trí giá trị thay đổi và xác định các đoạn `(code, start, length)`. Một đoạn được chấp nhận làm epoch khi mã của nó thuộc tập Arabica hoặc Robusta và độ dài tối thiểu là 300 mẫu; trong trường hợp đó, ta cắt đúng 300 mẫu đầu tiên của đoạn để giữ độ dài đồng nhất cho mọi epoch. Các đoạn có mã control, baseline hay nghỉ đều bị bỏ qua ở bài toán Arabica–Robusta. Đầu ra của bước này là một tensor `(n_epochs, 16, 300)` cùng nhãn `"Arabica"`/`"Robusta"` và mã kích thích gốc cho mỗi epoch — mã gốc sẽ dùng lại ở phần đánh giá nhãn theo thứ tự P019.

![Hình 4.1](../../outputs/report/fig_4_1_epoch_cutting.png)

*Hình 4.1. Minh họa quá trình cắt epoch theo marker trên P001 (kênh C3), 60 giây tín hiệu với các đoạn Arabica/Robusta/Control được tô màu.*

### 4.2. Lọc băng thông, common average reference, loại artifact

Khâu lọc tần số được áp dụng trên **toàn bộ session** trước khi cắt epoch, không phải áp riêng cho từng đoạn 3 giây. Cách làm này nhằm tránh artifact mép (edge transient) phát sinh khi `filtfilt` chạy trên đoạn ngắn: nếu lọc sau khi cắt, mỗi epoch sẽ mang theo một dao động giả ở hai biên có khả năng bị mô hình "học" như một đặc trưng phân biệt lớp. Bộ lọc được dùng là Butterworth bậc 4, dải thông 1–45 Hz, chạy zero-phase bằng `scipy.signal.filtfilt`. Cận dưới 1 Hz đủ để loại trôi điện thế chậm và drift DC, cận trên 45 Hz nằm dưới Nyquist 50 Hz đồng thời loại được nhiễu lưới điện 50 Hz Việt Nam và phần lớn nhiễu EMG tần số cao.

Sau khi lọc và cắt epoch, mỗi epoch được tham chiếu lại bằng **common average reference (CAR)**: trừ giá trị trung bình của 16 kênh tại mỗi thời điểm. CAR là lựa chọn hợp lý cho thiết kế 16 kênh phân bố đều trên da đầu vì giúp loại nhiễu chung (nhiễu nguồn, chuyển động đồng pha) mà không phụ thuộc vào một điện cực tham chiếu cụ thể nào.

Bước cuối cùng là loại artifact ở mức epoch. Một epoch bị loại nếu thỏa mãn ít nhất một trong ba điều kiện sau: chứa giá trị không hữu hạn (NaN hoặc inf, thường do hàng rỗng cuối file), có kênh "chết" với độ lệch chuẩn dưới 0.5 µV (kênh phẳng do mất tiếp xúc), hoặc có biên độ đỉnh-đỉnh trên bất kỳ kênh nào vượt ngưỡng được tính bằng phân vị 99 của phân phối peak-to-peak trên toàn tập epoch. Việc dùng ngưỡng theo phân vị thay vì giá trị tuyệt đối cố định giúp quy tắc tự thích nghi với mức nhiễu chung của tập dữ liệu sạch; trên bộ dữ liệu 8 người sạch, ngưỡng này loại đúng 1 epoch của P020 và giữ lại toàn bộ 239 epoch còn lại — tỉ lệ loại bỏ thấp cho thấy artifact rejection không phải là yếu tố dẫn dắt kết quả phân loại.

![Hình 4.2](../../outputs/report/fig_4_2_preprocess_stages.png)

*Hình 4.2. Một epoch Arabica của P001 qua ba giai đoạn tiền xử lý: raw → bandpass 1–45 Hz → bandpass + CAR.*

### 4.3. Các phát hiện về chất lượng dữ liệu

Ba phát hiện sau đây xuất hiện trong quá trình chạy `scripts/01_data_quality.py` và quyết định bộ dữ liệu cuối cùng được dùng cho mô hình hóa.

#### 4.3.1. Clipping phần cứng ở P003–P013 (over-amplification)

Đếm số mẫu chạm trần ±204 µV (sát rail full-scale ±204.8 µV của ADC 12-bit) cho ra một sự phân tách hai mode rất rõ. Nhóm P001 và P014–P020 có từ ~2,500 đến ~25,000 mẫu bão hòa, tương đương dưới 0.4% tổng số mẫu của mỗi người. Trong khi đó, **11 người liên tiếp từ P003 đến P013** ghi nhận từ ~140,000 đến hơn 1,1 triệu mẫu bão hòa — chiếm 2.3% đến 17% toàn bộ session. Hình dạng tín hiệu của nhóm này cho thấy đỉnh dương và đỉnh âm bị xén phẳng tại đúng giá trị rail, đặc trưng kinh điển của hiện tượng **clipping do over-amplification**: hệ số khuếch đại cài quá cao khiến biên độ thực vượt dải biểu diễn của ADC. Đây không phải nhiễu sinh học rời rạc mà là méo tín hiệu có hệ thống, ảnh hưởng tới mọi kênh và mọi trial của những người này.

Vấn đề quan trọng cho thiết kế đánh giá là **clipping bị nhiễu lẫn (confound) với danh tính subject**: tất cả 11 người clipped đến liên tiếp trong khoảng P003–P013, không có người clipped nào nằm xen kẽ với nhóm sạch. Trong sơ đồ Leave-One-Subject-Out, nếu giữ nguyên các người này, mô hình có thể học "fingerprint" méo phi tuyến của ADC clipping thay vì học đặc trưng thần kinh của phản ứng ngửi, và phép kiểm chéo sẽ báo cáo độ chính xác phóng đại một cách không trung thực. Lý do bảo toàn tính honest của LOSO, 11 người này được liệt vào danh sách `EXCLUDE_SUBJECTS` trong `src/config.py` và bị loại trước cả khi cắt epoch.

![Hình 4.3](../../outputs/report/fig_4_3_clipping_evidence.png)

*Hình 4.3. Bằng chứng clipping phần cứng. (a) Phân phối biên độ kênh F3: P001 sạch dạng chuông, P006 hai đỉnh nhọn tại ±204.8 µV. (b) 5 giây tín hiệu thô của P006 cho thấy đỉnh bị xén phẳng. (c) Tỉ lệ % mẫu bão hòa theo subject (log scale); đỏ là nhóm clipping bị loại.*

#### 4.3.2. Hàng rỗng cuối file ở P014

P014 ban đầu khiến pipeline bị NaN propagation trong khâu `filtfilt`. Truy nguyên thì thấy tệp CSV của người này có **hai hàng cuối toàn NaN** trên tất cả các cột EEG — nhiều khả năng do quá trình xuất từ phần mềm KT88 chèn thêm dòng trống. Vì `filtfilt` lan truyền NaN ra toàn bộ tín hiệu, nếu không xử lý thì cả 31 epoch của P014 đều bị mất sau bước loại artifact. Cách khắc phục là chỉnh `load_subject` để phát hiện và bỏ những hàng có **mọi** kênh EEG đồng thời là NaN ngay sau khi đọc CSV, trước khi đi vào bước lọc. Sau khi sửa, P014 đóng góp đủ 16 epoch Arabica và 15 epoch Robusta (1 trial Arabica dư hợp lệ so với 15 trial chuẩn của các người khác). Đây là lý do P014 vẫn nằm trong nhóm sạch dù từng "trông như" có lỗi nghiêm trọng.

#### 4.3.3. Lệch thứ tự marker ở P019 và đối chiếu với file trật tự gốc

Khi mở rộng pipeline sang bài toán hồi quy điểm đánh giá cảm quan (sensory rating), chúng tôi nhận thấy P019 cho ra kết quả khác hẳn các người còn lại — và truy nguyên thì thấy **thứ tự các mã kích thích trong cột `code` của P019 không khớp với trật tự chuẩn ghi trong `protocol/experiment_sequences.xlsx`**. Với 17 người còn lại, mã đầu tiên trong dòng marker EEG khớp đúng với mã đầu tiên trong cột thứ tự thử nghiệm canonical; với P019, hai chuỗi này bị xáo trộn (ví dụ canonical bắt đầu bằng 712-902-692 còn dòng marker EEG thì khác hẳn). Nguyên nhân nhiều khả năng đến từ thao tác ấn marker thủ công trong lúc thu, không phải lỗi tín hiệu sinh học.

Cách xử lý là **coi `experiment_sequences.xlsx` là nguồn dữ liệu chính**: với mọi người tham gia, ta dùng epoch thứ k theo thứ tự thời gian và gán nhãn bằng mã canonical thứ k của người đó, thay vì tin tuyệt đối vào cột `code`. Cách đặt nhãn này có hai lợi ích. Một là cứu lại được P019 cho phần phân tích cảm quan — vì điểm đánh giá cảm quan được ghi theo trật tự canonical, không phải theo marker EEG. Hai là vô hại với 17 người còn lại, bởi với họ trật tự canonical và marker EEG là một. Đối với bài toán Arabica–Robusta báo cáo ở đây, P019 không bị loại: cả hai cách đặt nhãn đều dẫn đến cùng tập epoch coffee với cùng phân bố nhãn vì các mã Arabica/Robusta xuất hiện đúng số lần, chỉ thứ tự đảo nhau.

![Hình 4.4](../../outputs/report/fig_4_4_marker_order.png)

*Hình 4.4. Đối chiếu trật tự kích thích canonical (đọc từ `experiment_sequences.xlsx`) và trật tự suy từ cột `code` của tệp EEG, cho P001 (khớp) và P019 (lệch).*

### 4.4. Bộ dữ liệu sạch dùng cho phân tích

Tổng kết các lựa chọn ở trên, bộ dữ liệu chính thức đi vào mô hình phân loại Arabica vs Robusta bao gồm **8 người tham gia** (P001 và P014–P020) sau khi loại 11 người clipping (P003–P013) khỏi 18 người ban đầu. Sau khi cắt epoch 3 giây theo run-length, lọc băng thông 1–45 Hz toàn session, áp common average reference, và loại epoch artifact bằng ngưỡng peak-to-peak theo phân vị 99, ta thu được **239 epoch cà phê**, gần như cân bằng giữa hai lớp (~120 Arabica và ~119 Robusta). Số epoch bị loại ở bước artifact rejection chỉ là 1 trên 240, xác nhận rằng việc loại trừ 11 người clipping đã thực sự lọc sạch nguồn nhiễu hệ thống lớn nhất — phần còn lại của tập dữ liệu sạch tới mức không cần dựa nhiều vào quy tắc loại epoch cấp dưới.

Bộ dữ liệu này được lưu thành `outputs/epochs.npz` với ba mảng đồng độ dài: `X` kích thước `(239, 16, 300)` chứa tín hiệu đã tiền xử lý, `y` chứa nhãn `"Arabica"`/`"Robusta"`, và `subjects` chứa định danh người tham gia tương ứng. Mảng `subjects` chính là cột "group" cho Leave-One-Subject-Out cross-validation ở các phần tiếp theo, đảm bảo mô hình không bao giờ thấy dữ liệu của cùng một người ở cả pha huấn luyện và pha kiểm thử.

---

## 5. Phương pháp phân tích

Sau khi có được tập epoch sạch, bước kế tiếp là biến mỗi đoạn EEG 3 giây thành một vector đặc trưng số học, đưa qua một bộ phân loại thống kê, và đánh giá khả năng tổng quát hóa giữa các người tham gia. Cả ba khâu này được thiết kế theo nguyên tắc subject-independent: mọi quá trình học (chuẩn hóa, chọn đặc trưng, fit mô hình) chỉ được thực hiện trên các fold huấn luyện, để con số báo cáo cuối cùng phản ánh đúng năng lực mô hình trên người chưa từng thấy.

### 5.1. Trích đặc trưng (band power và đặc trưng mở rộng)

Bộ đặc trưng chính được dùng cho mọi thí nghiệm là **band power** dựa trên ước lượng mật độ phổ công suất Welch. Với mỗi epoch `(16 kênh × 300 mẫu)`, ta tính PSD theo phương pháp Welch với `nperseg = min(256, n_samples)`, sau đó tích phân năng lượng trên năm dải tần kinh điển — delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz — cho từng kênh. Với mỗi dải và mỗi kênh, ta tính hai con số: công suất tuyệt đối (đơn vị µV²) và công suất tương đối (tỉ lệ trên tổng công suất 1–45 Hz của kênh đó). Kết quả là một vector đặc trưng dài `2 × 5 × 16 = 160` chiều cho mỗi epoch. Lý do giữ cả hai dạng tuyệt đối và tương đối là chúng nắm hai khía cạnh khác nhau của tín hiệu: thành phần tuyệt đối chứa thông tin về biên độ tổng thể (có thể bị ảnh hưởng bởi trở kháng điện cực và độ dày tóc/da đầu của từng người), còn thành phần tương đối tự chuẩn hóa và bền hơn với khác biệt cá nhân.

Để kiểm tra liệu các họ đặc trưng phong phú hơn có cứu được kết quả hay không, chúng tôi xây thêm năm họ đặc trưng mở rộng cài trong `src/engineered_features.py`. Họ **time** (128 chiều) tóm tắt thống kê biên độ theo kênh: trung bình, độ lệch chuẩn, độ lệch (skewness), độ nhọn (kurtosis), giá trị peak-to-peak, RMS, zero-crossing rate, và line-length. Họ **hjorth** (48 chiều) bổ sung ba tham số Hjorth — activity, mobility, complexity — cho từng kênh, vốn được dùng phổ biến trong BCI để mô tả độ phức tạp của tín hiệu trong miền thời gian. Họ **narrow_bandpower** (256 chiều) chia thêm dải tần ra các sub-band hẹp 2 Hz để bắt các đỉnh tần số tinh tế hơn. Họ **temporal_bandpower** (480 chiều) tính band power trên ba cửa sổ con (1 s đầu, 1 s giữa, 1 s cuối) để bắt động học của phản ứng trong epoch — nếu có hiệu ứng tăng/giảm theo thời gian, đặc trưng tĩnh 3 giây sẽ không thấy. Cuối cùng, họ **engineered** (912 chiều) là tổ hợp của tất cả các họ trên — một bộ đặc trưng "kitchen sink" cho phép pipeline tự chọn ra tập con tốt nhất qua `SelectKBest`.

*Bảng 5.1. Sáu họ đặc trưng được so sánh trong feature search.*

| Họ đặc trưng | Số chiều | Mô tả ngắn |
|---|---|---|
| `bandpower` | 160 | Welch PSD, 5 dải × 16 kênh × {abs, rel} |
| `time` | 128 | mean, std, skew, kurt, ptp, RMS, zero-cross, line-length × 16 kênh |
| `hjorth` | 48 | Hjorth activity / mobility / complexity × 16 kênh |
| `narrow_bandpower` | 256 | PSD chia sub-band 2 Hz × 16 kênh |
| `temporal_bandpower` | 480 | Band power trên 3 cửa sổ con (1 s đầu/giữa/cuối) × 5 dải × 16 kênh × {abs, rel} |
| `engineered` | 912 | Tổ hợp tất cả các họ trên |

### 5.2. Mô hình phân loại

Ba bộ phân loại được dùng song song: **Logistic Regression** (mô hình tuyến tính, đối chứng cơ sở), **Support Vector Machine** với kernel RBF (mô hình phi tuyến với khả năng tách margin lớn), và **Random Forest** với 300 cây (ensemble phi tuyến, mạnh trong điều kiện đặc trưng đa dạng và có nhiễu). Mọi mô hình đều được đóng gói trong một `sklearn.Pipeline` cùng với `StandardScaler` ở bước đầu để chuẩn hóa đặc trưng về trung bình 0, phương sai 1 — bước này cần thiết với LogReg và SVM, và trung tính với RF. Khi đánh giá các họ đặc trưng mở rộng, ta thêm một bước `SelectKBest` (mutual information) đặt trước classifier để chọn top-k đặc trưng có thông tin tương hỗ cao nhất với nhãn; điều quan trọng là cả scaler lẫn selector đều được fit lại từ đầu trong mỗi fold LOSO, nên không có rò rỉ thông tin từ test fold. Các siêu tham số được giữ ở mặc định sklearn để tránh tuning trên test set vốn dễ tạo ảo giác về mức năng lực thật của mô hình.

### 5.3. Đánh giá Leave-One-Subject-Out và các chỉ số

Sơ đồ đánh giá chính là **Leave-One-Subject-Out cross-validation (LOSO)**, hiện thực qua `LeaveOneGroupOut` của sklearn với `groups = subjects`. Với 8 người sạch, ta có 8 fold: mỗi fold huấn luyện trên 7 người và kiểm thử trên người còn lại, không có epoch nào của test subject xuất hiện trong tập huấn luyện. Lựa chọn này quan trọng vì đơn vị tổng quát hóa thực sự mà bài toán đặt ra là *người mới*, không phải *trial mới của cùng một người*: chia ngẫu nhiên theo trial sẽ tạo data leakage qua "fingerprint" sinh học của mỗi cá nhân (vị trí điện cực, độ dày sọ, baseline EEG cá nhân) và làm phồng accuracy một cách giả tạo.

Bốn chỉ số được báo cáo song song. **Accuracy** chỉ tỉ lệ phán đoán đúng — dễ đọc nhưng có thể đánh lừa khi mất cân bằng lớp. **Balanced accuracy** lấy trung bình recall của hai lớp, do đó vẫn trung thực ngay cả khi lớp Arabica và Robusta hơi lệch nhau; trên tập 119–120 epoch mỗi lớp của bài toán Arabica–Robusta, accuracy và balanced accuracy rất gần nhau, còn ở bài toán Control vs Coffee với tỉ lệ ~1:2 thì balanced accuracy mới là con số đáng tin. **Macro F1** trung bình hài hòa precision và recall trên từng lớp rồi mean — phản ánh năng lực mô hình trên cả hai phía của quyết định. **ROC-AUC** đo khả năng xếp hạng xác suất, được tính từ `predict_proba` cho phía lớp dương (Arabica); AUC = 0.5 nghĩa là mô hình không phân biệt được hai lớp ngay cả khi cho phép chọn ngưỡng quyết định tùy ý — một bằng chứng mạnh hơn accuracy = 50%, vì AUC còn tóm tắt toàn bộ đường cong ROC.

*Bảng 5.2. Các chỉ số đánh giá phân loại được dùng trong báo cáo.*

| Chỉ số | Định nghĩa | Mức ngẫu nhiên (2 lớp) | Ghi chú |
|---|---|---|---|
| Accuracy | (TP+TN)/N | 0.5 | Dễ đọc; lệch khi lớp mất cân bằng |
| Balanced accuracy | mean(recall theo lớp) | 0.5 | Bền với mất cân bằng lớp |
| Macro F1 | mean(F1 theo lớp) | ≈0.5 | Cân bằng precision và recall |
| ROC-AUC | diện tích dưới ROC | 0.5 | Đo khả năng xếp hạng, độc lập với ngưỡng |

---

## 6. Kết quả

Tất cả con số ở phần này được lấy từ các tệp CSV trong `outputs/` do pipeline tự sinh, không qua làm tròn thủ công. Vì câu chuyện trung tâm của báo cáo là *trung thực với dữ liệu*, các kết quả gần mức ngẫu nhiên được giữ nguyên thay vì tô vẽ.

### 6.1. Phân loại Arabica vs Robusta

Bài toán chính cho thấy ba mô hình cổ điển không vượt qua mức tình cờ một cách có ý nghĩa. Với bộ đặc trưng band power tiêu chuẩn 160 chiều, Random Forest đạt accuracy = 0.546, SVM-RBF đạt 0.513, và Logistic Regression đạt 0.466 trên LOSO 8 người, với ROC-AUC tương ứng 0.553, 0.466, và 0.496. Random Forest là mô hình "tốt nhất" nhưng cách trần lý tưởng 1.0 rất xa và chỉ cách mức ngẫu nhiên 0.5 khoảng 5 điểm phần trăm — biên độ này nằm trong sai số kì vọng của một bộ dữ liệu 239 epoch, 8 fold.

![Hình 6.1](../../outputs/report/fig_6_1_arabica_robusta_bars.png)

*Hình 6.1. So sánh accuracy LOSO của ba mô hình trên band power 160 chiều; error bar = độ lệch chuẩn giữa 8 fold, đường ngang là mức ngẫu nhiên 0.5.*

Khi mở rộng thí nghiệm bằng feature search trên 6 họ đặc trưng (bandpower, time, hjorth, narrow_bandpower, temporal_bandpower, và engineered) kết hợp với 4 mức `SelectKBest` (k = 20, 40, 80 hoặc all) và 3 mô hình, ta thu được 72 cấu hình; cấu hình tốt nhất là họ engineered 912 chiều + Random Forest với balanced accuracy ≈ 0.588, kế đó là band power 160 chiều + Linear SVM đạt 0.585. Toàn bộ 72 cấu hình dao động trong khoảng 0.43–0.59 balanced accuracy, không có "đường biên" tách rõ một họ thắng thế: cả ba mô hình trên họ engineered đều quanh 0.58, còn các họ time và temporal_bandpower theo sau ở 0.55–0.57. Mức cao nhất 0.588 vẫn cách 0.5 chưa đầy 9 điểm phần trăm, nằm trong dải dao động kì vọng của 8 fold LOSO trên 239 epoch và không cấu thành bằng chứng vững chắc cho một mô hình tổng quát hóa giữa người. Nói cách khác, không phải vấn đề chọn sai đặc trưng — tín hiệu phân biệt Arabica/Robusta giữa các người đơn giản là không hiện diện đủ mạnh trong các họ tay làm này để vượt qua phương sai giữa cá nhân.

![Hình 6.4](../../outputs/report/fig_6_4_feature_search_heatmap.png)

*Hình 6.4. Heatmap feature search: balanced accuracy theo họ đặc trưng (trục y) × mô hình (trục x); ô là cấu hình tốt nhất theo `k`. Đường ngang trắng ở 0.5 trên colorbar đánh dấu mức ngẫu nhiên.*

Chiến lược thay thế là làm việc trực tiếp với tín hiệu thô (`scripts/05_train_raw_loso.py`): áp PCA trên dạng phẳng `(n_epochs, 16×300=4800)` rồi đưa vào LogReg/SVM/RF. Cách này cho balanced accuracy 0.489 (logreg), 0.483 (linear SVM), 0.485 (Random Forest) — cũng ở mức ngẫu nhiên. Việc PCA trên tín hiệu thô không cải thiện so với band power tay làm cho thấy: chiều thông tin mà mô hình tuyến tính cần phân biệt hai lớp đơn giản không nổi lên trong cả miền tần số (band power) lẫn miền thời gian (raw + PCA).

Để hiểu sâu hơn nguồn gốc của con số 0.546 chung, ta nhìn vào accuracy LOSO theo từng người: phương sai giữa các fold rất lớn, có người mô hình đoán đúng 70%, có người chỉ 30%. Nói cách khác, accuracy "trung bình" không phản ánh một mô hình ổn định mà là trung bình của các kết quả may rủi giữa các cá nhân.

![Hình 6.2](../../outputs/report/fig_6_2_per_subject_rf.png)

*Hình 6.2. Accuracy của Random Forest trên từng người trong LOSO. Xanh: ≥ 0.5; đỏ: < 0.5. Đường nét đứt = mức ngẫu nhiên; đường chấm = trung bình toàn bộ.*

Confusion matrix của Random Forest gộp 8 fold cho thấy mô hình không có xu hướng thiên lệch về một lớp — sai số phân bố đều giữa hai phía, lần nữa khẳng định mô hình không nắm được tín hiệu phân biệt.

![Hình 6.3](../../outputs/report/fig_6_3_confusion_rf.png)

*Hình 6.3. Confusion matrix của Random Forest, gộp 8 fold LOSO; ô ghi số đếm và recall theo hàng.*

### 6.2. Phân loại Control (không khí) vs Coffee

Để kiểm tra xem pipeline có khả năng tách *bất kỳ* hai trạng thái khứu giác nào hay không, chúng tôi xây dựng bài toán đối chứng dễ hơn về mặt lý thuyết: **Control (không khí)** vs **Coffee** (gộp Arabica + Robusta). Trong thiết kế ngửi, control là ngửi không khí sạch không có mùi nào, nên khoảng cách phân biệt giữa "không có mùi" và "có mùi" được kỳ vọng lớn hơn khoảng cách giữa hai loại cà phê — nếu pipeline không tách được kể cả bài toán này thì không thể tách được bài toán Arabica–Robusta. Kết quả LOSO 8 người sạch với raw + PCA (4800 → ~50 thành phần chính) cho balanced accuracy = 0.489 (logreg), 0.483 (linear SVM), 0.485 (Random Forest) và ROC-AUC tương ứng 0.489, 0.492, 0.493. Cả ba mô hình đều ở mức ngẫu nhiên, dù tỉ lệ lớp đã được tính đến qua balanced accuracy. Đây là kết quả mang tính chẩn đoán quan trọng: trong điều kiện 100 Hz, 16 kênh, 3 giây, không có lý do nào để mong đợi rằng phân biệt khó hơn (Arabica vs Robusta) sẽ làm được khi phân biệt dễ hơn (Coffee vs no-odor) đã không tách được.

### 6.3. Thử nghiệm pseudo-subject

Một cách diễn giải khả dĩ của kết quả mức ngẫu nhiên là LOSO 8 fold *quá khắc nghiệt* với mẫu nhỏ — mỗi người chỉ đóng góp ~30 epoch nhưng lại chiếm một fold đầy đủ, nên phương sai giữa các fold rất lớn. Để kiểm tra giả thuyết này, chúng tôi xây dựng cấu trúc **pseudo-subject**: chia mỗi người sạch thành các block 5 epoch theo lớp (Control/Coffee), gắn mã định danh `P0XX_Coffee_B01`, `P0XX_Coffee_B02`, ... biến 8 người thành 72 pseudo-subject. Sau đó chạy LOSO trên 72 nhóm này thay vì 8 nhóm gốc. Kỳ vọng: nếu mức ngẫu nhiên là do mẫu quá nhỏ ở mỗi test fold, chia mịn hơn sẽ làm metric trông cao hơn nhờ giảm phương sai; nếu mức ngẫu nhiên là do tín hiệu thật sự không có, kết quả vẫn ở mức ngẫu nhiên.

Kết quả khẳng định khả năng thứ hai. Trên thiết kế pseudo-subject (72 nhóm), balanced accuracy tốt nhất là 0.491 với raw + PCA + LogReg, và các họ đặc trưng còn lại đều dưới 0.46. Đáng chú ý, kết quả pseudo-block thấp hơn kết quả real-subject (8 nhóm): trên cùng họ engineered + LogReg, real-subject đạt 0.525 trong khi pseudo-block chỉ đạt 0.451. Điều này hợp lý — chia mịn không tạo ra tín hiệu mới mà chỉ thay đổi cấu trúc đánh giá, và ở chiều ngược lại còn loại bỏ một số quy luật theo subject mà mô hình bám vào ngẫu nhiên ở thiết kế thô. Thí nghiệm pseudo-subject vì thế trở thành một dạng *negative control*: nó loại trừ giả thuyết "tín hiệu có tồn tại nhưng bị che bởi phương sai fold".

![Hình 6.5](../../outputs/report/fig_6_5_real_vs_pseudo.png)

*Hình 6.5. So sánh balanced accuracy giữa thiết kế real-subject (8 nhóm) và pseudo-block (72 nhóm) trên bài toán Control vs Coffee. Pseudo-block không cao hơn — bác bỏ giả thuyết "phương sai fold gây mức ngẫu nhiên".*

### 6.4. Liên kết giữa EEG và đánh giá cảm quan

Ngoài phân loại nhãn vật lý (loại cà phê), chúng tôi còn kiểm tra liên kết giữa EEG và **đánh giá cảm quan chủ quan**: ba điểm số `valence` (mức dễ chịu), `intensity` (mức nồng độ cảm nhận), và `favourite` (mức ưa thích) mà mỗi người đánh giá sau mỗi trial. Đây là loại biến mục tiêu khác hẳn — không cần tổng quát hóa giữa người, có thể dùng baseline cá nhân, và là biến liên tục thay vì rời rạc.

Bước đầu, chúng tôi tính **tương quan within-subject** giữa mỗi đặc trưng band power và mỗi điểm số: trên mỗi người, tính hệ số tương quan Pearson của một đặc trưng với một điểm số, sau đó trung bình hệ số trên 8 người. Trong số 480 cặp (160 đặc trưng × 3 điểm số), hệ số tương quan trung bình nằm trong khoảng `[-0.20, +0.22]`, với cực đại tuyệt đối ≈ 0.22 ở các kênh trung tâm/thái dương cho theta band liên hệ với favourite (ví dụ `rel_theta_C3` với favourite r ≈ 0.219, `abs_theta_C3` với favourite r ≈ 0.186). Đây là tương quan yếu — chưa đủ vượt qua hiệu chỉnh đa kiểm định Bonferroni — nhưng *hướng* các tương quan thì nhất quán: nhóm theta vùng C3/C4/T3 dương cho `favourite`, trong khi nhóm delta vùng trán dương âm cho cả `favourite` lẫn `valence`, gợi ý một dạng phản ứng theo dải tần và vị trí kênh có cơ sở sinh lý nhưng yếu.

![Hình 6.6](../../outputs/report/fig_6_6_sensory_correlation_heatmap.png)

*Hình 6.6. Heatmap tương quan within-subject trung bình giữa 160 đặc trưng band power (trục y, gộp theo abs/rel × dải tần) và 3 điểm số cảm quan (trục x). Colormap diverging quanh 0; dải giá trị ~[-0.22, +0.22].*

Bước hai, chúng tôi thử **hồi quy Ridge** lên ba điểm số. Đây là chỗ thiết kế đánh giá quan trọng: trên cùng tập dự đoán, nếu tính R² gộp tất cả epoch của tất cả người vào một mảng (cross-subject), ta được R² âm rất sâu (valence -0.381, intensity -0.464, favourite -0.116) — chứng tỏ mô hình không bù được giữa các người. Khi tính R² *within-subject* (tính R² trên epoch của từng người rồi trung bình), kết quả vẫn âm nhưng nhẹ hơn: valence -0.197, intensity -0.075, favourite -0.228. R² âm đồng nghĩa với "tệ hơn dự đoán bằng trung bình"; mô hình hồi quy không cung cấp tín hiệu hữu ích trên tập 237 epoch ở mức trial-by-trial.

Bước cuối, chúng tôi rời rạc hóa các điểm số thành **3 mức** (thấp/trung bình/cao theo phân vị 33/66 của từng người để tôn trọng thang đánh giá cá nhân) và phân loại bằng Random Forest và LogReg. Mức ngẫu nhiên là 1/3 ≈ 0.333. Kết quả within-subject vượt mức ngẫu nhiên một cách khiêm tốn nhưng nhất quán: valence đạt balanced accuracy 0.48 (RF) / 0.45 (LogReg), intensity 0.408 / 0.44, favourite 0.399 / 0.417 — cao hơn 33% từ ~7 đến ~15 điểm phần trăm. Đây là kết quả khả quan nhất của toàn bộ báo cáo và đáng chú ý: ở cùng người, EEG mang được một phần thông tin về mức cảm nhận chủ quan của họ, dù lượng thông tin đó không đủ để tổng quát hóa giữa các người (cross-subject balanced accuracy rơi về 0.27–0.40, tức ngang hoặc dưới mức ngẫu nhiên). Câu chuyện này lặp lại mẫu hình quen thuộc trong văn liệu BCI: tín hiệu EEG khứu giác mang tính cá nhân hóa cao, dấu hiệu within-subject yếu nhưng có thật, còn dấu hiệu giữa các người gần như không tồn tại với band power thô.

---

## 7. Thảo luận

### 7.1. Diễn giải kết quả

Kết quả trung tâm của báo cáo — phân loại Arabica vs Robusta giữa các người ở mức tình cờ — nhất quán với những gì lý thuyết và văn liệu dự đoán cho thiết lập đo của chúng tôi. EEG là tín hiệu trường xa thu được trên da đầu, đã đi qua sọ và da với suy hao và làm mờ không gian rất lớn. Phản ứng khứu giác trong vỏ não tập trung ở vùng olfactory cortex (piriform cortex và các vùng kế cận), nằm sâu trong thùy thái dương trung gian — đây là vùng *khó* nhất đối với EEG da đầu vì xa các điện cực bề mặt và bị che bởi nhiều cấu trúc dẫn điện trung gian. Khác biệt giữa hai loại hạt cà phê cùng chi *Coffea* càng là khác biệt tinh tế ở mức nhận biết khứu giác, nên kỳ vọng có một "chữ ký" tần số ổn định và đo được giữa các người là rất lạc quan. Hơn nữa, band power 5 dải tần là biểu diễn rất nén — nó loại bỏ thông tin pha, định vị tinh trong miền thời gian, và quan hệ liên kênh (connectivity). Nếu tín hiệu phân biệt hai mùi tồn tại ở dạng pha-khóa với kích thích hoặc ở dạng đồng bộ liên vùng, bộ đặc trưng của chúng tôi sẽ không nhìn thấy nó. Việc kết quả pseudo-subject cũng ở mức ngẫu nhiên loại trừ giả thuyết "tín hiệu có nhưng bị che bởi phương sai fold", và kết quả Control vs Coffee cũng ở mức ngẫu nhiên loại trừ giả thuyết "nhiệm vụ phân biệt giữa hai loại cà phê quá khó nhưng phân biệt mùi/không mùi thì làm được". Hai phép phản chứng này chỉ ra rằng vấn đề nằm sâu hơn — trong sự tương thích giữa thiết bị, phương pháp đặc trưng, và bản chất tín hiệu sinh học.

Một mảng kết quả khả quan hơn xuất hiện ở phân tích cảm quan within-subject: phân loại 3 mức valence vượt mức ngẫu nhiên ~15 điểm phần trăm, và các tương quan band power–rating tuy yếu nhưng có *hướng* hợp lý sinh lý (theta trung tâm dương với mức ưa thích). Điều này nhất quán với hiểu biết phổ biến trong văn liệu EEG: dấu hiệu cá nhân *có thể* trích xuất từ band power khi không cần tổng quát hóa giữa người, và baseline cá nhân (chuẩn hóa theo người) là yếu tố quyết định.

### 7.2. Hạn chế

Có bốn hạn chế lớn cần ghi nhận khi diễn giải kết quả. Thứ nhất, **mật độ điện cực thấp** — chỉ 16 kênh hệ 10–20 phân bố thô trên toàn da đầu. Các thí nghiệm khứu giác EEG hiện đại thường dùng 32 đến 64 kênh để bù cho suy hao không gian; với 16 kênh, độ phân giải không gian không đủ để tách các vùng vỏ não kế cận, và càng không đủ để áp các kỹ thuật như source localization. Thứ hai, **tốc độ lấy mẫu 100 Hz** giới hạn dải tần phân tích ở 1–45 Hz (do Nyquist), nên các thành phần gamma cao và đặc biệt là high-gamma thường gắn liền với xử lý cảm giác sâu nằm ngoài tầm với. Đặc trưng tinh tế nhất của phản ứng khứu giác có thể nằm chính ở dải tần này. Thứ ba, **cỡ mẫu rất nhỏ** — 8 người sạch, ~30 epoch mỗi người, tổng 239 epoch — khiến mọi mô hình học máy hiện đại bị giới hạn nghiêm trọng và rất dễ overfit ngay cả với regularization. Thứ tư, và quan trọng nhất về phương pháp luận: **clipping phần cứng ở 11/18 người** đã loại đi phần lớn dữ liệu mà nếu giữ được sẽ tăng cỡ mẫu lên gấp 2.5 lần. Việc loại 11 người này là quyết định trung thực với phương pháp (clipping confound với subject identity, sẽ làm hỏng LOSO), nhưng cái giá phải trả là cỡ mẫu cuối cùng quá nhỏ để phát hiện bất kỳ hiệu ứng yếu nào ở mức có ý nghĩa thống kê. Cuối cùng, thiết kế phân tích không có **baseline cá nhân**: mỗi epoch 3 giây được dùng nguyên dạng, không trừ đi baseline 2 giây trước trial. Văn liệu ERP và phân tích power EEG thường khuyến nghị trừ baseline để loại trôi cá nhân — bước này có thể là một cải tiến đơn lẻ có tác động.

### 7.3. Bài học về chất lượng dữ liệu và quy trình thu thập

Phần lớn giá trị của dự án này không nằm ở kết quả phân loại mà nằm ở các phát hiện chất lượng dữ liệu. Ba bài học cụ thể sau đáng giữ lại cho bất kỳ kỳ thu EEG nào tiếp theo. Một là, **mọi pipeline tiền xử lý cần kiểm tra biên độ tín hiệu so với rail của ADC trước khi tin con số nào**. Việc 11 người liên tiếp bị clipping mà không phát hiện sớm là một lỗi quy trình nghiêm trọng — nếu nhìn histogram biên độ ngay sau mỗi buổi đo, vấn đề đã được phát hiện và sửa hệ số khuếch đại trước khi thu thêm người. Hai là, **trật tự kích thích cần được lưu ở nguồn dữ liệu độc lập với marker phần cứng**. Cột `code` trong file CSV phụ thuộc vào thao tác bấm marker đúng thời điểm, vốn có thể sai sót như trường hợp P019. Nếu không có `experiment_sequences.xlsx` làm nguồn canonical, ta đã không thể cứu được dữ liệu của P019 cho phần phân tích cảm quan, và có thể còn tin nhầm vào nhãn sai cho phần phân loại. Ba là, **kiểm tra dữ liệu nên là một module có thể chạy độc lập từ ngày đầu**, không phải thứ thêm vào sau. Trong dự án này, `scripts/01_data_quality.py` được viết sớm và đã phát hiện cả ba vấn đề (clipping, hàng rỗng P014, lệch marker P019) trước khi bước vào mô hình hóa — đây là cách làm đúng. Nếu các thí nghiệm khứu giác EEG tương lai bắt đầu từ một quality check tự động chạy ngay sau mỗi buổi đo, các vấn đề như clipping P003–P013 sẽ được sửa trong vòng vài ngày, không phải bị phát hiện vào pha phân tích sau khi đã thu xong toàn bộ.

---

## 8. Kết luận và hướng phát triển

Báo cáo này trình bày một pipeline machine learning hoàn chỉnh cho bài toán phân loại mùi cà phê Arabica vs Robusta từ tín hiệu EEG 16 kênh thu bằng máy KT88 ở 100 Hz. Pipeline thực hiện đầy đủ các bước chuẩn mực — cắt epoch theo marker, lọc băng thông 1–45 Hz zero-phase, common average reference, loại artifact theo phân vị peak-to-peak, trích đặc trưng band power 160 chiều, phân loại bằng ba mô hình cổ điển, và đánh giá bằng Leave-One-Subject-Out trên 8 người sạch. Kết quả chính cho thấy các mô hình band power + ML cổ điển không vượt qua mức ngẫu nhiên một cách có ý nghĩa: Random Forest đạt accuracy 0.546, LogReg 0.466, SVM 0.513 trên 239 epoch cân bằng; feature search trên 6 họ đặc trưng và 72 cấu hình nâng mức cao nhất lên balanced accuracy 0.588 (engineered + RF) — vẫn nằm trong dải dao động kì vọng của LOSO 8 fold, không cấu thành bằng chứng vững chắc về một mô hình tổng quát hóa giữa người. Mở rộng sang bài toán đối chứng dễ hơn (Control vs Coffee) cũng cho kết quả ở mức ngẫu nhiên, và thử nghiệm pseudo-subject xác nhận đây không phải artifact của thiết kế đánh giá. Khía cạnh khả quan duy nhất nằm ở phân tích cảm quan within-subject: phân loại 3 mức điểm số vượt mức ngẫu nhiên 33% khoảng 7–15 điểm phần trăm, gợi ý dấu hiệu EEG–cảm nhận yếu nhưng có thật ở cấp độ cá nhân.

Ba hướng phát triển có cơ sở rõ ràng để theo đuổi tiếp. Thứ nhất, chuyển sang **phân tích within-subject** thay vì cross-subject là hướng có khả năng đem lại kết quả thực dụng cao nhất: dùng baseline cá nhân, huấn luyện một mô hình riêng cho mỗi người trên ~25 epoch và đánh giá bằng k-fold trong người. Bài toán ứng dụng "máy đoán đánh giá cảm quan" cho từng cá nhân khả thi hơn nhiều so với "máy đọc mùi cà phê chung cho tất cả người". Thứ hai, thay đổi bộ đặc trưng theo hướng **kết nối liên kênh và miền pha**: coherence, phase-locking value, common spatial patterns (CSP), Riemannian geometry trên ma trận hiệp phương sai — những đại diện này thường mạnh hơn band power trong các paradigm BCI khi tín hiệu yếu. Thứ ba, thử **deep learning trực tiếp trên tín hiệu thô** với các kiến trúc đặc thù cho EEG như EEGNet, ShallowConvNet, hay DeepConvNet, kết hợp data augmentation và transfer learning từ các bộ EEG công khai lớn. Tuy nhiên ở quy mô dữ liệu hiện tại, deep learning nhiều khả năng vẫn overfit; bước này nên đi cùng với việc mở rộng cỡ mẫu — thu thêm dữ liệu mới với hệ số khuếch đại đã sửa, hoặc kết hợp pipeline với một bộ EEG khứu giác đã công bố. Trong mọi hướng đi, bài học cốt lõi rút ra từ dự án này vẫn là: chất lượng dữ liệu là điều kiện tiên quyết, và một quality check chạy ngay sau buổi đo có giá trị lớn hơn bất kỳ kỹ thuật mô hình hóa tinh xảo nào.
