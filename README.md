# Regression with an Insurance Dataset

## 1. Tổng quan bài toán và cấu trúc tập dữ liệu

Dự án này giải quyết bài toán Hồi quy (Regression) dựa trên Bộ dữ liệu Bảo hiểm từ cuộc thi **[Kaggle Playground Series - Season 4 Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12)**. Mục tiêu của mô hình là dự đoán Phí bảo hiểm (`Premium Amount`) cho khách hàng dựa trên các thuộc tính của họ.

Dữ liệu được cung cấp bao gồm ba thành phần chính:

* **Tập Huấn luyện (`train.csv`):** Bao gồm 21 cột (1 định danh `id`, 19 cột đặc trưng đầu vào, và 1 biến mục tiêu liên tục là `Premium Amount`).
* **Tập Kiểm thử (`test.csv`):** Bao gồm 20 cột (cấu trúc đặc trưng giống hệt tập huấn luyện, nhưng không có biến mục tiêu).
* **Mẫu Nộp bài (`sample_submission.csv`):** Định dạng chuẩn yêu cầu để nộp bài, chỉ bao gồm đúng 2 cột: `id` và giá trị dự đoán `Premium Amount`.

## 2. Kết nối các thư viện cần thiết

Thuật toán lõi được lựa chọn cho bài toán hồi quy này là **XGBoost** (Extreme Gradient Boosting). Thuộc họ thuật toán học tập hợp (Ensemble Boosting), XGBoost xây dựng các cây quyết định một cách tuần tự, trong đó mỗi cây phía sau sẽ học hỏi và tối thiểu hóa sai số của các cây phía trước.

Luồng xử lý phụ thuộc vào các thư viện lõi sau:

* `xgboost`: Cung cấp mô hình hồi quy Gradient Boosting.
* `pandas`: Sử dụng để thao tác, phân tích dữ liệu dạng bảng và DataFrame.
* `numpy`: Thực hiện các phép tính số học và ma trận với hiệu suất cao.
* `pathlib`: Quản lý đường dẫn tập tin theo hướng đối tượng.
* `scikit-learn` (sklearn): Cung cấp các thành phần cấu trúc cho Pipeline học máy, bao gồm:
  * `train_test_split`: Phân tách tập dữ liệu.
  * `Pipeline` & `ColumnTransformer`: Liên kết và điều phối các bước tiền xử lý.
  * `SimpleImputer`: Xử lý dữ liệu khuyết thiếu.
  * `OneHotEncoder`: Biến đổi các đặc trưng phân loại.
  * `mean_squared_log_error`: Thang đo đánh giá độ lỗi (RMSLE).

## 3. Configuration

Dữ liệu được nạp vào không gian làm việc thông qua Google Drive. Bước cấu hình sẽ cô lập các đặc trưng tính toán khỏi siêu dữ liệu (metadata) và biến mục tiêu.

* **Không gian Đặc trưng (X):** Tập hợp 19 cột mang thông tin. Cột `id` và biến mục tiêu bị loại bỏ hoàn toàn khỏi không gian này để ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage).
* **Vector Mục tiêu (y):** Cột `Premium Amount` từ tập huấn luyện.
* *Lưu ý đối với tập Test:* Cột `id` trong tập kiểm thử được tách ra và lưu trữ riêng biệt, chỉ dùng để phục vụ việc ghép nối kết quả ở bước cuối cùng.

## 4. Feature Engineering

Tập dữ liệu chứa biến thời gian (`Policy Start Date`) mà các mô hình toán học không thể xử lý trực tiếp. Kỹ thuật trích xuất đặc trưng được áp dụng để phân rã biến này thành 4 đặc trưng số học độc lập:

* `Policy Start Year` (Năm)
* `Policy Start Month` (Tháng)
* `Policy Start Day` (Ngày)
* `Policy Start DayOfWeek` (Ngày trong tuần)

Sau khi trích xuất, cột văn bản ngày tháng ban đầu sẽ bị xóa bỏ. Hàm biến đổi này được áp dụng đồng nhất cho cả tập Huấn luyện và tập Kiểm thử nhằm duy trì tính nhất quán về số lượng chiều dữ liệu.

Ngoài việc phân rã biến thời gian (`Policy Start Date`) thành các cột số độc lập (Năm, Tháng, Ngày, Ngày trong tuần), hệ thống còn tạo ra các đặc trưng phức hợp, giúp mô hình bắt quy luật sâu hơn:

*   **Chỉ số Gánh nặng tài chính (`Income_per_Dependent`):** Được tính bằng tổng thu nhập chia cho số người phụ thuộc. Khách hàng có thu nhập $100.000 nhưng phải nuôi 4 người sẽ có rủi ro chi trả hoàn toàn khác một người độc thân có cùng mức thu nhập. Đặc trưng này giúp mô hình đánh giá đúng "độ dư dả tài chính" thực tế.
*   **Tỷ lệ Rủi ro phương tiện (`Vehicle_Age_Ratio`):** Lấy tuổi của xe chia cho tuổi của người lái. Phép tính này tạo ra một mỏ neo cảnh báo rủi ro cực mạnh: Một thanh niên rất trẻ (non kinh nghiệm) lại điều khiển một chiếc xe rất cũ (dễ hỏng hóc) sẽ tạo ra xác suất tai nạn cao hơn rất nhiều so với người trung niên lái xe mới.
*   **Tần suất Bồi thường (`Claims_per_Year`):** Số lần yêu cầu bồi thường chia cho số năm tham gia. Việc chỉ đếm số lần báo tai nạn thô là rất phiến diện (báo tai nạn 5 lần trong suốt 20 năm tham gia là rất an toàn, nhưng 5 lần chỉ trong 1 năm lại là thảm họa). Đặc trưng này đưa lịch sử rủi ro về cùng một hệ quy chiếu thời gian.
*   **Chỉ số Hao mòn Sức khỏe (`Health_Age_Index`):** Nhân điểm sức khỏe với độ tuổi để tạo thành một hệ số rủi ro y tế kép (Interaction Feature). Nó giúp mô hình hiểu rằng: Cùng một mức điểm sức khỏe suy giảm, nhưng nếu rơi vào một người cao tuổi thì xác suất xảy ra biến chứng viện phí sẽ tăng theo cấp số nhân so với người trẻ.
## 5. Tiền xử lý

Do dữ liệu chứa các giá trị khuyết thiếu và các kiểu dữ liệu phi số học, hệ thống sử dụng `ColumnTransformer` để định tuyến các đặc trưng qua các luồng xử lý chuyên biệt:

* **Đặc trưng Số học (Numerical Features):** Các ô dữ liệu trống được điền khuyết bằng **Trung vị (Median)** của cột tương ứng, giúp mô hình ít bị ảnh hưởng bởi các giá trị ngoại lai.
* **Đặc trưng Phân loại (Categorical Features):**
  1. Dữ liệu khuyết thiếu được điền bằng hằng số (chuỗi văn bản `"Missing"`).
  2. Các biến này sau đó trải qua quá trình **Mã hóa One-Hot (One-Hot Encoding)**. Quá trình này chuyển đổi các biến phân loại thành một ma trận thưa chứa các vector nhị phân. Ví dụ: Nếu một cột có 3 phân loại $[Giỏi, Khá, Xuất Sắc]$, nó sẽ được biến đổi thành 3 cột nhị phân độc lập. Một mẫu dữ liệu thuộc loại $Giỏi$ sẽ được biểu diễn dưới dạng vector $[1, 0, 0]$.

* **Kết quả:** Dữ liệu đầu ra của bước này là một ma trận hoàn toàn mang tính số học, không còn giá trị khuyết, sẵn sàng để đưa vào thuật toán.

## 6. Chiến lược Đánh giá chéo (Stratified K-Fold Cross Validation)

Để ngăn chặn tối đa hiện tượng học vẹt (Overfitting) và tận dụng 100% lượng dữ liệu, chiến lược đánh giá K-Fold được áp dụng thay vì chỉ chia tách 80/20 một lần duy nhất.

*   **Phân hoạch mục tiêu & Chia Fold:** Cột giá tiền được chia thành 20 khoảng (bins) để làm mỏ neo phân tầng. Dữ liệu sau đó được chia đều thành **5 nếp gấp (5 Folds)**.
*   **Logarithmic Transformation (Biến đổi Logarit):** Ở mỗi lượt huấn luyện, biến mục tiêu $y$ (giá tiền) được ép qua hàm `np.log1p` trước khi đưa vào thuật toán. Điều này giúp thu hẹp sự chênh lệch của các hợp đồng bảo hiểm giá trị cực đoan (Outliers), đồng thời đồng bộ hóa hoàn toàn hàm mục tiêu (MSE) của XGBoost với thang đo chấm điểm của cuộc thi (RMSLE).
*   **Huấn luyện xoay vòng:** Hệ thống sẽ chạy 5 vòng lặp. Ở mỗi vòng, 4 nếp gấp được dùng để huấn luyện và nếp gấp còn lại dùng để thi thử (Validation). Kết thúc 5 vòng, toàn bộ 100% dữ liệu đều đã được đánh giá chéo (Out-of-Fold), tạo ra thước đo sát thực tế nhất.

## 7. Dự đoán & Kết xuất (Ensembling)

Quy trình dự đoán không dựa vào một mô hình đơn lẻ mà sử dụng chiến lược dự đoán tổ hợp (Blending).

*   **Dự đoán tập thể:** Tại mỗi nếp gấp trong quá trình K-Fold, mô hình XGBoost hiện tại sẽ tiến hành giải đề trên tập `test.csv`. Các kết quả dự đoán (sau khi được dịch ngược logarit bằng hàm `np.expm1`) sẽ được lưu trữ lại.
*   **Lấy Trung bình cộng (Averaging):** Kết quả cuối cùng là trung bình cộng của 5 bảng dự đoán từ 5 nếp gấp. Việc này đóng vai trò như một "hội đồng giám khảo", triệt tiêu các sai số ngẫu nhiên của từng cá nhân mô hình, mang lại một kết quả nộp bài ổn định và có tính tổng quát hóa cao nhất. Kết quả được lưu dưới định dạng `sample_submission.csv`.
