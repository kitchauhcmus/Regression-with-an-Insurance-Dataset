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

Ngoài việc phân rã biến thời gian (`Policy Start Date`) thành các cột số độc lập (Năm, Tháng, Ngày, Ngày trong tuần), hệ thống còn ứng dụng tư duy phân tích nghiệp vụ bảo hiểm (Domain Knowledge) để tạo ra các đặc trưng phức hợp, giúp mô hình bắt quy luật sâu hơn:

*   **Chỉ số Gánh nặng tài chính (`Income_per_Dependent`):** Thu nhập bình quân trên mỗi người phụ thuộc, phản ánh khả năng tài chính thực tế của khách hàng.
*   **Tỷ lệ Rủi ro phương tiện (`Vehicle_Age_Ratio`):** Tương quan giữa độ tuổi xe và độ tuổi người lái, giúp nhận diện nhóm rủi ro cao (người trẻ lái xe cũ).
*   **Tần suất Bồi thường (`Claims_per_Year`):** Số lần yêu cầu bồi thường chia cho số năm tham gia, đo lường lịch sử lái xe chính xác hơn thay vì chỉ đếm số lần tai nạn thô.
*   **Chỉ số Hao mòn Sức khỏe (`Health_Age_Index`):** Tương tác giữa điểm sức khỏe và tuổi tác, tạo ra hệ số rủi ro y tế kép.
*   
## 5. Tiền xử lý

Do dữ liệu chứa các giá trị khuyết thiếu và các kiểu dữ liệu phi số học, hệ thống sử dụng `ColumnTransformer` để định tuyến các đặc trưng qua các luồng xử lý chuyên biệt:

* **Đặc trưng Số học (Numerical Features):** Các ô dữ liệu trống được điền khuyết bằng **Trung vị (Median)** của cột tương ứng, giúp mô hình ít bị ảnh hưởng bởi các giá trị ngoại lai.
* **Đặc trưng Phân loại (Categorical Features):**
  1. Dữ liệu khuyết thiếu được điền bằng hằng số (chuỗi văn bản `"Missing"`).
  2. Các biến này sau đó trải qua quá trình **Mã hóa One-Hot (One-Hot Encoding)**. Quá trình này chuyển đổi các biến phân loại thành một ma trận thưa chứa các vector nhị phân. Ví dụ: Nếu một cột có 3 phân loại $[Giỏi, Khá, Xuất Sắc]$, nó sẽ được biến đổi thành 3 cột nhị phân độc lập. Một mẫu dữ liệu thuộc loại $Giỏi$ sẽ được biểu diễn dưới dạng vector $[1, 0, 0]$.

* **Kết quả:** Dữ liệu đầu ra của bước này là một ma trận hoàn toàn mang tính số học, không còn giá trị khuyết, sẵn sàng để đưa vào thuật toán.

## 6. Chia dữ liệu và đánh giá cục bộ

Để đảm bảo quá trình đánh giá mô hình khách quan và giảm thiểu rủi ro lệch phân phối dữ liệu, một chiến lược chia dữ liệu Train-Validation theo tỷ lệ 80/20 được áp dụng.

* **Phân hoạch mục tiêu:** Cột giá tiền được phân hoạch thành 20 khoảng dựa trên các phân vị. Kỹ thuật này đảm bảo số lượng mẫu trong mỗi khoảng phân hoạch là hoàn toàn tương đương nhau.

* **Chia dữ liệu:** Sử dụng 20 khoảng trên làm tiêu chí phân tầng. Cụ thể, hệ thống sẽ truy xuất vào **bên trong từng khoảng một** và thực hiện rút ngẫu nhiên đúng **80% số lượng mẫu để đưa vào tập huấn luyện (`X_train`, `y_train`), và 20% còn lại đưa vào tập kiểm thử cục bộ (`X_val`, `y_val`)**. Cơ chế chia này đảm bảo cấu trúc và phân phối giá tiền của tập Validation luôn là một bản sao thu nhỏ của tập Train ban đầu.

* **Đánh giá:** Mô hình được huấn luyện trên tập 80% và kiểm thử trên tập 20% thông qua thang đo RMSLE. Tại bước này, các siêu tham số (Hyperparameters) như tốc độ học (`learning_rate`) hay độ sâu của cây (`max_depth`) được tinh chỉnh lặp đi lặp lại để tối ưu hóa điểm số.

## 7. Huấn luyện trên toàn bộ tập dữ liệu

Dùng kiến trúc mô hình tốt nhất sau khi xác định được bộ siêu tham số tối ưu thông qua quá trình đánh giá cục bộ.

* **Huấn luyện toàn diện:** Mô hình XGBRegressor cuối cùng được huấn luyện lại trên **100%** tập dữ liệu ban đầu (`X` và `y`). Việc không giữ lại tập Validation ở bước này giúp mô hình tối đa hóa được lượng thông tin học hỏi.
* **Xuất file CSV:** Mô hình sẽ tiếp nhận tập `X_test` đã qua tiền xử lý để đưa ra các dự đoán cuối cùng. Các kết quả dự đoán này sau đó được ghép nối với tập hợp `id` đã cất riêng ban đầu, và xuất ra tệp CSV có cấu trúc chuẩn khớp hoàn toàn với định dạng của `sample_submission.csv`.
