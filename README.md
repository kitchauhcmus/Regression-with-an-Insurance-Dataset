# Regression with an Insurance Dataset

## 1. Tổng quan bài toán và cấu trúc tập dữ liệu

Dự án này giải quyết bài toán Hồi quy (Regression) dựa trên Bộ dữ liệu Bảo hiểm từ cuộc thi **[Kaggle Playground Series - Season 4 Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12)**. Mục tiêu của mô hình là dự đoán Phí bảo hiểm (`Premium Amount`) cho khách hàng dựa trên các thuộc tính của họ.

Dữ liệu được cung cấp bao gồm ba thành phần chính:

* **Tập Huấn luyện (`train.csv`):** Bao gồm 21 cột (1 định danh `id`, 19 cột đặc trưng đầu vào, và 1 biến mục tiêu liên tục là `Premium Amount`).
* **Tập Kiểm thử (`test.csv`):** Bao gồm 20 cột (cấu trúc đặc trưng giống hệt tập huấn luyện, nhưng không có biến mục tiêu).
* **Mẫu Nộp bài (`sample_submission.csv`):** Định dạng chuẩn yêu cầu để nộp bài, chỉ bao gồm đúng 2 cột: `id` và giá trị dự đoán `Premium Amount`.

## 2. Import Libraries
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

Dữ liệu được nạp trực tiếp từ hệ thống lưu trữ của cuộc thi trên nền tảng Kaggle. Bước cấu hình sẽ cô lập các đặc trưng tính toán khỏi siêu dữ liệu (metadata) và biến mục tiêu.

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
## 5. PREPROCESSOR

Do dữ liệu chứa các giá trị khuyết thiếu và các kiểu dữ liệu phi số học, hệ thống sử dụng `ColumnTransformer` để định tuyến các đặc trưng qua các luồng xử lý chuyên biệt:

* **Đặc trưng Số học (Numerical Features):** Các ô dữ liệu trống được điền khuyết bằng **Trung vị (Median)** của cột tương ứng, giúp mô hình ít bị ảnh hưởng bởi các giá trị ngoại lai.
* **Đặc trưng Phân loại (Categorical Features):**
  1. Dữ liệu khuyết thiếu được điền bằng hằng số (chuỗi văn bản `"Missing"`).
  2. Các biến này sau đó trải qua quá trình **Mã hóa One-Hot (One-Hot Encoding)**. Quá trình này chuyển đổi các biến phân loại thành một ma trận thưa chứa các vector nhị phân. Ví dụ: Nếu một cột có 3 phân loại $[Giỏi, Khá, Xuất Sắc]$, nó sẽ được biến đổi thành 3 cột nhị phân độc lập. Một mẫu dữ liệu thuộc loại $Giỏi$ sẽ được biểu diễn dưới dạng vector $[1, 0, 0]$.

* **Kết quả:** Dữ liệu đầu ra của bước này là một ma trận hoàn toàn mang tính số học, không còn giá trị khuyết, sẵn sàng để đưa vào thuật toán.

## 6. VALIDATION & XGBOOST

* **Chia dữ liệu:** Thay vì chia cắt ngẫu nhiên 80/20 trên toàn bộ đám đông, hệ thống sắp xếp cột giá tiền và cắt thành 20 nhóm (bins) từ thấp đến cao[cite: 48]. Sau đó, thuật toán `train_test_split` sẽ chui vào *từng nhóm một*, trích ra đúng 80% cho tập Huấn luyện (Train) và 20% cho tập Kiểm thử (Validation)[cite: 48]. Kỹ thuật này đảm bảo phân phối giá tiền (tỷ lệ hợp đồng giá rẻ, trung bình, VIP) ở cả hai tập là giống hệt nhau, giúp mô hình được đánh giá một cách khách quan nhất.
* **Logarithmic Transformation (Biến đổi Logarit):** Ở mỗi lượt huấn luyện, biến mục tiêu `y` (giá tiền) được ép qua hàm `np.log1p` trước khi đưa vào thuật toán[cite: 48, 49]. Điều này giúp làm "mềm" **các giá trị ngoại lai (Outliers)** hoặc các hợp đồng có mức phí đột biến, đồng thời đồng bộ hóa hoàn toàn hàm mục tiêu (MSE) của XGBoost với thang đo chấm điểm của cuộc thi (RMSLE)[cite: 48].
* **Cấu hình Siêu tham số (Hyperparameters):** Mô hình XGBoost được tinh chỉnh với các tham số[cite: 49]:
    * `n_estimators=1000` & `learning_rate=0.01`: Cấu hình sử dụng một lượng lớn cây quyết định (1000 cây) nhưng ép mô hình học với tốc độ bước tiến rất nhỏ (0.01)[cite: 49]. Điều này giúp mô hình hội tụ từ từ, tìm ra quy luật tinh tế nhất mà không bị "trượt" qua điểm tối ưu.
    * `max_depth=8`: Giới hạn độ sâu (độ phức tạp) của mỗi cây ở mức 8 tầng để mô hình không ghi nhớ máy móc (học vẹt/Overfitting) dữ liệu nhiễu[cite: 49].
    * `subsample=0.8` & `colsample_bytree=0.8`: Ở mỗi cây, hệ thống chỉ lấy ngẫu nhiên 80% lượng dữ liệu (khách hàng) và 80% số lượng đặc trưng (cột)[cite: 49]. Cơ chế này ép mô hình phải nhìn bài toán từ nhiều góc độ khác nhau thay vì phụ thuộc vào một vài cột hoặc một vài tệp khách hàng cố định.
    * `min_child_weight=3`: Tăng tính bảo thủ cho mô hình; một quy luật chỉ được công nhận nếu nó được hỗ trợ bởi một lượng dữ liệu tối thiểu, giúp triệt tiêu các suy luận vô căn cứ[cite: 49].
    * `objective="reg:squarederror"`: Sử dụng hàm mất mát bình phương tối thiểu (MSE) làm mục tiêu tối ưu[cite: 49].
* **Cấu hình GPU:** Quá trình huấn luyện sử dụng tham số `tree_method='hist'` và `device='cuda'` để tận dụng tối đa sức mạnh tính toán song song của GPU trên nền tảng Kaggle[cite: 48, 49].

## 7. Huấn luyện toàn bộ & Kết xuất (Full Training & Prediction)

Sau khi tìm được cấu hình tối ưu và chứng minh tính hiệu quả qua tập Validation[cite: 49], luồng xử lý cuối cùng được thực hiện như sau:

* **Huấn luyện 100% dữ liệu:** Khởi tạo lại mô hình XGBoost với cấu hình xuất sắc nhất và tiến hành học trên toàn bộ 100% dữ liệu gốc (`X_full`), tối đa hóa lượng thông tin đầu vào.
* **Dự đoán & Dịch ngược:** Mô hình tiến hành dự đoán trên tập `test.csv`[cite: 49]. Do biến mục tiêu đã bị ép qua biến đổi Logarit lúc học, các kết quả dự đoán này sẽ được dịch ngược về giá trị tiền tệ thực tế bằng hàm lũy thừa `np.expm1`[cite: 49].
* **Kiểm soát giá trị âm (Clipping):** Hàm `np.clip(val_pred, 0, None)` được sử dụng như một chốt chặn an toàn cuối cùng, tự động ép mọi dự đoán rủi ro có giá trị âm trở về mức 0 (bởi vì phí bảo hiểm thực tế không thể nhỏ hơn 0)[cite: 49].
* **Định dạng đầu ra:** Các dự đoán hoàn chỉnh được ghép nối với cột `id` và xuất thẳng ra file `submission_final.csv`.

## 8. Kết quả (Leaderboard Scores)

* **Public Score:** 1.04534
* **Private Score:** 1.04745
