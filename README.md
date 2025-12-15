# Vaccination Tweets Analysis & Prediction with NumPy

> **Mô tả ngắn:** Dự án Phân tích dữ liệu (EDA) và Xây dựng mô hình Máy học (Machine Learning) nhằm nghiên cứu sự lan truyền thông tin và độ tin cậy của các bài đăng về Vaccine trên Twitter. Điểm đặc biệt của dự án là quy trình xử lý và các thuật toán lõi được xây dựng thủ công (from scratch) sử dụng thư viện **NumPy**, thay vì dùng các thư viện cấp cao như Scikit-learn, Pandas, OpenCV,...

---

## Mục lục
1. [Giới thiệu](#1-giới-thiệu)
2. [Dataset](#2-dataset)
3. [Phương pháp & Thuật toán](#3-phương-pháp--thuật-toán)
4. [Cài đặt (Installation)](#4-cài-đặt-installation)
5. [Hướng dẫn sử dụng](#5-hướng-dẫn-sử-dụng)
6. [Kết quả (Results)](#6-kết-quả-results)
7. [Cấu trúc dự án](#7-cấu-trúc-dự-án)
8. [Thách thức & Giải pháp](#8-thách-thức--giải-pháp)
9. [Hướng phát triển](#9-hướng-phát-triển)
10. [Contributors](#10-contributors)
11. [License](#11-license)

---

## 1. Giới thiệu

### Bài toán
Trong bối cảnh đại dịch, thông tin trên mạng xã hội đóng vai trò then chốt trong nhận thức cộng đồng. Dự án này giải quyết hai bài toán chính:
1.  **Regression (Hồi quy):** Dự đoán mức độ lan truyền (Viral) của một bài đăng dựa trên số lượng Retweets.
2.  **Classification (Phân loại):** Xác định uy tín của nguồn tin (dự đoán tài khoản `Verified` hay `Unverified`).

### Động lực & Ứng dụng
* Hiểu được các yếu tố ảnh hưởng đến sự lan truyền thông tin y tế.
* Hỗ trợ lọc tin giả hoặc ưu tiên hiển thị các nguồn tin chính thống (Verified) trong các hệ thống gợi ý.
* Rèn luyện tư duy lập trình toán học và xử lý dữ liệu vector hóa với Python/NumPy.

---

## 2. Dataset

* **Nguồn dữ liệu:** `vaccination_tweets.csv` (Dữ liệu thu thập từ Twitter API về chủ đề tiêm chủng).
* **URL**: https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets
* **Kích thước:**
    * Raw: ~11,000 dòng.
    * Parsed & Cleaned: ~6,000 dòng chất lượng cao dùng cho huấn luyện.
* **Các đặc trưng (Features) chính:**
    * `user_followers`: Số người theo dõi (Numerical).
    * `user_friends`: Số người tài khoản đó đang theo dõi (Numerical).
    * `user_verified`: Trạng thái xác thực (Boolean/Binary) - **Target cho bài toán phân loại**.
    * `date`: Thời gian đăng bài.
    * `retweets`: Số lượt chia sẻ lại - **Target cho bài toán hồi quy**.
    * `text`: Nội dung bài đăng (sử dụng cho EDA).

---

## 3. Phương pháp & Thuật toán

Quy trình được thực hiện qua 3 giai đoạn chính, ưu tiên sử dụng `numpy` để thao tác trên ma trận.

### Quy trình xử lý (Pipeline)
1.  **Parsing thủ công:** Đọc file CSV, tách chuỗi để xử lý các dòng bị lỗi định dạng.
2.  **Data Cleaning:**
    * Loại bỏ ngoại lai (Outliers) của `user_followers` bằng phương pháp **IQR** (Interquartile Range).
    * Công thức: $IQR = Q3 - Q1$, Giữ lại dữ liệu trong khoảng $[Q1 - 1.5*IQR, Q3 + 1.5*IQR]$.
3.  **Imputation (Điền khuyết):** Sử dụng **Linear Regression** để dự đoán và điền giá trị thiếu cho cột `retweets` dựa trên `followers`.
4.  **Feature Engineering:**
    * *Account Age:* Tính tuổi tài khoản từ `user_created`.
    * *Log Transformation:* Áp dụng $log(1+x)$ cho `followers` để xử lý phân phối lệch (skewed).
    * *Standardization:* Chuẩn hóa Z-score cho `friends`.

### Thuật toán (Implemented from scratch)

#### A. Linear Regression (Dùng cho Imputation)
Sử dụng phương pháp **Normal Equation** để tìm trọng số tối ưu $w$ một cách trực tiếp không cần vòng lặp:
$$w = (X^T X)^{-1} X^T y$$
* *NumPy Implementation:* Sử dụng `np.linalg.pinv` để tính nghịch đảo giả (tránh lỗi ma trận suy biến) và `np.dot` cho tích vô hướng.

#### B. K-Nearest Neighbors (KNN - Regression)
Dự đoán số Retweet dựa trên $k$ láng giềng gần nhất.
* **Khoảng cách:** Euclidean Distance $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$.
* **Dự đoán:** Trung bình cộng giá trị target của $k$ láng giềng. $\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$.
* *Kỹ thuật:* Sử dụng Broadcasting của NumPy để tính khoảng cách giữa 1 điểm test và toàn bộ tập train cùng lúc.

#### C. Logistic Regression (Classification)
Dự đoán xác suất tài khoản là Verified.
* **Hàm kích hoạt:** Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Hàm mất mát (Loss Function):** Log-Loss (Binary Cross Entropy).
* **Tối ưu hóa:** Gradient Descent.
    $$w = w - \alpha \cdot \frac{\partial J}{\partial w}$$
    $$\text{với } \frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

---

## 4. Cài đặt (Installation)
### Yêu cầu môi trường
- Python >= 3.8
- Hệ điều hành: Windows / Linux / macOS
- Không yêu cầu GPU

### Thư viện sử dụng
Dự án ưu tiên **NumPy from scratch**, chỉ sử dụng các thư viện sau:
- `numpy`
- `matplotlib`
- `seaborn`
- `tabulate` (trình bày bảng)
- `jupyter`

Cài đặt bằng pip:
bash
pip install -r requirements.txt

---

## 5. Hướng dẫn sử dụng (Usage)

Dự án được tổ chức theo từng notebook tương ứng với các giai đoạn xử lý dữ liệu và xây dựng mô hình.
Dự án gồm 3 notebook, chạy theo đúng thứ tự sau:

### Bước 1: Khám phá dữ liệu (EDA)
notebooks/01_data_exploration.ipynb

**Mục tiêu:**
- Khám phá dữ liệu thô về Vaccine Tweets
- Phát hiện các vấn đề dữ liệu trước khi modeling

**Nội dung chính:**
- Parse CSV thủ công → lọc từ 21,731 dòng thô xuống 6,841 dòng hợp lệ
- Thống kê missing values (hashtags, user_location, description, …)
- Phân tích phân phối các biến số:
  - Followers, Friends, Retweets, Favorites
- Trực quan hóa:
  - Histogram & Boxplot (phân phối lệch mạnh – long-tail)
  - Pie chart: Verified vs Unverified (Verified chỉ ~8.6%)
  - Boxplot so sánh Followers theo Verified
  - Heatmap tương quan
  - Scatter plot Log(Followers) vs Log(Retweets)
  - Line chart xu hướng số lượng tweet theo ngày
  - Bar chart Top Hashtags, Top Sources

**Kết luận EDA chính:**
- Dữ liệu rất lệch (skewness cao) → cần Log-transform
- Followers không có quan hệ tuyến tính đơn giản với Retweets
- Verified accounts chiếm thiểu số nhưng có ảnh hưởng vượt trội
- Cần:
  - Loại bỏ outliers
  - Feature engineering
  - Imputation thông minh

### Bước 2: Tiền xử lý dữ liệu (Preprocessing)
> notebooks/02_preprocessing.ipynb

**Mục tiêu:**
Xây dựng **Clean Numerical Dataset** để đưa vào mô hình học máy

**Các bước chính:**
1. **Data Cleaning**
- Chuyển dữ liệu sang numeric an toàn
- Loại bỏ outliers của `followers` bằng IQR
- Kết quả:
  - Loại bỏ 924 outliers
  - Mean Followers giảm từ ~31,786 → ~843

2. **Feature Engineering**
- `Account Age`: số ngày từ khi tạo tài khoản đến lúc tweet
- `Engagement Ratio`: Retweets / Friends

3. **Imputation**
- Giả lập 100 giá trị retweets bị thiếu
- Dùng **Linear Regression (NumPy)** để điền khuyết
- Sau imputation: 0 missing values

4. **Scaling & Transformation**
- Log-transform: Followers
- Min–Max Scaling: Account Age
- Z-score Standardization: Friends
- Kiểm tra:
    - Mean ≈ 0, Std ≈ 1 (đạt yêu cầu)

5. **Hypothesis Testing**
- Welch’s T-test:
    - H0: Mean Followers (Verified) = Mean Followers (Unverified)
    - Kết quả:
        - T-statistic ≈ 18.07
        - Bác bỏ H0 (p < 0.05)
        - Verified có followers cao hơn có ý nghĩa thống kê

6. **Export** 
- Xuất file:
> data/processed/processed_dataset.csv
- Gồm 5 cột:
    - `followers_log`
    - `friends_std`
    - `1account_age_norm`
    - `retweets_imputed`
    - `label_verified`
 
### Bước 3: Xây dựng mô hình
> jupyter notebook notebooks/03_modeling.ipynb
- Giải quyết 2 bài toán ML bằng NumPy from scratch

---

## 6. Kết quả (Results)

### 6.1. Task A – Regression: Dự đoán số lượng Retweets

**Đầu vào (Features):**
- `followers_log`: Log-transformed Followers
- `friends_std`: Friends (Z-score Standardization)
- `account_age_norm`: Account Age (Min–Max Scaling)

**Đầu ra (Target):**
- `retweets_imputed`: Số Retweets sau khi xử lý missing values

#### Mô hình 1: Linear Regression (Gradient Descent – NumPy)

- Learning rate: 0.01  
- Epochs: 2000  

**Kết quả đánh giá trên tập Test (20%):**
- **MSE:** `15.9308`
- **R²:** `0.0187`

**Nhận xét:**
- Loss (MSE) giảm dần và hội tụ ổn định theo số epoch
- Mô hình học được xu hướng tổng quát nhưng khả năng giải thích phương sai còn hạn chế
- Phù hợp làm baseline cho bài toán hồi quy

#### Mô hình 2: K-Nearest Neighbors (KNN Regression)

- Số láng giềng: `k = 5`
- Khoảng cách: Euclidean Distance

**Kết quả đánh giá trên tập Test:**
- **MSE:** `17.6912`
- **R²:** `-0.0897`

**So sánh hai mô hình Regression:**

| Mô hình | MSE | R² |
|------|-----|----|
| Linear Regression | **15.9308** | **0.0187** |
| KNN Regression (k=5) | 17.6912 | -0.0897 |

**Kết luận Task A:**
- Linear Regression hoạt động **tốt hơn KNN** trên bộ dữ liệu này
- Dữ liệu có phương sai lớn và phân phối lệch làm giảm hiệu quả của KNN
- Quan hệ giữa đặc trưng và Retweets không mang tính cục bộ rõ ràng

### 6.2. Task B – Classification: Dự đoán tài khoản Verified

**Bài toán:** Phân loại nhị phân  
- `1`: Verified  
- `0`: Unverified  

**Mô hình sử dụng:**
- Logistic Regression (Gradient Descent – NumPy)

#### Kết quả đánh giá

| Metric | Giá trị |
|------|--------|
| Accuracy | **0.9814** |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-score | 0.0000 |

#### Phân tích kết quả

- Dataset **mất cân bằng nghiêm trọng**:
  - Verified chiếm khoảng **8–9%**
  - Unverified chiếm hơn **90%**
- Accuracy cao nhưng mô hình **không dự đoán đúng lớp thiểu số**
- Precision, Recall và F1-score bằng 0 cho thấy:
  - Mô hình thiên lệch hoàn toàn về lớp Unverified

**Kết luận Task B:**
- Accuracy **không phải metric phù hợp** cho bài toán mất cân bằng
- F1-score và Recall quan trọng hơn trong bối cảnh này
- Cần cải thiện bằng các kỹ thuật xử lý imbalance

### So sánh & Phân tích tổng hợp

| Bài toán | Mô hình | Nhận xét |
|--------|--------|---------|
| Regression | Linear Regression | Ổn định, dễ diễn giải |
| Regression | KNN | Không phù hợp dữ liệu lệch |
| Classification | Logistic Regression | Bị ảnh hưởng bởi imbalance |

**Insight chính:**
- Log-transformation là bước then chốt giúp mô hình ổn định
- Feature `user_verified` có ý nghĩa thống kê mạnh (đã kiểm định T-test)
- NumPy vectorization giúp kiểm soát toàn bộ pipeline và thuật toán

--- 

## 7. Cấu trúc dự án
project-root/
├── README.md              # Mô tả dự án
├── requirements.txt       # Thư viện cần thiết
├── data/
│   ├── raw/vaccination_tweets.csv               # Dữ liệu gốc
│   └── processed/processed_dataset.csv     # Dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_processing.py # Làm sạch & feature engineering
│   ├── visualization.py  # Hàm vẽ biểu đồ
│   └── models.py          # Các mô hình ML cài đặt bằng NumPy

---

## 8. Thách thức & Giải pháp (Challengers & Solutions)
**Thách thức:**
- Không sử dụng Pandas/Scikit-learn
- Hiệu suất thấp nếu dùng vòng lặp for
- Dữ liệu có phân phối lệch và nhiều ngoại lai

**Giải pháp:**
- Áp dụng vectorization & broadcasting của NumPy
- Sử dụng np.linalg.pinv để đảm bảo ổn định số học
- Áp dụng log-transform và chuẩn hóa dữ liệu
- Tách rõ pipeline xử lý dữ liệu và mô hình

---
### 9. Hướng phát triển tiếp theo
- Áp dụng kỹ thuật xử lý mất cân bằng:
  - Class weighting
  - Threshold tuning
  - Oversampling (nếu cho phép)
- Cross-validation thủ công bằng NumPy
- So sánh kết quả với Scikit-learn (benchmark)
- Mở rộng sang phân tích nội dung (NLP) trong các nghiên cứu tiếp theo

---
## 10. Contributors
- Họ và tên: Nguyễn Hữu Kiến Phi
- MSSV: 23127242
- Môn học: CSC17104 – Programming for Data Science
- Email: nhkphi23@clc.fitus.edu.vn

---
## 11. License

Dự án này được thực hiện **chỉ nhằm mục đích học tập** trong khuôn khổ học phần  
**CSC17104 – Programming for Data Science** tại **Trường Đại học Khoa học Tự nhiên – ĐHQG TP.HCM**.

Mã nguồn và các tài liệu trong dự án **không được sử dụng cho mục đích thương mại**.

