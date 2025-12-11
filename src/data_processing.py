import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Loading data from CSV
def load_csv(path):
    """
    Load CSV safely for mixed text data containing commas.
    Instead of letting NumPy parse columns, we read raw lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Bỏ header
    header = lines[0]
    rows = lines[1:]

    return np.array(rows), header

def parse_csv_line(line):
    # Loại bỏ dấu ngoặc kép bao quanh
    parts = []
    temp = ""
    inside_quote = False

    for ch in line:
        if ch == '"':
            inside_quote = not inside_quote
        elif ch == ',' and not inside_quote:
            parts.append(temp)
            temp = ""
        else:
            temp += ch

    parts.append(temp)  # cột cuối
    return parts


# Processing missing values
def replace_missing_with_value(arr, value=""):
    arr[arr == ""] = value
    return arr

# Encode from text to numeric values
def label_encode(arr):
    """
    Convert categorical string array into integer codes.
    Example: ["positive", "negative", "neutral"] → [0,1,2]
    """
    unique_vals = np.unique(arr)
    mapping = {val: i for i, val in enumerate(unique_vals)}
    encoded = np.array([mapping[v] for v in arr])
    return encoded, mapping

# Normalization functions
def min_max_scale(x):
    x = x.astype(float)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-9)

def standardize_zscore(x):
    x = x.astype(float)
    mean = np.mean(x)
    std = np.std(x) + 1e-9
    return (x - mean) / std

def log_transform(x):
    x = x.astype(float)
    return np.log1p(np.abs(x))

# Calculate tweet's length
def compute_tweet_length(text_arr):
    """Feature: number of characters."""
    return np.array([len(t) for t in text_arr], dtype=float)

# Calculate number of words
def compute_word_count(text_arr):
    """Feature: number of words."""
    return np.array([len(t.split()) for t in text_arr], dtype=float)


# Split train and test, random seed 42
def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def summarize_numeric_features_pretty(data, numeric_cols, replace_missing_with_value):
    """
    Vẽ bảng thống kê đẹp (Min, Max, Mean, Median) bằng matplotlib.
    Không dùng pandas.
    """
    # Chuẩn bị dữ liệu bảng
    header = ["Feature", "Min", "Max", "Mean", "Median"]
    rows = []

    for name, idx in numeric_cols.items():
        col_data = replace_missing_with_value(data[:, idx], '0')

        try:
            values = col_data.astype(float)
            row = [
                name,
                f"{np.min(values):.2f}",
                f"{np.max(values):.2f}",
                f"{np.mean(values):.2f}",
                f"{np.median(values):.2f}",
            ]
        except:
            row = [name, "ERR", "ERR", "ERR", "ERR"]

        rows.append(row)

    # ========== TẠO FIGURE ==========
    fig, ax = plt.subplots(figsize=(10, 1 + 0.42 * len(rows)))
    ax.axis("off")

    # ========== TẠO TABLE ==========
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="right",
        colLoc="center",
    )

    # Font và scale
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.25)

    for col in range(len(header)):
        cell = table[0, col]
        cell.set_facecolor("#dbe5f1")   # xanh nhạt pastel
        cell.set_edgecolor("#888888")
        cell.set_text_props(weight="bold", color="black")

    for i in range(1, len(rows) + 1):
        bg = "#f7f7f7" if i % 2 == 0 else "#ffffff"
        for j in range(len(header)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#c0c0c0")

            # Feature column căn trái
            if j == 0:
                cell._loc = "left"

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.4)

    plt.title(
        "📊 Thống kê cơ bản cho các Features số",
        fontsize=15,
        pad=20,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def compute_skew(arr):
    arr = arr.astype(float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0: return 0
    return np.mean(((arr - mean) / std) ** 3)

# Chuyển đổi mảng string sang float
def to_numeric_safe(arr):
    def convert(x):
        try:
            return float(x)
        except ValueError:
            return np.nan
    return np.array([convert(x) for x in arr])

# Loại bỏ outliers sử dụng Z-Scores
def remove_outliers_zscore(data, threshold=3):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    
    if std == 0:
        return data, np.ones(len(data), dtype=bool)
        
    z_scores = (data - mean) / std
    mask = np.abs(z_scores) < threshold
    return data[mask], mask

# Loại bỏ outliers sử dụng IQR
def remove_outliers_iqr(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask], mask

# Chuẩn hóa về [0, 1] theo min-max
def min_max_scaling(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
        
    return (data - min_val) / (max_val - min_val)

# Chuẩn hóa bằng log-transform
def log_transformation(data):
    data_safe = np.where(data < 0, 0, data) 
    return np.log1p(data_safe)

# Chuẩn hóa bằng Z-Scores
def standardization(data):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    
    # Thêm epsilon (1e-8) để tránh chia cho 0 (Numerical Stability)
    return (data - mean) / (std + 1e-8)

# Decimal Scaling: x = x / 10^j
def decimal_scaling(data):
    max_val = np.nanmax(np.abs(data))
    if max_val == 0: return data
    
    j = np.ceil(np.log10(max_val))
    return data / (10 ** j)

# Điền giá trị bị thiếu theo chiến thuật
def impute_simple(data, strategy='mean'):
    """
    Điền giá trị thiếu bằng mean hoặc median.
    """
    data_filled = data.copy()
    if strategy == 'mean':
        fill_val = np.nanmean(data)
    elif strategy == 'median':
        fill_val = np.nanmedian(data)
    else:
        fill_val = 0
        
    mask = np.isnan(data_filled)
    data_filled[mask] = fill_val
    return data_filled

# Chuyển đổi mảng chuỗi ngày tháng 'YYYY-MM-DD HH:MM:SS' sang np.datetime64
def parse_date_numpy(date_str_arr):
    # Cắt chuỗi để lấy phần ngày 'YYYY-MM-DD' nếu định dạng phức tạp
    # Giả sử format chuẩn trong dataset là 'YYYY-MM-DD HH:MM:SS'
    clean_dates = []
    for d in date_str_arr:
        try:
            # Chỉ lấy 10 ký tự đầu (YYYY-MM-DD) để đơn giản hóa
            clean_dates.append(d[:10]) 
        except:
            clean_dates.append('1970-01-01') # Default error date
            
    return np.array(clean_dates, dtype='datetime64[D]')

# Tạo đặc trưng tuổi tài khoản
def create_account_age_feature(user_created_arr, tweet_date_arr):
    created = parse_date_numpy(user_created_arr)
    tweeted = parse_date_numpy(tweet_date_arr)
    
    # Tính hiệu (kết quả là timedelta), chuyển sang integer (số ngày)
    age_days = (tweeted - created).astype(int)
    return age_days

# Tạo đặc trưng tỉ lệ tương tác = retweets / followers 
def create_engagement_ratio(retweets, followers):
    followers_safe = np.where(followers == 0, 1, followers)
    return retweets / followers_safe

def describe_data(data, name="Data"):
    """
    Tính toán thống kê mô tả.
    """
    print(f"--- Thống kê mô tả: {name} ---")
    print(f"Mean:   {np.nanmean(data):.4f}")
    print(f"Median: {np.nanmedian(data):.4f}")
    print(f"Std:    {np.nanstd(data):.4f}")
    print(f"Min:    {np.nanmin(data):.4f}")
    print(f"Max:    {np.nanmax(data):.4f}")
    print("-" * 30)

def perform_ttest_ind(group1, group2):
    """
    Thực hiện kiểm định T-test 2 mẫu độc lập (Two-sample T-test).
    Giả định phương sai không bằng nhau (Welch's t-test).
    
    Giả thiết H0: Trung bình của 2 nhóm bằng nhau (mu1 = mu2).
    Giả thiết H1: Trung bình của 2 nhóm khác nhau (mu1 != mu2).
    """
    n1 = len(group1)
    n2 = len(group2)
    
    m1 = np.mean(group1)
    m2 = np.mean(group2)
    
    v1 = np.var(group1, ddof=1) 
    v2 = np.var(group2, ddof=1)
    
    # Tính t-statistic
    numerator = m1 - m2
    denominator = np.sqrt((v1 / n1) + (v2 / n2))
    
    if denominator == 0:
        return 0.0
        
    t_stat = numerator / denominator
    
    print(f"Kiểm định T-test (Welch's):")
    print(f"   H0: Mean Group 1 == Mean Group 2")
    print(f"   H1: Mean Group 1 != Mean Group 2")
    print(f"   Mean 1: {m1:.2f}, Mean 2: {m2:.2f}")
    print(f"   T-statistic: {t_stat:.4f}")
    
    is_significant = np.abs(t_stat) > 1.96
    print(f"   Kết luận (mức ý nghĩa 5%): {'Bác bỏ H0 (Khác biệt có ý nghĩa)' if is_significant else 'Chấp nhận H0 (Không đủ bằng chứng khác biệt)'}")
    
    return t_stat