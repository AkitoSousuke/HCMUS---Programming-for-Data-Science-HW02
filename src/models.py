import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-9))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r / (p + r + 1e-9)

class LinearRegressionGD:
    """
    Hồi quy tuyến tính sử dụng thuật toán tối ưu Gradient Descent.
    Loss Function: Mean Squared Error (MSE)
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.history = []

        for i in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            loss = mean_squared_error(y, y_pred)
            self.history.append(loss)

            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

class KNN:
    """
    K-Nearest Neighbors implementation (From Scratch).
    Hỗ trợ cả Classification và Regression.
    """
    def __init__(self, k=3, mode='classification'):
        self.k = k
        self.mode = mode
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        if self.mode == 'classification':
            # Lấy label xuất hiện nhiều nhất
            counts = np.bincount(k_nearest_labels.astype(int))
            return np.argmax(counts)
        else:
            # Tính trung bình
            return np.mean(k_nearest_labels)
        
class LogisticRegressionNP:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize(self, n_features):
        self.w = np.zeros(n_features)

    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(float)

        n_samples, n_features = X.shape
        self.initialize(n_features)

        losses = []

        for _ in range(self.epochs):
            linear = X.dot(self.w) + self.b
            preds = self.sigmoid(linear)

            # Loss (binary cross entropy)
            loss = -np.mean(y*np.log(preds+1e-9) + (1-y)*np.log(1-preds+1e-9))
            losses.append(loss)

            # Gradients
            dw = np.dot(X.T, (preds - y)) / n_samples
            db = np.mean(preds - y)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return losses

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def k_fold_split(X, k=5, seed=42):
    """
    Generator trả về các chỉ số train/val cho K-Fold Cross Validation.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Shuffle indices
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Chia thành k phần
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        yield train_indices, val_indices
        current = stop

def impute_linear_regression(X, y):
    """
    Model Predictor: Dùng Hồi quy tuyến tính để dự đoán giá trị thiếu.
    Giải bài toán: y = wX + b bằng Normal Equation.
    
    X: Feature dùng để dự đoán (ví dụ: user_followers) - Phải đầy đủ, ko NaN.
    y: Target có chứa NaN cần điền (ví dụ: retweets).
    """
    # 1. Tách dữ liệu thành 2 tập: Train (y có giá trị) và Target (y là NaN)
    mask_valid = ~np.isnan(y)
    mask_missing = np.isnan(y)
    
    if np.sum(mask_missing) == 0:
        return y # Không có gì để điền
    
    X_train = X[mask_valid]
    y_train = y[mask_valid]
    X_missing = X[mask_missing]
    
    # 2. Thêm cột Bias (hệ số chặn) vào X
    # X_train_b = [1, x]
    ones = np.ones((X_train.shape[0], 1))
    X_train_b = np.hstack([ones, X_train.reshape(-1, 1)])
    
    # 3. Tính toán trọng số w bằng Normal Equation: w = (X^T * X)^-1 * X^T * y
    # Sử dụng np.linalg.pinv (Pseudo-inverse) để tránh lỗi ma trận không khả nghịch
    w = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
    
    # 4. Dự đoán giá trị thiếu
    ones_missing = np.ones((X_missing.shape[0], 1))
    X_missing_b = np.hstack([ones_missing, X_missing.reshape(-1, 1)])
    y_pred = X_missing_b @ w
    
    # 5. Điền vào
    y_filled = y.copy()
    y_filled[mask_missing] = y_pred
    
    # Đảm bảo không có giá trị âm (nếu dữ liệu gốc là số lượng)
    y_filled = np.where(y_filled < 0, 0, y_filled)
    
    return y_filled
