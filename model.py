import numpy as np
import warnings
import itertools

# 1. StandardScaler
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit_transform(self, X):
        X_np = np.array(X, dtype=float)
        self.mean_ = np.mean(X_np, axis=0)
        self.scale_ = np.std(X_np, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Prevent division by zero
        return (X_np - self.mean_) / self.scale_
        
    def transform(self, X):
        X_np = np.array(X, dtype=float)
        return (X_np - self.mean_) / self.scale_

# 2. train_test_split (with stratify for classification)
def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    X_np = np.array(X)
    y_np = np.array(y)
    n_samples = len(X_np)
    
    if stratify is not None:
        stratify_np = np.array(stratify)
        classes, y_indices = np.unique(stratify_np, return_inverse=True)
        
        n_test = int(np.ceil(n_samples * test_size)) if isinstance(test_size, float) else test_size
        n_train = n_samples - n_test
        
        # Calculate ideal test sizes per class
        class_counts = np.bincount(y_indices)
        test_class_counts = np.round(n_test * class_counts / n_samples).astype(int)
        
        # Adjust if rounding causes total to mismatch n_test
        diff = np.sum(test_class_counts) - n_test
        if diff > 0:
            for i in range(diff): test_class_counts[np.argmax(test_class_counts)] -= 1
        elif diff < 0:
            for i in range(-diff): test_class_counts[np.argmin(test_class_counts)] += 1
            
        train_idx = []
        test_idx = []
        
        for i, c in enumerate(classes):
            c_idx = np.where(stratify_np == c)[0]
            np.random.shuffle(c_idx)
            t_count = test_class_counts[i]
            test_idx.extend(c_idx[:t_count])
            train_idx.extend(c_idx[t_count:])
            
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
    else:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_point = int(n_samples * (1 - test_size))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        
    return X_np[train_idx], X_np[test_idx], y_np[train_idx], y_np[test_idx]

# 3. Metrics
def accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def roc_auc_score(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    desc_score_indices = np.argsort(y_scores, kind="stable")[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Gộp các điểm có score trùng nhau (giống sklearn)
    distinct_idx = np.where(np.diff(y_scores_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(y_true_sorted) - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idx]
    fps = 1 + threshold_idx - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] == 0 or tps[-1] == 0:
        return 0.5  # Undefined AUC

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Tương thích cả NumPy 2.0+ (np.trapezoid) và NumPy cũ (np.trapz)
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(tpr, fpr)
    else:
        auc = np.trapz(tpr, fpr)
    return auc

def average_precision_score(y_true, y_scores):
    """Tinh PR-AUC (Area Under Precision-Recall Curve).
    Phu hop hon ROC AUC khi du lieu mat can bang (imbalanced).
    Equivalent voi sklearn.metrics.average_precision_score."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    desc_idx = np.argsort(y_scores, kind='stable')[::-1]
    y_true_sorted = y_true[desc_idx]

    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0

    tp_cumsum = np.cumsum(y_true_sorted)
    n_pred_arr = np.arange(1, len(y_true_sorted) + 1)

    precision_arr = tp_cumsum / n_pred_arr
    recall_arr    = tp_cumsum / total_pos

    # Them diem (recall=0, precision=1) vao dau
    precision_arr = np.r_[1.0, precision_arr]
    recall_arr    = np.r_[0.0, recall_arr]

    # AP = sum(delta_recall * precision) — khong dung trapz de tranh over-estimate
    ap = float(np.sum((recall_arr[1:] - recall_arr[:-1]) * precision_arr[1:]))
    return ap

# 4. Logistic Regression
class LogisticRegression:
    def __init__(self, max_iter=1000, random_state=42, C=1.0, learning_rate=0.01, penalty='l2', tol=1e-4, l1_ratio=0.5, class_weight=None, init='zeros'):
        self.max_iter = max_iter
        self.random_state = random_state
        self.C = C  # Inverse of regularization strength (lambda = 1/C)
        self.learning_rate = learning_rate
        self.penalty = penalty  # 'l1', 'l2', 'elasticnet', or None
        self.tol = tol # Early stopping tolerance
        self.l1_ratio = l1_ratio  # Ty le L1 khi dung penalty='elasticnet'
        self.class_weight = class_weight  # None or 'balanced'
        self.init = init  # 'zeros' hoac 'random' (khoi tao trong so)
        self.weights = None
        self.bias = 0.0
        self.loss_history_ = {'total': [], 'bce': [], 'penalty': []}
        
    def _sigmoid(self, z):
        # Cong thuc chong tran so (numerical stability)
        # z >= 0: 1 / (1 + exp(-z))
        # z < 0: exp(z) / (1 + exp(z))
        z = np.clip(z, -250, 250)
        return np.where(z >= 0, 
                        1.0 / (1.0 + np.exp(-z)), 
                        np.exp(z) / (1.0 + np.exp(z)))

    def _get_sample_weights(self, y):
        n_samples = len(y)
        if self.class_weight == 'balanced':
            n_classes = 2
            counts = np.bincount(y, minlength=2)
            # Tinh toan chinh xac: w_c = n_samples / (n_classes * count_c)
            w0 = n_samples / (n_classes * counts[0]) if counts[0] > 0 else 1.0
            w1 = n_samples / (n_classes * counts[1]) if counts[1] > 0 else 1.0
            sample_weights = np.where(y == 0, w0, w1)
            return sample_weights, {0: float(w0), 1: float(w1)}
        else:
            return np.ones(n_samples), {0: 1.0, 1: 1.0}

    def compute_loss(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)
        linear_model = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(linear_model)
        p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
        
        sample_weights, _ = self._get_sample_weights(y)
            
        bce = -(y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip))
        bce_loss = float(np.sum(bce * sample_weights) / n_samples)
        
        lambda_reg = 1.0 / self.C if (self.C is not None and self.C > 0) else 0.0
        if self.penalty == 'l2':
            pen_loss = float((lambda_reg / 2.0) * np.sum(self.weights ** 2))
        elif self.penalty == 'l1':
            pen_loss = float(lambda_reg * np.sum(np.abs(self.weights)))
        elif self.penalty == 'elasticnet':
            l1_part = self.l1_ratio * np.sum(np.abs(self.weights))
            l2_part = 0.5 * (1.0 - self.l1_ratio) * np.sum(self.weights ** 2)
            pen_loss = float(lambda_reg * (l1_part + l2_part))
        else:
            pen_loss = 0.0
            
        return {
            'total_loss': float(bce_loss + pen_loss),
            'bce_loss': float(bce_loss),
            'penalty_loss': float(pen_loss),
            'l1_norm': float(np.sum(np.abs(self.weights))),
            'l2_norm': float(np.sum(self.weights ** 2)),
            'zero_weights': int(np.sum(np.abs(self.weights) < 1e-8))
        }
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Khoi tao trong so va bias (Tuy chon 'zeros' hoac 'random')
        if self.init == 'random':
            if self.random_state is not None:
                np.random.seed(self.random_state)
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = float(np.random.randn(1)[0] * 0.01)
        else:  # mac dinh: 'zeros'
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            
        self.loss_history_ = {'total': [], 'bce': [], 'penalty': []}
        
        sample_weights, self.class_weights_ = self._get_sample_weights(y)
        lambda_reg = 1.0 / self.C if (self.C is not None and self.C > 0) else 0.0
        
        prev_loss = np.inf
        
        for i in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            p = self._sigmoid(linear_model)
            
            p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
            bce_term = -(y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip))
            bce_loss = float(np.sum(bce_term * sample_weights) / n_samples)
            
            if self.penalty == 'l2':
                pen_loss = float((lambda_reg / 2.0) * np.sum(self.weights ** 2))
            elif self.penalty == 'l1':
                pen_loss = float(lambda_reg * np.sum(np.abs(self.weights)))
            elif self.penalty == 'elasticnet':
                l1_p = self.l1_ratio * np.sum(np.abs(self.weights))
                l2_p = 0.5 * (1.0 - self.l1_ratio) * np.sum(self.weights ** 2)
                pen_loss = float(lambda_reg * (l1_p + l2_p))
            else:
                pen_loss = 0.0
                
            total_loss = bce_loss + pen_loss
            self.loss_history_['bce'].append(bce_loss)
            self.loss_history_['penalty'].append(pen_loss)
            self.loss_history_['total'].append(total_loss)
            
            # Early stopping check
            if abs(prev_loss - total_loss) < self.tol:
                # print(f"Hội tụ sớm tại epoch {i+1} do thay đổi loss < {self.tol}.")
                break
            prev_loss = total_loss
            
            # Tinh Gradient tuong minh
            error = p - y
            error_weighted = error * sample_weights
            
            # Dao ham cua Loss (BCE) theo w va b
            grad_bce_w = (1.0 / n_samples) * np.dot(X.T, error_weighted)
            grad_bce_b = (1.0 / n_samples) * np.sum(error_weighted)
            
            self.bias -= self.learning_rate * grad_bce_b
            
            # Dao ham phan Penalty (Regularization) - su dung sub-gradient cho L1
            if self.penalty == 'l2':
                grad_pen_w = lambda_reg * self.weights
                dw = grad_bce_w + grad_pen_w
                self.weights -= self.learning_rate * dw
            elif self.penalty == 'l1':
                # Dao ham cua |w| la sign(w), tai 0 coi nhu la 0
                grad_pen_w = lambda_reg * np.sign(self.weights)
                dw = grad_bce_w + grad_pen_w
                self.weights -= self.learning_rate * dw
            elif self.penalty == 'elasticnet':
                grad_pen_l1 = lambda_reg * self.l1_ratio * np.sign(self.weights)
                grad_pen_l2 = lambda_reg * (1.0 - self.l1_ratio) * self.weights
                dw = grad_bce_w + grad_pen_l1 + grad_pen_l2
                self.weights -= self.learning_rate * dw
            else:
                self.weights -= self.learning_rate * grad_bce_w
            
        return self
        
    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        prob_1 = self._sigmoid(linear_model)
        prob_0 = 1 - prob_1
        return np.column_stack((prob_0, prob_1))
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
        
    def get_params(self, deep=True):
        return {
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'C': self.C,
            'learning_rate': self.learning_rate,
            'penalty': self.penalty,
            'tol': self.tol,
            'l1_ratio': self.l1_ratio,
            'class_weight': self.class_weight,
            'init': self.init
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# 5. StratifiedKFold
class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        if self.random_state is not None and not self.shuffle:
            warnings.warn(
                "StratifiedKFold: random_state has no effect when shuffle=False. "
                "Set shuffle=True to enable reproducible random shuffling.",
                UserWarning,
                stacklevel=2
            )
        
    def split(self, X, y):
        if self.shuffle and self.random_state is not None:
            np.random.seed(self.random_state)
            
        y = np.array(y)
        classes, y_indices = np.unique(y, return_inverse=True)
        n_samples = len(y)
        
        class_indices = [np.where(y == c)[0] for c in classes]
        if self.shuffle:
            for indices in class_indices:
                np.random.shuffle(indices)
                
        folds = [[] for _ in range(self.n_splits)]
        
        for indices in class_indices:
            current_fold = 0
            for idx in indices:
                folds[current_fold].append(idx)
                current_fold = (current_fold + 1) % self.n_splits
                
        for i in range(self.n_splits):
            test_indices = folds[i]
            train_indices = []
            for j in range(self.n_splits):
                if j != i:
                    train_indices.extend(folds[j])
            yield np.array(train_indices), np.array(test_indices)

# 6. GridSearchCV (Zero-Leakage Hyperparameter Tuning)
class GridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring='roc_auc', scaler=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.scaler = scaler
        self.scaler_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        
    def _generate_param_combinations(self, param_grid):
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
            
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        best_score = -np.inf
        best_params = None
        best_fold_scores = None

        # cv_results_table_: danh sach dict cho moi to hop param da thu,
        # bao gom mean_f1 va std_f1 duoc tinh tu tat ca fold scores thuc te.
        self.cv_results_table_ = []
        # cv_results_: luu danh sach diem theo tung fold, key = str(params)
        self.cv_results_ = {}
        
        for params in self._generate_param_combinations(self.param_grid):
            self.estimator.set_params(**params)
            
            scores = []
            for train_idx, test_idx in self.cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Zero-Leakage: Chuan hoa rieng biet ben trong tung Fold CV
                if self.scaler is not None:
                    fold_scaler = type(self.scaler)()
                    X_train = fold_scaler.fit_transform(X_train)
                    X_test = fold_scaler.transform(X_test)
                
                model = type(self.estimator)(**self.estimator.get_params())
                model.fit(X_train, y_train)
                
                if self.scoring == 'roc_auc':
                    y_proba = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_proba)
                elif self.scoring == 'pr_auc':
                    y_proba = model.predict_proba(X_test)[:, 1]
                    score = average_precision_score(y_test, y_proba)
                elif self.scoring == 'f1':
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred)
                elif self.scoring == 'recall':
                    y_pred = model.predict(X_test)
                    score = recall_score(y_test, y_pred)
                elif self.scoring == 'precision':
                    y_pred = model.predict(X_test)
                    score = precision_score(y_test, y_pred)
                else:
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                scores.append(score)
                
            mean_score = np.mean(scores)
            std_score  = np.std(scores)
            params_key = str(params)
            self.cv_results_[params_key] = list(scores)

            # Luu vao bang tong hop: moi dong la 1 to hop param + mean + std thuc
            row = {k: v for k, v in params.items()}
            row['mean_f1'] = float(mean_score)
            row['std_f1']  = float(std_score)
            self.cv_results_table_.append(row)

            print(f"Tested params {params}: Score = {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score       = mean_score
                best_params      = params.copy()
                best_fold_scores = list(scores)
                
        self.best_score_  = best_score
        self.best_params_ = best_params
        # best_scores_: danh sach n_splits diem F1 theo tung fold cua best_params_.
        # Dung de tinh std that su: np.std(grid_search.best_scores_).
        self.best_scores_ = best_fold_scores
        
        # Huan luyen best_estimator_ tren toan bo tap du lieu
        if self.scaler is not None:
            self.scaler_ = type(self.scaler)()
            X_fit = self.scaler_.fit_transform(X)
        else:
            X_fit = X
            
        self.estimator.set_params(**self.best_params_)
        self.best_estimator_ = type(self.estimator)(**self.estimator.get_params())
        self.best_estimator_.fit(X_fit, y)
        return self
        
    def predict(self, X):
        X = np.array(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.best_estimator_.predict(X)
        
    def predict_proba(self, X):
        X = np.array(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.best_estimator_.predict_proba(X)


class CreditDefaultInferencePipeline:
    """
    Pipeline suy luận tự động hóa cho mô hình dự báo rủi ro tín dụng.
    Tự động tiếp nhận dữ liệu khách hàng mới (thô), tiền xử lý chuẩn hóa,
    tính toán xác suất và phân định quyết định cấp tín dụng theo ngưỡng tối ưu.
    """
    def __init__(self, model, scaler, threshold, feature_names):
        self.model = model
        self.scaler = scaler
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        
    def predict_proba(self, X_input):
        X_mat = self._prepare_matrix(X_input)
        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict_proba(X_scaled)[:, 1]
        
    def predict(self, X_input):
        proba = self.predict_proba(X_input)
        return (proba >= self.threshold).astype(int)
        
    def _prepare_matrix(self, X_input):
        if hasattr(X_input, 'columns'):  # DataFrame
            missing = [c for c in self.feature_names if c not in X_input.columns]
            if missing:
                raise ValueError(f"Dữ liệu đầu vào thiếu các đặc trưng: {missing}")
            return X_input[self.feature_names].values.astype(float)
        else:
            arr = np.array(X_input, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"Số lượng đặc trưng không khớp: kỳ vọng {len(self.feature_names)}, nhận {arr.shape[1]}")
            return arr


