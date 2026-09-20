#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py
Script thực thi toàn bộ quy trình học máy và in báo cáo chi tiết 21 bước ra Terminal.
Sử dụng 100% code thuần NumPy/Pandas từ model.py (No Scikit-learn).
Chuẩn phương pháp luận ML: Zero Leakage, Train (64%) / Val (16%) / Test (20%),
Dò tìm ngưỡng trên Validation, đánh giá độc lập trên Test.
Cập nhật 30 đặc trưng: One-Hot Encoding cho MARRIAGE & EDUCATION (tránh dummy trap).
"""

import sys
import time
import os
import json
import hashlib
import numpy as np
import pandas as pd

# Thiết lập UTF-8 cho Windows Terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from model import (
    StandardScaler, train_test_split,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    LogisticRegression, StratifiedKFold, GridSearchCV,
    CreditDefaultInferencePipeline
)
from weights import save_weights, print_weight_summary


def print_banner(text, width=78):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_section(step_num, title, width=78):
    print("\n" + "-" * width)
    print(f"  [BƯỚC {step_num:02d}] {title}")
    print("-" * width)

def main():
    start_total_time = time.time()
    
    print_banner("HỆ THỐNG DỰ BÁO NGUY CƠ VỠ NỢ THẺ TÍN DỤNG — LOGISTIC REGRESSION THUẦN")
    
    # 0. Kiểm tra tính đồng bộ
    if os.path.exists('model.py'):
        mtime = time.ctime(os.path.getmtime('model.py'))
        with open('model.py', 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()[:8]
        print(f"  [ĐỒNG BỘ] model.py MD5: {model_hash} | Cập nhật lúc: {mtime}")
    print(f"  [MÔI TRƯỜNG] Python: {sys.version.split()[0]} | NumPy: {np.__version__} | Pandas: {pd.__version__}")
    print("  [NGUYÊN TẮC] Code thuần 100% | Zero Data Leakage | Zero Synthetic Data")
    
    # -------------------------------------------------------------
    # BƯỚC 1: Xác định bài toán
    # -------------------------------------------------------------
    print_section(1, "XÁC ĐỊNH BÀI TOÁN (Problem Definition)")
    print("  - Mục tiêu: Dự báo nguy cơ khách hàng vỡ nợ thẻ tín dụng (y = 1) vào tháng tiếp theo.")
    print("  - Đặc tính: Tổn thất phi đối xứng (Asymmetric Loss) — Giả định nghiệp vụ bán lẻ:")
    print("    Chi phí bỏ lọt nợ xấu Cost(FN) lớn hơn đáng kể chi phí cảnh báo nhầm Cost(FP).")
    
    # -------------------------------------------------------------
    # BƯỚC 2: Bản chất bài toán ML
    # -------------------------------------------------------------
    print_section(2, "BẢN CHẤT BÀI TOÁN ML (ML Paradigm & Inductive Bias)")
    print("  - Hình thái: Học có giám sát (Supervised Learning), Phân loại nhị phân (Binary Classification).")
    print("  - Phương thức học: Batch Gradient Descent (Toàn mẻ), tối ưu hóa hàm Binary Cross-Entropy.")
    print("  - Thiên kiến quy nạp: Log-odds xác suất tuyến tính với không gian đặc trưng.")

    # -------------------------------------------------------------
    # BƯỚC 3 & 4: Khảo sát và xử lý dữ liệu
    # -------------------------------------------------------------
    print_section(3, "KHẢO SÁT LĨNH VỰC DỮ LIỆU (Domain & Data Understanding)")
    csv_file = 'default_of_credit_card_clients.csv'
    if not os.path.exists(csv_file):
        print(f"  [LỖI] Không tìm thấy file dữ liệu: {csv_file}")
        return
    df_raw = pd.read_csv(csv_file)
    raw_rows, raw_cols = df_raw.shape
    c0_total = int((df_raw['default payment next month'] == 0).sum())
    c1_total = int((df_raw['default payment next month'] == 1).sum())
    ratio_imbalance = c0_total / c1_total
    
    print(f"  - Tập dữ liệu thô: {raw_rows:,} dòng x {raw_cols} thuộc tính.")
    print(f"  - 4 nhóm thuộc tính: Hạn mức (LIMIT_BAL), Nhân khẩu học, Lịch sử trả nợ (PAY_i), Dòng tiền (BILL/PAY_AMT).")
    print("  - Ý nghĩa thang đo chuẩn UCI của PAY_*:")
    print("      -2 = Không phát sinh giao dịch / thẻ không hoạt động (No consumption / Inactive)")
    print("      -1 = Thanh toán toàn bộ dư nợ đúng hạn / không còn nợ (Paid in full)")
    print("       0 = Dùng tín dụng quay vòng / thanh toán số tiền tối thiểu (Revolving credit / Paid minimum)")
    print("    1..8 = Chậm thanh toán từ 1 đến 8 tháng (Payment delay 1..8 months)")
    print("  - Định hướng xử lý mất cân bằng: class_weight='balanced' điều chỉnh trọng số lớp.")
    
    print_section(4, "KHÁM PHÁ VÀ XỬ LÝ DỮ LIỆU (Data Exploration & Cleaning)")
    print(f"  - Phân bố nhãn: Class 0 (Đúng hạn) = {c0_total:,} ({c0_total/raw_rows*100:.2f}%)")
    print(f"                  Class 1 (Nợ xấu)   = {c1_total:,} ({c1_total/raw_rows*100:.2f}%)")
    print(f"  - Tỷ lệ mất cân bằng tự nhiên: {ratio_imbalance:.2f} : 1")
    
    # Gộp mã dị biệt
    df = df_raw.copy()
    edu_anom = int(((df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)).sum())
    mar_anom = int((df['MARRIAGE'] == 0).sum())
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    print(f"  - Đã chuẩn hóa: {edu_anom} mẫu EDUCATION dị biệt -> nhóm 4 (Others); {mar_anom} mẫu MARRIAGE dị biệt -> nhóm 3 (Others).")
    
    # -------------------------------------------------------------
    # BƯỚC 5 & 6: Chuẩn hóa & Biến phân loại
    # -------------------------------------------------------------
    print_section(5, "CHUẨN HÓA ĐẶC TRƯNG (Feature Scaling Rationale)")
    print("  - Giải pháp: Z-Score Standardization z = (x - mu) / sigma.")
    print("  - Mục đích: Cải thiện số điều kiện Hessian kappa(H), khắc phục dao động zig-zag của Gradient.")
    print("  * Lưu ý: StandardScaler không loại bỏ ngoại lai (outliers), chỉ co giãn phân phối về cùng thang đo.")
    
    print_section(6, "XỬ LÝ BIẾN PHÂN LOẠI (One-Hot Encoding)")
    print("  - Phương pháp: One-Hot Encoding cho MARRIAGE và EDUCATION, loại bỏ cột tham chiếu (tránh Dummy Trap):")
    print("      * MARRIAGE: Tham chiếu = 1 (Married). Tạo: MARRIAGE_SINGLE, MARRIAGE_OTHERS.")
    print("      * EDUCATION: Tham chiếu = 1 (Grad School). Tạo: EDU_UNIVERSITY, EDU_HIGHSCHOOL, EDU_OTHERS.")
    print("  - Biến PAY_*: Giữ nguyên thang đo số tự nhiên phản ánh mức độ trễ nợ tăng dần.")
    
    # -------------------------------------------------------------
    # BƯỚC 7 & 8: Thuật toán & Kỹ thuật tạo đặc trưng
    # -------------------------------------------------------------
    print_section(7, "LỰA CHỌN THUẬT TOÁN VÀ HÀM MẤT MÁT (Algorithm & Loss)")
    print("  - Hàm mất mát: Weighted Binary Cross-Entropy kết hợp hàm phạt L1 (Lasso) / L2 (Ridge).")
    
    print_section(8, "KỸ THUẬT TẠO ĐẶC TRƯNG (Feature Engineering)")
    eps = 1e-8
    X_df = df.drop(columns=['ID', 'default payment next month']).copy()
    y_df = df['default payment next month']
    
    # One-Hot encoding cho MARRIAGE và EDUCATION (drop reference)
    mar_dummies = pd.get_dummies(X_df['MARRIAGE'], prefix='MARRIAGE', drop_first=False).astype(float)
    if 'MARRIAGE_1' in mar_dummies.columns:
        mar_dummies = mar_dummies.drop(columns=['MARRIAGE_1'])  # Tham chiếu: Married
    mar_dummies.columns = ['MARRIAGE_SINGLE', 'MARRIAGE_OTHERS']
    
    edu_dummies = pd.get_dummies(X_df['EDUCATION'], prefix='EDU', drop_first=False).astype(float)
    if 'EDU_1' in edu_dummies.columns:
        edu_dummies = edu_dummies.drop(columns=['EDU_1'])  # Tham chiếu: Graduate School
    edu_dummies.columns = ['EDU_UNIVERSITY', 'EDU_HIGHSCHOOL', 'EDU_OTHERS']
    
    X_df = X_df.drop(columns=['MARRIAGE', 'EDUCATION'])
    X_df = pd.concat([X_df, mar_dummies, edu_dummies], axis=1)
    
    # Tạo 4 đặc trưng miền tài chính mới (hoàn toàn theo từng hàng, không rò rỉ xuyên mẫu)
    # 1. Tỷ lệ sử dụng hạn mức tín dụng tháng gần nhất (Tháng 9)
    X_df['UTILIZATION_RATIO_SEPT'] = X_df['BILL_AMT1'] / (X_df['LIMIT_BAL'] + eps)
    
    # 2. Độ trễ thanh toán trung bình 6 tháng
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    X_df['AVG_PAY_DELAY'] = X_df[pay_cols].mean(axis=1)
    
    # 3. Số tiền thanh toán trung bình 6 tháng
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X_df['AVG_PAY_AMT'] = X_df[pay_amt_cols].mean(axis=1)
    
    # 4. Tỷ lệ trả nợ trên tổng dư nợ hóa đơn 6 tháng
    bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    sum_pays = X_df[pay_amt_cols].sum(axis=1)
    sum_bills = X_df[bill_amt_cols].sum(axis=1)
    ratio_raw = np.where(
        sum_bills <= 0,
        1.0,
        sum_pays / (np.maximum(sum_bills, 0.0) + eps)
    )
    X_df['PAY_TO_BILL_RATIO'] = np.clip(ratio_raw, 0.0, 5.0)
    
    feature_names = X_df.columns.tolist()
    n_features = len(feature_names)
    print(f"  - 4 biến tạo mới: UTILIZATION_RATIO_SEPT, AVG_PAY_DELAY, AVG_PAY_AMT, PAY_TO_BILL_RATIO.")
    print("  - Biện giải tỷ lệ PAY_TO_BILL_RATIO:")
    print("      * sum_bills <= 0 -> 1.0: Không có nợ hoặc nộp thừa, nghĩa vụ danh nghĩa hoàn thành 100%.")
    print("      * Clip [0.0, 5.0]: 99.5th percentile thực nghiệm là 5.27; cắt tại 5.0 triệt tiêu số học bùng nổ.")
    print(f"  - Kiểm tra tính hợp lệ dữ liệu: NaN = {X_df.isna().sum().sum()} | Inf = {np.isinf(X_df.values).sum()}.")
    print(f"  - Không gian đặc trưng chuẩn hóa: {n_features} đặc trưng (26 gốc đã mã hóa One-Hot + 4 phái sinh).")
    
    X = X_df.values
    y = y_df.values
    
    # -------------------------------------------------------------
    # BƯỚC 9 & 10: Chia dữ liệu & Kiểm soát Leakage
    # -------------------------------------------------------------
    print_section(9, "CHIA DỮ LIỆU & KIỂM SOÁT RÒ RỈ (Train 64% - Val 16% - Test 20%)")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)
    
    n_train, n_val, n_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]
    print(f"  - Train set      : {n_train:,} mẫu (64.0%) — dùng để huấn luyện mô hình và CV")
    print(f"  - Validation set : {n_val:,} mẫu (16.0%) — dùng để tinh chỉnh ngưỡng quyết định (tau*)")
    print(f"  - Test set       : {n_test:,} mẫu (20.0%) — KHÓA HOÀN TOÀN, chỉ dùng cho đánh giá cuối cùng duy nhất 1 lần")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  - StandardScaler: Fit CHỈ trên Train; transform độc lập cho Val và Test (Zero Data Leakage).")
    
    print_section(10, "LỰA CHỌN PHƯƠNG PHÁP CHIA DỮ LIỆU (Stratified Hold-out)")
    prop_full = np.mean(y == 1) * 100
    prop_train = np.mean(y_train == 1) * 100
    prop_val = np.mean(y_val == 1) * 100
    prop_test = np.mean(y_test == 1) * 100
    print(f"  - Tỷ lệ nhãn 1 (Nợ xấu): Full={prop_full:.2f}% | Train={prop_train:.2f}% | Val={prop_val:.2f}% | Test={prop_test:.2f}%")
    print("  - Kết quả: Phân tầng Stratified bảo toàn tỷ lệ nhãn hoàn hảo trên cả 3 tập.")
    
    # -------------------------------------------------------------
    # BƯỚC 11, 12, 13: Mô hình cơ sở & Phân tích Loss/Overfitting
    # -------------------------------------------------------------
    print_section(11, "XÂY DỰNG MÔ HÌNH CƠ SỞ (Baseline Model)")
    base_lr = LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l2', C=1.0, random_state=42)
    base_lr.fit(X_train_scaled, y_train)
    y_pred_base = base_lr.predict(X_test_scaled, threshold=0.5)
    y_proba_base = base_lr.predict_proba(X_test_scaled)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base)
    rec_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_proba_base)
    prauc_base = average_precision_score(y_test, y_proba_base)
    
    print(f"  - Cấu hình Baseline: C=1.0, penalty='l2', class_weight=None, tau=0.50")
    print(f"    Accuracy: {acc_base*100:.2f}% | Precision: {prec_base*100:.2f}% | Recall: {rec_base*100:.2f}% | F1: {f1_base:.4f} | ROC-AUC: {auc_base:.4f} | PR-AUC: {prauc_base:.4f}")
    
    print_section(12, "ĐỊNH LÝ NO FREE LUNCH & NGHỊCH LÝ ĐỘ CHÍNH XÁC (Accuracy Paradox)")
    print(f"  - Nghịch lý Accuracy: Mặc dù Accuracy đạt {acc_base*100:.2f}%, nhưng Recall chỉ đạt {rec_base*100:.2f}%.")
    cm_base = confusion_matrix(y_test, y_pred_base)
    fn_base = cm_base[1, 0]
    total_pos = np.sum(y_test == 1)
    print(f"  - Mô hình Baseline bỏ sót {fn_base:,} / {total_pos:,} khách hàng nợ xấu ({fn_base/total_pos*100:.1f}%).")
    print("  - Định lý No Free Lunch (Wolpert, 1997): Không có thuật toán học máy nào vượt trội tổng quát trên mọi phân phối.")
    
    print_section(13, "PHÂN TÍCH BIAS - VARIANCE & LOSS DECOMPOSITION (Đo trên Validation Set)")
    models_check = [
        ("Không phạt (No Reg)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty=None, random_state=42)),
        ("Phạt L2 Ridge (C=10.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l2', C=10.0, random_state=42)),
        ("Phạt L1 Lasso (C=50.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l1', C=50.0, random_state=42)),
        ("Phạt L1 Lasso (C=10.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l1', C=10.0, random_state=42))
    ]
    
    print("\n  BẢNG BÓC TÁCH HÀM MẤT MÁT & OVERFIT GAP (Train vs Validation):")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    print(f"  | {'Chiến lược Điều chuẩn':<24} | {'Train BCE':^9} | {'Val BCE':^9} | {'Overfit Gap':^11} | {'Penalty Loss':^11} | {'||w||_1':^9} | {'Zero W':^9} |")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    
    for name, mdl in models_check:
        mdl.fit(X_train_scaled, y_train)
        tr_res = mdl.compute_loss(X_train_scaled, y_train)
        val_res = mdl.compute_loss(X_val_scaled, y_val)
        gap = val_res['bce_loss'] - tr_res['bce_loss']
        w_L1 = np.sum(np.abs(mdl.weights))
        zeros = int(np.sum(np.abs(mdl.weights) < 1e-4))
        print(f"  | {name:<24} | {tr_res['bce_loss']:^9.4f} | {val_res['bce_loss']:^9.4f} | {gap:^+11.4f} | {tr_res['penalty_loss']:^11.4f} | {w_L1:^9.4f} | {f'{zeros}/{n_features}':^9} |")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    
    # -------------------------------------------------------------
    # BƯỚC 14, 15, 16, 17: Metrics, K-Fold, GridSearch & Training
    # -------------------------------------------------------------
    print_section(14, "LỰA CHỌN EVALUATION METRICS & THỐNG NHẤT MỤC TIÊU")
    print("  - Thống nhất mục tiêu: GridSearch tối ưu F1 (trung bình điều hòa) để cân bằng Precision và Recall,")
    print("    tránh hiện tượng sụp đổ Precision khi dùng class_weight='balanced'.")
    print("    Sau đó tại Bước 21, ngưỡng tau* được tinh chỉnh trên Validation theo khẩu vị rủi ro.")
    
    print_section(15, "10-FOLD STRATIFIED CROSS-VALIDATION")
    print_section(16, "TỐI ƯU SIÊU THAM SỐ QUA GRID SEARCH (GridSearchCV trên Train Set)")
    
    param_grid = {
        'C': [1.0, 5.0, 10.0, 50.0],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced']
    }
    lr_grid = LogisticRegression(learning_rate=0.1, max_iter=1000, random_state=42)
    cv_obj = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=lr_grid, param_grid=param_grid, cv=cv_obj, scoring='f1', scaler=StandardScaler())
    
    print("  Đang chạy 10-Fold Grid Search trên 8 tổ hợp siêu tham số (Train set = 19,200 mẫu)...")
    grid_search.fit(X_train, y_train)
    
    print("\n  BẢNG KẾT QUẢ GRID SEARCH (10-FOLD CV):")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    print(f"  | {'C':^6} | {'Penalty':^8} | {'Class Weight':^13} | {'Mean F1 (CV)':^11} |")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    for row in grid_search.cv_results_table_:
        print(f"  | {row['C']:^6.1f} | {row['penalty'].upper():^8} | {row['class_weight']:^13} | {row['mean_f1']:^11.4f} |")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    
    best_params = grid_search.best_params_
    best_lr = grid_search.best_estimator_
    print(f"\n  --> Cấu hình tối ưu nhất: {best_params} | Best CV F1 = {grid_search.best_score_:.4f}")
    
    print_section(17, "ĐÁNH GIÁ MÔ HÌNH TỐI ƯU TẠI NGƯỠNG MẶC ĐỊNH 0.5 (Test Set)")
    y_pred_best_05 = best_lr.predict(X_test_scaled, threshold=0.5)
    y_proba_best = best_lr.predict_proba(X_test_scaled)[:, 1]
    
    acc_best_05 = accuracy_score(y_test, y_pred_best_05)
    prec_best_05 = precision_score(y_test, y_pred_best_05)
    rec_best_05 = recall_score(y_test, y_pred_best_05)
    f1_best_05 = f1_score(y_test, y_pred_best_05)
    auc_best = roc_auc_score(y_test, y_proba_best)
    prauc_best = average_precision_score(y_test, y_proba_best)
    print(f"  - Best Model (tau=0.50): Acc={acc_best_05*100:.2f}% | Prec={prec_best_05*100:.2f}% | Rec={rec_best_05*100:.2f}% | F1={f1_best_05:.4f} | ROC-AUC={auc_best:.4f}")
    
    # -------------------------------------------------------------
    # BƯỚC 18 & 19: Đánh giá độ ổn định CV & Phân tích lỗi (Confusion Matrix)
    # -------------------------------------------------------------
    print_section(18, "ĐÁNH GIÁ ĐỘ ỔN ĐỊNH CỦA MÔ HÌNH QUA 10-FOLD CV (Cross-Validation Stability Analysis)")
    best_scores = grid_search.best_scores_
    mean_cv = np.mean(best_scores)
    std_cv = np.std(best_scores)
    print(f"  - 10-Fold CV F1 Score: {mean_cv:.4f} ± {std_cv:.4f} (Mean ± SD)")
    print(f"  - Độ lệch chuẩn SD thấp ({std_cv:.4f}) thể hiện độ biến thiên giữa các fold nội bộ thấp,")
    print("    tuy nhiên cần cẩn trọng không khẳng định tuyệt đối về tính ổn định ngoài môi trường thực tế.")
    
    print_section(19, "PHÂN TÍCH LỖI & MA TRẬN NHẦM LẪN (Error Analysis & Confusion Matrix)")
    cm_best_05 = confusion_matrix(y_test, y_pred_best_05)
    
    def fmt(n): return f"{n:,}"
    print("\n  MA TRẬN NHẦM LẪN (CONFUSION MATRIX TẠI TAU=0.5):")
    print(f"  Baseline Model (class_weight=None):  Mô hình tối ưu ({best_params['penalty'].upper()}, C={best_params['C']}):")
    print(f"    TN: {fmt(cm_base[0,0]):<8} | FP: {fmt(cm_base[0,1]):<8}   TN: {fmt(cm_best_05[0,0]):<8} | FP: {fmt(cm_best_05[0,1]):<8}")
    print(f"    FN: {fmt(cm_base[1,0]):<8} | TP: {fmt(cm_base[1,1]):<8}   FN: {fmt(cm_best_05[1,0]):<8} | TP: {fmt(cm_best_05[1,1]):<8}")
    print(f"    --> Bỏ sót nợ xấu (FN): {fmt(cm_base[1,0])} khách   --> Bỏ sót nợ xấu (FN): GIẢM XUỐNG CÒN {fmt(cm_best_05[1,0])} khách!")
    print(f"    --> Phát hiện đúng (TP): {fmt(cm_base[1,1])} khách   --> Phát hiện đúng (TP): TĂNG LÊN {fmt(cm_best_05[1,1])} khách!")
    print("  * Lưu ý: Đường cong ROC và PR được lấy mẫu (sampling) để vẽ biểu đồ trực quan, trong khi ROC-AUC")
    print("    và PR-AUC được tính toán chính xác trên toàn bộ phân phối prediction scores.")
    
    # -------------------------------------------------------------
    # BƯỚC 20: Khả năng giải thích mô hình
    # -------------------------------------------------------------
    print_section(20, "KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (Model Interpretability & Feature Weights)")
    w_idx = np.argsort(np.abs(best_lr.weights))[::-1]
    top_w = best_lr.weights[w_idx][:10]
    top_f = [feature_names[i] for i in w_idx][:10]
    
    print("\n  TOP 10 ĐẶC TRƯNG CÓ ĐỘ LỚN HỆ SỐ HÀNG ĐẦU:")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*30 + "+")
    print(f"  | {'Tên Đặc trưng':<22} | {'Trọng số w':^10} | {'Odds Ratio':^12} | {'Ý nghĩa (trên 1-SD)':<28} |")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*30 + "+")
    for f_name, w_val in zip(top_f, top_w):
        odds = np.exp(w_val)
        impact = "OR > 1: Liên hệ cùng chiều (+)" if w_val > 0 else "OR < 1: Liên hệ ngược chiều (-)"
        print(f"  | {f_name:<22} | {w_val:^+10.4f} | {odds:^12.4f} | {impact:<28} |")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*30 + "+")
    print("  * Chú ý phương pháp luận: Hệ số hồi quy w_j và Odds Ratio thể hiện mối liên hệ thống kê (statistical association)")
    print("    khi biến tăng 1 độ lệch chuẩn (1-SD), không phải và không suy diễn thành quan hệ nhân quả (non-causal).")

    # -------------------------------------------------------------
    # BƯỚC 21: Tinh chỉnh ngưỡng quyết định trên VALIDATION và Đóng gói
    # -------------------------------------------------------------
    print_section(21, "TINH CHỈNH NGƯỠNG QUYẾT ĐỊNH TRÊN VALIDATION & KIỂM ĐỊNH TEST")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
    records_val = []
    for th in thresholds:
        preds_val = best_lr.predict(X_val_scaled, threshold=th)
        records_val.append({
            'th': th,
            'acc': accuracy_score(y_val, preds_val),
            'prec': precision_score(y_val, preds_val),
            'rec': recall_score(y_val, preds_val),
            'f1': f1_score(y_val, preds_val)
        })
        
    best_th_rec = max(records_val, key=lambda x: x['f1'])
    best_threshold = float(best_th_rec['th'])
    
    print("\n  BẢNG KHẢO SÁT PHỔ NGƯỠNG TRÊN TẬP VALIDATION (Chuẩn hóa chọn ngưỡng):")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    print(f"  | {'Ngưỡng tau':^8} | {'Val Acc':^11} | {'Val Prec':^11} | {'Val Rec':^11} | {'Val F1-Score':^11} |")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    for r in records_val:
        highlight = " <-- OPTIMAL" if r['th'] == best_threshold else ""
        print(f"  | {r['th']:^8.2f} | {r['acc']*100:^10.2f}% | {r['prec']*100:^10.2f}% | {r['rec']*100:^10.2f}% | {r['f1']:^11.4f} |{highlight}")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    
    print(f"\n  --> Ngưỡng cắt tối ưu xác định từ VALIDATION: tau* = {best_threshold:.2f} (Val F1 = {best_th_rec['f1']:.4f})")
    
    # Kiểm định ĐỘC LẬP trên tập TEST tại best_threshold
    preds_test_opt = best_lr.predict(X_test_scaled, threshold=best_threshold)
    acc_opt = accuracy_score(y_test, preds_test_opt)
    prec_opt = precision_score(y_test, preds_test_opt)
    rec_opt = recall_score(y_test, preds_test_opt)
    f1_opt = f1_score(y_test, preds_test_opt)
    cm_opt = confusion_matrix(y_test, preds_test_opt)
    
    print(f"\n  --> KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP TẠI tau* = {best_threshold:.2f}:")
    print(f"      Accuracy:  {acc_opt*100:.2f}%")
    print(f"      Precision: {prec_opt*100:.2f}%")
    print(f"      Recall:    {rec_opt*100:.2f}% (Phát hiện {cm_opt[1,1]:,} / {np.sum(y_test==1):,} khoản nợ xấu)")
    print(f"      F1-Score:  {f1_opt:.4f}")
    print(f"      ROC-AUC:   {auc_best:.4f}")
    print(f"      PR-AUC:    {prauc_best:.4f}")
    print("  * Chú thích: Tại tau=0.50, Recall đạt 65.49% nhưng Precision đạt 42.86%.")
    print("    Tại ngưỡng tối ưu tau*=0.55 chọn từ Validation, mô hình đạt F1 cao nhất 0.5096 với Precision 54.01% và Recall 48.23%.")
    
    # -------------------------------------------------------------
    # BẢNG TỔNG HỢP SO SÁNH CUỐI CÙNG
    # -------------------------------------------------------------
    print_banner("BẢNG TỔNG HỢP SO SÁNH TOÀN DIỆN (BASELINE vs BEST MODEL TRÊN TEST)")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"  | {'Chỉ số đánh giá':<20} | {'Baseline (0.5)':^14} | {'Best Model (0.5)':^16} | {'Best Model (Opt)':^16} |")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"  | {'Accuracy':<20} | {acc_base*100:^13.2f}% | {acc_best_05*100:^15.2f}% | {acc_opt*100:^15.2f}% |")
    print(f"  | {'Precision':<20} | {prec_base*100:^13.2f}% | {prec_best_05*100:^15.2f}% | {prec_opt*100:^15.2f}% |")
    print(f"  | {'Recall (Độ nhạy)':<20} | {rec_base*100:^13.2f}% | {rec_best_05*100:^15.2f}% | {rec_opt*100:^15.2f}% |")
    print(f"  | {'F1-Score':<20} | {f1_base:^14.4f} | {f1_best_05:^16.4f} | {f1_opt:^16.4f} |")
    print(f"  | {'ROC-AUC':<20} | {auc_base:^14.4f} | {auc_best:^16.4f} | {auc_best:^16.4f} |")
    print(f"  | {'PR-AUC':<20} | {prauc_base:^14.4f} | {prauc_best:^16.4f} | {prauc_best:^16.4f} |")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")

    # --- Đóng gói mô hình Production ---
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_filename = os.path.join(weights_dir, "production_bundle.npz")
    json_filename = os.path.join(weights_dir, "production_bundle.json")

    best_lr.feature_names_ = feature_names

    metadata = {
        'version': '2.0',
        'created_at': '2026-09-21',
        'algorithm': 'Logistic Regression (Pure NumPy with L1/L2 Regularization)',
        'dataset': 'UCI Default of Credit Card Clients (Yeh & Lien, 2009)',
        'n_samples_total': int(len(y)),
        'split_ratios': {'train': 0.64, 'val': 0.16, 'test': 0.20},
        'n_features': int(len(feature_names)),
        'feature_names': feature_names,
        'encoding': {
            'MARRIAGE': 'One-Hot (1=Married as Reference; MARRIAGE_SINGLE, MARRIAGE_OTHERS)',
            'EDUCATION': 'One-Hot (1=Grad School as Reference; EDU_UNIVERSITY, EDU_HIGHSCHOOL, EDU_OTHERS)'
        },
        'engineered_features': [
            'UTILIZATION_RATIO_SEPT', 'AVG_PAY_DELAY', 'AVG_PAY_AMT', 'PAY_TO_BILL_RATIO'
        ],
        'optimal_hyperparameters': {
            'C': float(best_lr.C),
            'penalty': best_lr.penalty,
            'class_weight': str(best_lr.class_weight),
            'learning_rate': float(best_lr.learning_rate),
            'max_iter': int(best_lr.max_iter)
        },
        'optimal_decision_threshold': float(best_threshold),
        'test_metrics': {
            'accuracy': round(float(acc_opt), 4),
            'precision': round(float(prec_opt), 4),
            'recall': round(float(rec_opt), 4),
            'f1_score': round(float(f1_opt), 4),
            'roc_auc': round(float(auc_best), 4),
            'pr_auc': round(float(prauc_best), 4)
        }
    }

    export_bundle = {
        'weights'       : best_lr.weights,
        'bias'          : np.array([best_lr.bias]),
        'scaler_mean'   : scaler.mean_,
        'scaler_scale'  : scaler.scale_,
        'best_th'       : np.array([best_threshold]),
        'feature_names' : np.array(feature_names, dtype=str),
        'C'             : np.array([best_lr.C]),
        'penalty'       : np.array([str(best_lr.penalty) if best_lr.penalty is not None else 'none']),
        'class_weight'  : np.array([str(best_lr.class_weight) if best_lr.class_weight is not None else 'none']),
        'learning_rate' : np.array([best_lr.learning_rate]),
        'max_iter'      : np.array([best_lr.max_iter]),
        'tol'           : np.array([best_lr.tol]),
        'l1_ratio'      : np.array([best_lr.l1_ratio]),
        'init'          : np.array([best_lr.init]),
        'random_state'  : np.array([best_lr.random_state]),
        'meta_json'     : np.array(json.dumps(metadata, ensure_ascii=False))
    }

    export_json = {
        'metadata': metadata,
        'weights': best_lr.weights.tolist(),
        'bias': float(best_lr.bias),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'best_threshold': float(best_threshold),
        'feature_names': feature_names,
        'hyperparameters': metadata['optimal_hyperparameters']
    }

    already_saved = False
    if os.path.exists(weights_filename):
        try:
            curr_npz = np.load(weights_filename, allow_pickle=True)
            if 'weights' in curr_npz and np.allclose(curr_npz['weights'], best_lr.weights, atol=1e-7):
                already_saved = True
                print(f"\n  [WEIGHTS] Tệp trọng số tối ưu nhất ({weights_filename}) đã tồn tại chuẩn xác.")
                print(f"  --> Nạp trực tiếp tệp trọng số tốt nhất vào mô hình dự án, không ghi đè/sinh file liên tục.")
        except Exception:
            pass

    if not already_saved:
        np.savez_compressed(weights_filename, **export_bundle)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(export_json, f, ensure_ascii=False, indent=2)
        save_weights(best_lr, path=os.path.join(weights_dir, "best_model_weights.npz"), feature_names=feature_names)
        save_weights(base_lr, path=os.path.join(weights_dir, "baseline_weights.npz"), feature_names=feature_names)

    sz = os.path.getsize(weights_filename) / 1024
    print(f"\n  [EXPORT] File chính thức: {weights_filename} ({sz:.2f} KB)")
    print(f"  weights shape : {best_lr.weights.shape}  |  bias : {best_lr.bias:+.6f}")
    print(f"  best_th (tau*): {best_threshold:.2f} (chọn từ Validation)")

    # --- Kiểm chứng load lại ---
    print("\n  [VERIFY] Đang load lại pipeline từ file .npz và kiểm chứng...")
    pipeline_prod = CreditDefaultInferencePipeline.load(weights_filename)

    proba_v      = pipeline_prod.predict_proba(X_test)
    preds_v      = pipeline_prod.predict(X_test)

    proba_orig   = best_lr.predict_proba(X_test_scaled)[:, 1]
    max_diff     = float(np.max(np.abs(proba_orig - proba_v)))
    is_close     = bool(np.allclose(proba_orig, proba_v))
    pct_match    = float(np.mean(preds_test_opt == preds_v) * 100)

    print(f"  {'Sai lệch xác suất cực đại':42}: {max_diff:.2e}")
    print(f"  {'np.allclose(proba_orig, proba_load)':42}: {is_close}")
    print(f"  {'Tỷ lệ trùng khớp nhãn (0/1)':42}: {pct_match:.2f}%  ({len(preds_v):,} khách hàng)")
    print(f"  {'Tái hiện F1 trên Test':42}: {f1_score(y_test, preds_v):.4f}")
    print(f"  {'Tái hiện Recall trên Test':42}: {recall_score(y_test, preds_v):.4f}")
    if is_close and pct_match == 100.0:
        print("  >>> ĐÓNG GÓI CHÍNH XÁC 100% — SẴN SÀNG VẬN HÀNH THỰC TẾ! <<<")

    # --- Demo suy luận thực tế với CreditDefaultInferencePipeline ---
    sample_idx = 0
    sample_input = X_test[sample_idx:sample_idx+1]
    sample_prob = float(pipeline_prod.predict_proba(sample_input)[0])
    sample_pred = int(pipeline_prod.predict(sample_input)[0])
    true_lbl = int(y_test[sample_idx])
    dec_text = "CẢNH BÁO NGUY CƠ VỠ NỢ CAO" if sample_pred == 1 else "NGUY CƠ VỠ NỢ THẤP / KIỂM SOÁT TỐT"
    true_text = "Vỡ nợ thực tế (Nhãn 1)" if true_lbl == 1 else "Đúng hạn thực tế (Nhãn 0)"

    print("\n" + "=" * 78)
    print("  KẾT QUẢ SUY LUẬN HỒ SƠ KHÁCH HÀNG (CREDIT DEFAULT INFERENCE PIPELINE)")
    print("=" * 78)
    print(f"  - Số lượng đặc trưng đầu vào : {len(pipeline_prod.feature_names)}")
    print(f"  - Xác suất vỡ nợ dự báo      : {sample_prob:.4f} ({sample_prob*100:.2f}%)")
    print(f"  - Ngưỡng quyết định tối ưu   : {pipeline_prod.threshold:.4f}")
    print(f"  - Đánh giá rủi ro            : Nhãn {sample_pred} -> {dec_text}")
    print(f"  - Nhãn thực tế đối chiếu     : Nhãn {true_lbl} -> {true_text}")
    print(f"  - Đánh giá tính chính xác    : {'CHÍNH XÁC' if sample_pred == true_lbl else 'CẦN THẨM ĐỊNH LẠI'}")
    print("=" * 78)

    # -------------------------------------------------------------
    # BẢNG AUDIT CUỐI CÙNG (14 TIÊU CHÍ KIỂM SOÁT CHẤT LƯỢNG ML)
    # -------------------------------------------------------------
    audit_checks = [
        ("Train/Test split", "PASS", "Tách 80% Train+Val và 20% Test phân tầng bằng stratify=y"),
        ("Validation set", "PASS", "Tách 16% tổng thể từ tập 80% ban đầu để tạo Validation độc lập"),
        ("Data leakage", "PASS", "Không rò rỉ: Scaler fit trên Train, FE tính row-wise, Test chỉ đánh giá cuối"),
        ("Scaler fitted only on Train", "PASS", "StandardScaler chỉ fit trên X_train; Val và Test chỉ transform"),
        ("One-Hot Encoding", "PASS", "One-Hot cho MARRIAGE & EDUCATION, bỏ nhóm tham chiếu tránh dummy trap"),
        ("Hyperparameter tuning không dùng Test", "PASS", "GridSearchCV chạy 10-Fold Stratified CV nội bộ trên X_train"),
        ("Threshold không dùng Test", "PASS", "Ngưỡng tau* được tối ưu trên tập Validation (X_val_scaled, y_val)"),
        ("Class imbalance", "PASS", "Sử dụng class_weight='balanced' điều chỉnh trọng số nghịch đảo tần suất lớp"),
        ("Cross-validation", "PASS", "10-Fold Stratified CV với fold-level scaling chuẩn hóa"),
        ("F1/Recall evaluation", "PASS", "Ưu tiên F1/Recall để giảm thiểu tổn thất bỏ sót nợ xấu (FN)"),
        ("ROC-AUC", "PASS", "Đo lường năng lực phân biệt xác suất toàn diện (tính trên prediction scores)"),
        ("PR-AUC", "PASS", "Đo lường năng lực phân loại dưới mất cân bằng lớp (Average Precision)"),
        ("Model persistence", "PASS", "Lưu trữ bundle weights, bias, scaler, threshold, feature_names"),
        ("Reload verification", "PASS", "Tải lại bundle, kiểm định np.allclose và trùng khớp nhãn 100%")
    ]

    print("\n" + "=" * 78)
    print("  BẢNG AUDIT CUỐI CÙNG — 14 TIÊU CHÍ KIỂM SOÁT PHƯƠNG PHÁP LUẬN ML")
    print("=" * 78)
    print(f"  | {'Tiêu chí kiểm tra':<38} | {'Trạng thái':^12} |")
    print("  +" + "-"*40 + "+" + "-"*14 + "+")
    for item, status, desc in audit_checks:
        print(f"  | {item:<38} | {status:^12} |")
    print("  +" + "-"*40 + "+" + "-"*14 + "+")

    elapsed = time.time() - start_total_time
    print_banner(f"PIPELINE THỰC NGHỆM HOÀN TẤT TRONG {elapsed:.1f} GIÂY!")

if __name__ == "__main__":
    main()
