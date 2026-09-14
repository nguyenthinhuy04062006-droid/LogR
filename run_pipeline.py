#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py
Script thực thi toàn bộ quy trình học máy và in báo cáo chi tiết 21 bước ra Terminal.
Sử dụng 100% code thuần NumPy/Pandas từ model.py (No Scikit-learn).
"""

import sys
import time
import os
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
    LogisticRegression, StratifiedKFold, GridSearchCV
)

def print_banner(text, width=76):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_section(step_num, title, width=76):
    print("\n" + "-" * width)
    print(f"  [BƯỚC {step_num:02d}] {title}")
    print("-" * width)

def main():
    start_total_time = time.time()
    
    print_banner("HỆ THỐNG DỰ BÁO XÁC SUẤT VỠ NỢ THẺ TÍN DỤNG — LOGISTIC REGRESSION THUẦN")
    
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
    print("  - Đặc tính: Tổn thất phi đối xứng (Asymmetric Loss) — Bỏ lọt nợ xấu (False Negative)")
    print("    gây thiệt hại tài chính trực tiếp lớn hơn nhiều so với từ chối nhầm (False Positive).")
    
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
    print(f"  - Đã chuẩn hóa: {edu_anom} mẫu EDUCATION dị biệt -> nhóm 4; {mar_anom} mẫu MARRIAGE dị biệt -> nhóm 3.")
    
    # -------------------------------------------------------------
    # BƯỚC 5 & 6: Chuẩn hóa & Biến phân loại
    # -------------------------------------------------------------
    print_section(5, "CHUẨN HÓA ĐẶC TRƯNG (Feature Scaling Rationale)")
    print("  - Giải pháp: Z-Score Standardization z = (x - mu) / sigma.")
    print("  - Mục đích: Cải thiện số điều kiện Hessian kappa(H), khắc phục dao động zig-zag của Gradient.")
    
    print_section(6, "XỬ LÝ BIẾN PHÂN LOẠI (Categorical Processing)")
    print("  - Duy trì các thang đo thứ bậc cho PAY_i ([-2, 8]), mã hóa số nguyên cho nhân khẩu học.")
    
    # -------------------------------------------------------------
    # BƯỚC 7 & 8: Thuật toán & Kỹ thuật tạo đặc trưng
    # -------------------------------------------------------------
    print_section(7, "LỰA CHỌN THUẬT TOÁN VÀ HÀM MẤT MÁT (Algorithm & Loss)")
    print("  - Hàm mất mát: Weighted Binary Cross-Entropy kết hợp hàm phạt L1 (Lasso) / L2 (Ridge).")
    
    print_section(8, "KỸ THUẬT TẠO ĐẶC TRƯNG (Feature Engineering)")
    eps = 1e-6
    X_df = df.drop(columns=['ID', 'default payment next month']).copy()
    y_df = df['default payment next month']
    
    # Tạo 4 đặc trưng miền tài chính mới
    X_df['UTILIZATION_RATIO'] = X_df['BILL_AMT1'] / (X_df['LIMIT_BAL'] + eps)
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    X_df['PAY_TREND'] = X_df[pay_cols].mean(axis=1)
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X_df['AVG_PAY_AMT'] = X_df[pay_amt_cols].mean(axis=1)
    bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    avg_pay = X_df[pay_amt_cols].mean(axis=1)
    avg_bill = X_df[bill_amt_cols].mean(axis=1)
    X_df['PAY_TO_BILL_RATIO'] = avg_pay / (avg_bill + eps)
    
    feature_names = X_df.columns.tolist()
    n_features = len(feature_names)
    print(f"  - Đã bổ sung 4 biến tài chính: UTILIZATION_RATIO, PAY_TREND, AVG_PAY_AMT, PAY_TO_BILL_RATIO.")
    print(f"  - Tổng số đặc trưng nâng từ 23 lên {n_features} biến.")
    
    X = X_df.values
    y = y_df.values
    
    # -------------------------------------------------------------
    # BƯỚC 9 & 10: Chia dữ liệu & Kiểm soát Leakage
    # -------------------------------------------------------------
    print_section(9, "CHIA DỮ LIỆU & KIỂM SOÁT RÒ RỈ (Data Splitting & Zero-Leakage)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    print(f"  - Phân tầng 80/20: Train = {n_train:,} dòng | Test = {n_test:,} dòng.")
    print("  - Fit StandardScaler chỉ trên Train; biến đổi Test độc lập để bảo đảm Zero-Leakage.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print_section(10, "LỰA CHỌN PHƯƠNG PHÁP CHIA DỮ LIỆU (Stratified Hold-out)")
    print(f"  - Tỷ lệ nhãn 1 trên Train: {(y_train==1).sum()/n_train*100:.2f}% | Test: {(y_test==1).sum()/n_test*100:.2f}% (Bảo toàn hoàn hảo).")
    
    # -------------------------------------------------------------
    # BƯỚC 11, 12, 13: Mô hình cơ sở & Phân tích Loss/Overfitting
    # -------------------------------------------------------------
    print_section(11, "XÂY DỰNG MÔ HÌNH CƠ SỞ (Baseline Model)")
    base_lr = LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l2', C=1.0)
    base_lr.fit(X_train_scaled, y_train)
    y_pred_base = base_lr.predict(X_test_scaled)
    y_proba_base = base_lr.predict_proba(X_test_scaled)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base)
    rec_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_proba_base)
    prauc_base = average_precision_score(y_test, y_proba_base)
    
    print(f"  - Baseline (C=1.0, không class_weight):")
    print(f"    Accuracy: {acc_base*100:.2f}% | Precision: {prec_base*100:.2f}% | Recall: {rec_base*100:.2f}% | F1: {f1_base:.4f} | ROC-AUC: {auc_base:.4f}")
    print(f"    --> Nhận xét: Accuracy cao ({acc_base*100:.2f}%) là ảo do bỏ sót nợ xấu (Recall chỉ đạt {rec_base*100:.2f}%).")
    
    print_section(12, "NGUYÊN LÝ NO FREE LUNCH (No Free Lunch Theorem)")
    print("  - Không có siêu tham số hay hàm điều chuẩn nào tối ưu cho mọi bài toán; cần thực nghiệm hệ thống.")
    
    print_section(13, "PHÂN TÍCH BIAS - VARIANCE & LOSS DECOMPOSITION")
    models_check = [
        ("Không phạt (No Reg)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty=None)),
        ("Phạt L2 Ridge (C=10.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l2', C=10.0)),
        ("Phạt L1 Lasso (C=50.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l1', C=50.0)),
        ("Phạt L1 Lasso (C=10.0)", LogisticRegression(learning_rate=0.1, max_iter=1000, penalty='l1', C=10.0))
    ]
    
    print("\n  BẢNG BÓC TÁCH HÀM MẤT MÁT & OVERFIT GAP:")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    print(f"  | {'Chiến lược Điều chuẩn':<24} | {'Train BCE':^9} | {'Test BCE':^9} | {'Overfit Gap':^11} | {'Penalty Loss':^11} | {'||w||_1':^9} | {'Zero W':^9} |")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    
    for name, mdl in models_check:
        mdl.fit(X_train_scaled, y_train)
        tr_res = mdl.compute_loss(X_train_scaled, y_train)
        te_res = mdl.compute_loss(X_test_scaled, y_test)
        gap = te_res['bce_loss'] - tr_res['bce_loss']
        w_L1 = np.sum(np.abs(mdl.weights))
        zeros = int(np.sum(np.abs(mdl.weights) < 1e-4))
        print(f"  | {name:<24} | {tr_res['bce_loss']:^9.4f} | {te_res['bce_loss']:^9.4f} | {gap:^+11.4f} | {tr_res['penalty_loss']:^11.4f} | {w_L1:^9.4f} | {f'{zeros}/{n_features}':^9} |")
    print("  +" + "-"*26 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*11 + "+" + "-"*11 + "+")
    
    # -------------------------------------------------------------
    # BƯỚC 14, 15, 16, 17: Metrics, K-Fold, GridSearch & Training
    # -------------------------------------------------------------
    print_section(14, "LỰA CHỌN EVALUATION METRICS")
    print("  - Thang đo ưu tiên: F1-Score (Trung hòa hài hòa giữa Precision & Recall), ROC-AUC và PR-AUC.")
    
    print_section(15, "10-FOLD STRATIFIED CROSS-VALIDATION")
    print_section(16, "TỐI ƯU SIÊU THAM SỐ QUA GRID SEARCH (GridSearchCV)")
    
    param_grid = {
        'C': [1.0, 5.0, 10.0, 50.0],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced']
    }
    lr_grid = LogisticRegression(learning_rate=0.1, max_iter=1000)
    cv_obj = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=lr_grid, param_grid=param_grid, cv=cv_obj, scoring='f1', scaler=StandardScaler())
    
    print("  Đang chạy 10-Fold Grid Search trên 8 tổ hợp siêu tham số...")
    grid_search.fit(X_train, y_train)
    
    print("\n  BẢNG KẾT QUẢ GRID SEARCH:")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    print(f"  | {'C':^6} | {'Penalty':^8} | {'Class Weight':^13} | {'Mean F1 (CV)':^11} |")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    for row in grid_search.cv_results_table_:
        print(f"  | {row['C']:^6.1f} | {row['penalty'].upper():^8} | {row['class_weight']:^13} | {row['mean_f1']:^11.4f} |")
    print("  +" + "-"*8 + "+" + "-"*10 + "+" + "-"*15 + "+" + "-"*13 + "+")
    
    best_params = grid_search.best_params_
    best_lr = grid_search.best_estimator_
    print(f"\n  --> Cấu hình tối ưu nhất: {best_params} | Best CV F1 = {grid_search.best_score_:.4f}")
    
    print_section(17, "ĐÁNH GIÁ MÔ HÌNH TỐI ƯU TRÊN TẬP TEST ĐỘC LẬP")
    # best_lr (GridSearchCV.best_estimator_) có scaler_ nội bộ → nhận X gốc, KHÔNG scale trước
    y_pred_best = best_lr.predict(X_test)
    y_proba_best = best_lr.predict_proba(X_test)[:, 1]
    
    acc_best = accuracy_score(y_test, y_pred_best)
    prec_best = precision_score(y_test, y_pred_best)
    rec_best = recall_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)
    auc_best = roc_auc_score(y_test, y_proba_best)
    prauc_best = average_precision_score(y_test, y_proba_best)
    
    # -------------------------------------------------------------
    # BƯỚC 18 & 19: Kiểm định độ tin cậy & Phân tích lỗi (Confusion Matrix)
    # -------------------------------------------------------------
    print_section(18, "KIỂM ĐỊNH THỐNG KÊ ĐỘ TIN CẬY (Confidence Intervals)")
    best_scores = grid_search.best_scores_
    mean_cv = np.mean(best_scores)
    std_cv = np.std(best_scores)
    ci95 = 1.96 * (std_cv / np.sqrt(len(best_scores)))
    print(f"  - 10-Fold CV F1: {mean_cv:.4f} +/- {std_cv:.4f} | 95% CI: [{mean_cv - ci95:.4f}, {mean_cv + ci95:.4f}]")
    
    print_section(19, "PHÂN TÍCH LỖI (Error Analysis & Confusion Matrix)")
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_best = confusion_matrix(y_test, y_pred_best)
    
    def fmt(n): return f"{n:,}"
    print("\n  MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print(f"  Baseline Model:              Mô hình tối ưu ({best_params['penalty'].upper()}, C={best_params['C']}):")
    print(f"    TN: {fmt(cm_base[0,0]):<8} | FP: {fmt(cm_base[0,1]):<8}   TN: {fmt(cm_best[0,0]):<8} | FP: {fmt(cm_best[0,1]):<8}")
    print(f"    FN: {fmt(cm_base[1,0]):<8} | TP: {fmt(cm_base[1,1]):<8}   FN: {fmt(cm_best[1,0]):<8} | TP: {fmt(cm_best[1,1]):<8}")
    print(f"    --> Bỏ sót nợ xấu (FN): {fmt(cm_base[1,0])} khách   --> Bỏ sót nợ xấu (FN): GIẢM XUỐNG CÒN {fmt(cm_best[1,0])} khách!")
    print(f"    --> Phát hiện đúng (TP): {fmt(cm_base[1,1])} khách   --> Phát hiện đúng (TP): TĂNG LÊN {fmt(cm_best[1,1])} khách!")
    
    # -------------------------------------------------------------
    # BƯỚC 20: Khả năng giải thích mô hình
    # -------------------------------------------------------------
    print_section(20, "KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (Model Interpretability & Feature Weights)")
    w_idx = np.argsort(np.abs(best_lr.weights))[::-1]
    top_w = best_lr.weights[w_idx][:10]
    top_f = [feature_names[i] for i in w_idx][:10]
    
    print("\n  TOP 10 ĐẶC TRƯNG CHI PHỐI QUYẾT ĐỊNH DỰ BÁO:")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*20 + "+")
    print(f"  | {'Tên Đặc trưng':<22} | {'Trọng số w':^10} | {'Odds Ratio':^12} | {'Tác động nghiệp vụ':<18} |")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*20 + "+")
    for f_name, w_val in zip(top_f, top_w):
        odds = np.exp(w_val)
        impact = "Tăng rủi ro (+)" if w_val > 0 else "Bảo vệ/Giảm rủi ro (-)"
        print(f"  | {f_name:<22} | {w_val:^+10.4f} | {odds:^12.4f} | {impact:<18} |")
    print("  +" + "-"*24 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*20 + "+")

    # -------------------------------------------------------------
    # BƯỚC 21: Tinh chỉnh ngưỡng quyết định (Threshold Tuning)
    # -------------------------------------------------------------
    print_section(21, "TINH CHỈNH NGƯỠNG QUYẾT ĐỊNH (Decision Threshold Tuning)")
    thresholds = [0.30, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70]
    records = []
    for th in thresholds:
        # best_lr tự scale bên trong → dùng X_train gốc
        preds_train = best_lr.predict(X_train, threshold=th)
        records.append({
            'th': th,
            'acc': accuracy_score(y_train, preds_train),
            'prec': precision_score(y_train, preds_train),
            'rec': recall_score(y_train, preds_train),
            'f1': f1_score(y_train, preds_train)
        })
        
    best_th_rec = max(records, key=lambda x: x['f1'])
    preds_test_th = best_lr.predict(X_test, threshold=best_th_rec['th'])
    
    print("\n  BẢNG KHẢO SÁT PHỔ NGƯỠNG TRÊN TẬP TRAIN (Chống rò rỉ dữ liệu):")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    print(f"  | {'Ngưỡng tau':^8} | {'Accuracy':^11} | {'Precision':^11} | {'Recall':^11} | {'F1-Score':^11} |")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    for r in records:
        highlight = " <-- BEST" if r['th'] == best_th_rec['th'] else ""
        print(f"  | {r['th']:^8.2f} | {r['acc']*100:^10.2f}% | {r['prec']*100:^10.2f}% | {r['rec']*100:^10.2f}% | {r['f1']:^11.4f} |{highlight}")
    print("  +" + "-"*10 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+")
    
    print(f"\n  --> Ngưỡng cắt tối ưu nhất theo tiêu chí F1: tau = {best_th_rec['th']:.2f}")
    print(f"  --> Kiểm định độc lập trên tập Test tại tau = {best_th_rec['th']:.2f}:")
    print(f"      Accuracy:  {accuracy_score(y_test, preds_test_th)*100:.2f}%")
    print(f"      Precision: {precision_score(y_test, preds_test_th)*100:.2f}%")
    print(f"      Recall:    {recall_score(y_test, preds_test_th)*100:.2f}% (Phát hiện {np.sum((y_test==1)&(preds_test_th==1)):,} / {np.sum(y_test==1):,} khoản nợ xấu)")
    print(f"      F1-Score:  {f1_score(y_test, preds_test_th):.4f}")
    
    # -------------------------------------------------------------
    # BẢNG TỔNG HỢP SO SÁNH CUỐI CÙNG
    # -------------------------------------------------------------
    print_banner("BẢNG TỔNG HỢP SO SÁNH TOÀN DIỆN (BASELINE vs BEST MODEL)")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"  | {'Chỉ số đánh giá':<20} | {'Baseline Model':^14} | {'Best Model (0.5)':^16} | {'Best Model (Opt)':^16} |")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"  | {'Accuracy':<20} | {acc_base*100:^13.2f}% | {acc_best*100:^15.2f}% | {accuracy_score(y_test, preds_test_th)*100:^15.2f}% |")
    print(f"  | {'Precision':<20} | {prec_base*100:^13.2f}% | {prec_best*100:^15.2f}% | {precision_score(y_test, preds_test_th)*100:^15.2f}% |")
    print(f"  | {'Recall (Độ nhạy)':<20} | {rec_base*100:^13.2f}% | {rec_best*100:^15.2f}% | {recall_score(y_test, preds_test_th)*100:^15.2f}% |")
    print(f"  | {'F1-Score':<20} | {f1_base:^14.4f} | {f1_best:^16.4f} | {f1_score(y_test, preds_test_th):^16.4f} |")
    print(f"  | {'ROC-AUC':<20} | {auc_base:^14.4f} | {auc_best:^16.4f} | {auc_best:^16.4f} |")
    print(f"  | {'PR-AUC':<20} | {prauc_base:^14.4f} | {prauc_best:^16.4f} | {prauc_best:^16.4f} |")
    print("  +" + "-"*22 + "+" + "-"*16 + "+" + "-"*18 + "+" + "-"*18 + "+")
    
    elapsed = time.time() - start_total_time
    print_banner(f"PIPELINE THỰC NGHỆM HOÀN TẤT TRONG {elapsed:.1f} GIÂY!")

if __name__ == "__main__":
    main()
