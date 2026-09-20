#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
Unit tests and pipeline integrity tests for Credit Card Default Logistic Regression.
Ensures zero-dependency operation, numerical correctness, and serialization consistency.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd

from model import (
    StandardScaler, train_test_split,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    LogisticRegression, StratifiedKFold
)
from weights import save_weights, load_weights


class TestDataAndPreprocessing:
    def test_data_loading(self):
        csv_path = 'default_of_credit_card_clients.csv'
        assert os.path.exists(csv_path), f"Không tìm thấy file dữ liệu: {csv_path}"
        df = pd.read_csv(csv_path)
        assert df.shape[0] == 30000, f"Kỳ vọng 30,000 dòng, nhận {df.shape[0]}"
        assert 'default payment next month' in df.columns

    def test_standard_scaler(self):
        np.random.seed(42)
        X = np.random.randn(1000, 5) * 20.0 + 50.0
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-7)
        assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-7)
        
        # Test transform on unseen data
        X_new = np.random.randn(10, 5) * 20.0 + 50.0
        X_new_scaled = scaler.transform(X_new)
        assert X_new_scaled.shape == (10, 5)

    def test_stratified_split(self):
        y = np.array([0] * 780 + [1] * 220)
        X = np.random.randn(1000, 4)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        prop_tr = np.mean(y_tr == 1)
        prop_te = np.mean(y_te == 1)
        assert abs(prop_tr - 0.22) < 0.01
        assert abs(prop_te - 0.22) < 0.01

    def test_train_val_test_split_proportions(self):
        y = np.array([0] * 7800 + [1] * 2200) # 10,000 mẫu
        X = np.random.randn(10000, 5)
        X_temp, X_te, y_temp, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)
        assert len(X_tr) == 6400
        assert len(X_va) == 1600
        assert len(X_te) == 2000
        for split_y in [y_tr, y_va, y_te]:
            assert abs(np.mean(split_y == 1) - 0.22) < 0.01


class TestLogisticRegressionModel:
    def test_fit_and_predict(self):
        np.random.seed(42)
        X = np.random.randn(500, 5)
        # linear combination with noise
        logits = X[:, 0] * 1.5 - X[:, 1] * 2.0 + 0.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs >= 0.5).astype(int)
        
        model = LogisticRegression(learning_rate=0.2, max_iter=200, penalty='l2', C=10.0, random_state=42)
        model.fit(X, y)
        
        preds_proba = model.predict_proba(X)[:, 1]
        assert np.all((preds_proba >= 0.0) & (preds_proba <= 1.0))
        
        preds = model.predict(X, threshold=0.5)
        acc = accuracy_score(y, preds)
        assert acc > 0.80, f"Độ chính xác {acc:.2f} thấp hơn kỳ vọng 0.80"

    def test_regularization_l1_sparsity(self):
        np.random.seed(42)
        X = np.random.randn(500, 8)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 > 0).astype(int)
        
        # Mô hình không điều chuẩn
        m_none = LogisticRegression(learning_rate=0.1, max_iter=200, penalty=None, random_state=42)
        m_none.fit(X, y)
        
        # Mô hình có điều chuẩn L1
        m_l1 = LogisticRegression(learning_rate=0.1, max_iter=200, penalty='l1', C=1.0, random_state=42)
        m_l1.fit(X, y)
        
        l1_norm_none = np.sum(np.abs(m_none.weights))
        l1_norm_l1 = np.sum(np.abs(m_l1.weights))
        
        assert l1_norm_l1 < l1_norm_none, f"Kỳ vọng L1 norm ({l1_norm_l1:.3f}) nhỏ hơn không điều chuẩn ({l1_norm_none:.3f})"
        assert len(m_l1.loss_history_['total']) > 0


class TestProductionBundleAndPipeline:
    def test_production_bundle_integrity(self):
        bundle_path = os.path.join("weights", "production_bundle.npz")
        assert os.path.exists(bundle_path), f"Không tìm thấy artifact: {bundle_path}"
        
        data = np.load(bundle_path, allow_pickle=True)
        required_keys = ['weights', 'bias', 'scaler_mean', 'scaler_scale', 'best_th', 'feature_names']
        for k in required_keys:
            assert k in data, f"Thiếu key '{k}' trong production_bundle.npz"
            
        assert len(data['weights']) == len(data['feature_names'])
        assert len(data['feature_names']) == 27, f"Kỳ vọng 27 đặc trưng sau FE, nhận {len(data['feature_names'])}"
        assert 0.30 <= float(data['best_th'][0]) <= 0.70

    def test_pipeline_inference(self):
        from model import CreditDefaultInferencePipeline
        bundle_path = os.path.join("weights", "production_bundle.npz")
        data = np.load(bundle_path, allow_pickle=True)
        
        scaler = StandardScaler()
        scaler.mean_ = data['scaler_mean']
        scaler.scale_ = data['scaler_scale']
        
        model = LogisticRegression(
            C=float(data['C'][0]),
            penalty=str(data['penalty'][0]) if str(data['penalty'][0]) != 'none' else None,
            class_weight=str(data['class_weight'][0]) if str(data['class_weight'][0]) != 'none' else None
        )
        model.weights = data['weights']
        model.bias = float(data['bias'][0])
        best_th = float(data['best_th'][0])
        feat_names = list(data['feature_names'])
        
        pipeline = CreditDefaultInferencePipeline(
            model=model, scaler=scaler, threshold=best_th, feature_names=feat_names
        )
        
        # Test input of 27 features
        dummy_input = np.zeros((1, 27))
        prob = pipeline.predict_proba(dummy_input)[0]
        assert 0.0 <= prob <= 1.0
        
        pred = pipeline.predict(dummy_input)[0]
        assert pred in [0, 1]


if __name__ == '__main__':
    print("=" * 60)
    print("  ĐANG CHẠY KIỂM THỬ HỆ THỐNG (TEST PIPELINE)...")
    print("=" * 60)
    
    t1 = TestDataAndPreprocessing()
    t1.test_data_loading()
    print("  [PASS] test_data_loading")
    t1.test_standard_scaler()
    print("  [PASS] test_standard_scaler")
    t1.test_stratified_split()
    print("  [PASS] test_stratified_split")
    t1.test_train_val_test_split_proportions()
    print("  [PASS] test_train_val_test_split_proportions")
    
    t2 = TestLogisticRegressionModel()
    t2.test_fit_and_predict()
    print("  [PASS] test_fit_and_predict")
    t2.test_regularization_l1_sparsity()
    print("  [PASS] test_regularization_l1_sparsity")
    
    t3 = TestProductionBundleAndPipeline()
    t3.test_production_bundle_integrity()
    print("  [PASS] test_production_bundle_integrity")
    t3.test_pipeline_inference()
    print("  [PASS] test_pipeline_inference")
    
    print("=" * 60)
    print("  TẤT CẢ 8/8 BÀI TEST ĐÃ VƯỢT QUA THÀNH CÔNG (ALL PASSED)!")
    print("=" * 60)
