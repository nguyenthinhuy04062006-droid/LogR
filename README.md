# DỰ ÁN MÁY HỌC: DỰ ĐOÁN NỢ XẤU THẺ TÍN DỤNG (CREDIT CARD DEFAULT PREDICTION)
## THUẬT TOÁN: HỒI QUY LOGISTIC (LOGISTIC REGRESSION) - CODE THUẦN (NO SCIKIT-LEARN)

---

### 1. Giới thiệu dự án
Dự án giải quyết bài toán phân loại nhị phân: **Dự đoán khả năng khách hàng vỡ nợ thẻ tín dụng trong tháng tiếp theo** dựa trên bộ dữ liệu nổi tiếng *Default of Credit Card Clients Dataset* (Yeh & Lien, 2009, UCI Machine Learning Repository).

**Điểm đặc biệt cốt lõi:**
- **Code thuần 100% (No Scikit-learn):** Toàn bộ thuật toán Logistic Regression, Gradient Descent, Điều chuẩn L1/L2, StandardScaler, StratifiedKFold, cross_validate, GridSearchCV và các hàm đánh giá (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC, PR-AUC) đều được tự viết từ đầu bằng Python/NumPy.
- **Tối thiểu hóa độ phức tạp chống Overfitting (Structural Risk Minimization):**
  - **Phạt L2 (Ridge):** Co nhỏ đều các trọng số (Weight Decay), hạn chế bùng nổ trọng số, kiểm soát phương sai.
  - **Phạt L1 (Lasso):** Ứng dụng **Proximal Gradient Descent (Soft-Thresholding Operator)** ép trực tiếp các trọng số nhiễu về đúng $0.0$, tạo tính thưa (Sparsity) và tự động chọn lọc các biến cốt lõi.
- **Bóc tách và Kiểm tra Mất mát chi tiết (Loss Decomposition & Checking):**
  - Phương thức `compute_loss(X, y)` tách bạch: Mất mát dữ liệu (Binary Cross-Entropy Loss), Mất mát phạt độ phức tạp (Penalty Loss), và Tổng hàm mất mát $\mathcal{J}$.
  - Đo lường khoảng cách **Overfit Gap** ($\Delta = \text{Test BCE} - \text{Train BCE}$) để đánh giá độ tổng quát hóa.
  - Theo dõi đường cong hội tụ mất mát `loss_history_` qua các vòng lặp.
- **Kiến trúc Zero Data Leakage:** Chuẩn hóa độc lập bên trong từng fold của Cross-Validation và GridSearch.
- **Tối ưu hóa bài toán mất cân bằng dữ liệu (Imbalanced Data ~3.5:1):** Ứng dụng `class_weight='balanced'` và khảo sát ngưỡng quyết định (Threshold Tuning) giúp tăng độ nhạy phát hiện nợ xấu (Recall) từ **1.28% lên 61.67%**.

---

### 2. Cấu trúc thư mục dự án

```text
d:\logistics  regression\
│
├── tests/
│   └── test_pipeline.py                        # Bộ kiểm thử tự động toàn bộ quy trình (7/7 tests pass)
│
├── .gitignore                                  # Cấu hình bỏ qua file tạm, cache
├── README.md                                   # Hướng dẫn chi tiết dự án
├── model.py                                    # Module chứa toàn bộ class và hàm ML code thuần
├── requirements.txt                            # Danh mục thư viện phụ thuộc tối thiểu
├── run_pipeline.py                             # Script thực thi toàn bộ pipeline báo cáo 21 bước trên CLI
├── default_of_credit_card_clients.csv          # Tập dữ liệu gốc (30,000 dòng, 25 cột)
├── credit_card_default_logistic_regression.ipynb # Jupyter Notebook chuẩn học thuật 21 bước hoàn chỉnh
├── weights.py                                  # Module lưu trữ & nạp trọng số (.npz, .json, .txt)
└── weights/                                    # Thư mục lưu trữ artifact mô hình
    ├── baseline_weights.{npz,json,txt}         # Trọng số mô hình cơ sở Baseline
    ├── best_model_weights.{npz,json,txt}       # Trọng số mô hình tối ưu Best Model
    ├── production_bundle.npz                   # Gói Production nhị phân (Scaler + Model + Ngưỡng)
    └── production_bundle.json                  # Gói Production JSON có nhãn rõ ràng
```

---

### 3. Hướng dẫn chạy dự án

#### Chạy pipeline đầy đủ qua Terminal
Mở terminal tại thư mục dự án và chạy:
```powershell
python run_pipeline.py
```

**Pipeline in ra 21 bước chuẩn học thuật:**
1. Xác định bài toán (Problem Definition)
2. Xác định bản chất của bài toán ML
3. Khảo sát lĩnh vực và không gian dữ liệu (Domain & Data Understanding)
4. Khám phá và xử lý dữ liệu (Data Exploration & Cleaning)
5. Chuẩn hóa đặc trưng (Feature Scaling)
6. Xử lý biến phân loại (Categorical Data & Encoding)
7. Lựa chọn thuật toán và hàm mất mát (Algorithm & Loss Function Selection)
8. Kỹ thuật tạo đặc trưng (Feature Engineering)
9. Chia dữ liệu và kiểm soát Data Leakage (Data Splitting & Leakage Prevention)
10. Lựa chọn phương pháp chia dữ liệu (Data Splitting Strategy)
11. Xây dựng mô hình cơ sở (Baseline Model)
12. Nguyên lý No Free Lunch (No Free Lunch Theorem)
13. Phân tích Bias và Variance (Bias–Variance Analysis)
14. Lựa chọn và đánh giá bằng Evaluation Metrics (Evaluation Metrics Selection)
15. Kiểm định chéo phân tầng K-Fold (10-Fold Stratified Cross-Validation)
16. Tối ưu siêu tham số (Hyperparameter Tuning via GridSearchCV)
17. Thực nghiệm huấn luyện mô hình (Model Training & Experimentation)
18. Kiểm định thống kê độ tin cậy (Statistical Significance & Confidence)
19. Phân tích lỗi (Error Analysis)
20. Khả năng giải thích mô hình (Model Interpretability)
21. Tinh chỉnh ngưỡng quyết định và Đóng gói mô hình suy luận (Threshold Tuning & Packaging)

#### Chạy kiểm thử hệ thống (Unit Tests)
```powershell
python tests/test_pipeline.py
```

---

### 4. Bảng tổng hợp thực nghiệm & Check mất mát phạt L1, L2

#### So sánh cơ chế phạt và mất mát chống Overfitting:
| Chiến lược Điều chuẩn | Train BCE | Test BCE | Overfit Gap | Penalty Loss | Tổng Loss | $||\mathbf{w}||_1$ | Trọng số Zero | Test ROC AUC | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không phạt (No Reg)** | 0.6095 | 0.6223 | +0.0129 | 0.0000 | 0.6095 | 2.0939 | 0/23 | 0.7077 | 0.4617 |
| **Phạt L2 (Ridge $C=10.0$)** | 0.6145 | 0.6244 | +0.0098 | 0.0104 | 0.6250 | 1.4096 | 0/23 | 0.7027 | 0.4573 |
| **Phạt L1 (Lasso $C=50.0$)** | 0.6172 | 0.6284 | +0.0112 | 0.0180 | 0.6352 | 0.8997 | **16/23** | 0.6997 | 0.4816 |
| **Phạt L1 (Lasso $C=10.0$)** | 0.6455 | 0.6489 | **+0.0034** | 0.0323 | 0.6778 | 0.3229 | **22/23** | 0.6759 | **0.4977** |

#### So sánh Baseline vs Best Tuned Model ($C=50$, `penalty='l2'`, `balanced`):
| Chỉ số đánh giá (Metric) | Baseline Model | Best Tuned Model | Ý nghĩa học thuật & nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 0.7805 | 0.6801 | Accuracy Baseline là "ảo giác" do chỉ đoán nhãn 0 |
| **Precision** | 0.7391 | 0.3673 | Chấp nhận đánh đổi để mở rộng độ phủ phát hiện |
| **Recall (Độ phát hiện nợ xấu)** | **0.0128** | **0.6167** | **Tăng vọt gấp ~48 lần (từ 17 lên 819 khách)** |
| **F1-Score** | **0.0252** | **0.4604** | **Tăng vọt gấp hơn 18 lần** |
| **ROC AUC** | **0.6847** | **0.7069** | Khả năng phân loại tổng thể tăng |
| **PR-AUC** | **0.4721** | **0.4884** | Độ chính xác dưới đường cong Precision-Recall tăng |
| **Ngưỡng tối ưu F1 ($p \ge 0.60$)** | - | **0.4927** | Tối ưu hóa điểm cân bằng toán học |

---

### 5. Kết luận nghiệp vụ ngân hàng
1. **Kiểm soát rủi ro tài chính:** Sai lầm loại 2 (False Negative - bỏ lọt khách nợ xấu) gây thiệt hại trực tiếp vào vốn của ngân hàng. Mô hình tối ưu hóa giúp ngăn chặn được hơn **61% rủi ro nợ xấu**.
2. **Chọn lọc biến tự động với L1:** Phạt L1 loại bỏ trực tiếp 16 biến ít quan trọng về $0.0000$, chỉ giữ lại các yếu tố rủi ro then chốt nhất giúp tiết kiệm chi phí thu thập dữ liệu và tăng tính giải thích được (interpretability).
3. **Top 3 yếu tố cảnh báo sớm:**
   - `PAY_0` ($w = +0.5211$): Lịch sử chậm trả tháng gần nhất (yếu tố quyết định).
   - `PAY_AMT1` ($w = -0.1315$): Số tiền thanh toán tháng gần nhất (thanh toán càng nhiều, rủi ro càng thấp).
   - `LIMIT_BAL` ($w = -0.1289$): Hạn mức tín dụng ban đầu.
