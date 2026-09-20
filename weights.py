#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weights.py
----------
Module lưu và tải trọng số (weights & bias) cho model LogisticRegression
trong project dự báo vỡ nợ thẻ tín dụng.

Hỗ trợ hai định dạng:
  - .npz  : NumPy binary (nhỏ gọn, nhanh — khuyên dùng)
  - .json : JSON văn bản (dễ đọc, dễ chia sẻ)

Sử dụng:
    from weights import save_weights, load_weights, print_weight_summary

    # Lưu sau khi huấn luyện
    save_weights(model, path="model_weights.npz", feature_names=feature_names)

    # Tải và phục hồi model
    model_loaded = load_weights(path="model_weights.npz")

    # In tóm tắt trọng số
    print_weight_summary(model_loaded)
"""

import os
import json
import numpy as np

# ---------------------------------------------------------------------------
# Hằng số phiên bản — tăng khi format thay đổi để tránh tải nhầm
# ---------------------------------------------------------------------------
_FORMAT_VERSION = "1.0"


# ===========================================================================
# 1. LƯU TRỌNG SỐ
# ===========================================================================

def save_weights(model, path: str = "weights/best_model_weights.npz", feature_names=None, verbose: bool = False) -> str:
    """
    Luu trong so va sieu tham so cua LogisticRegression ra file.
    Chi ghi file neu chua ton tai hoac co thay doi trong so, tranh ghi de lien tuc.

    Parameters
    ----------
    model : LogisticRegression
        Model da duoc huan luyen (da goi .fit()).
    path : str
        Duong dan file luu (.npz hoac .json).
    feature_names : list[str] | None
        Ten cac dac trung (tuy chon), giup dien giai trong so.
    verbose : bool, default=False
        In thong bao ra terminal neu True.

    Returns
    -------
    str
        Duong dan file chinh da luu.
    """
    if model.weights is None:
        raise ValueError(
            "[LOI] Model chua duoc huan luyen. Hay goi model.fit() truoc."
        )

    # Tu dong lay feature_names_ tu model neu khong truyen
    if feature_names is None:
        feature_names = getattr(model, "feature_names_", None)

    # Dam bao thu muc cha ton tai
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Kiem tra neu file da ton tai voi cung trong so thi giu nguyen, khong sinh lai lien tuc
    if os.path.exists(path):
        try:
            if path.endswith(".npz"):
                existing = np.load(path, allow_pickle=True)
                if "weights" in existing and np.allclose(existing["weights"], model.weights, atol=1e-7):
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        if verbose:
                            print(f"[WEIGHTS] File '{path}' da chua trong so toi uu hien tai, giu nguyen khong sinh lai.")
                        return path
        except Exception:
            pass

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        _save_json(model, path, feature_names)
    else:
        # Mac dinh .npz ke ca khi khong co duoi
        if ext != ".npz":
            path = path + ".npz"
        _save_npz(model, path, feature_names)

    # Luon ghi kem file .txt de doc ro trong so
    base = os.path.splitext(path)[0]
    txt_path = base + ".txt"
    _save_txt(model, txt_path, feature_names)

    size_kb  = os.path.getsize(path) / 1024
    txt_size = os.path.getsize(txt_path) / 1024
    if verbose:
        print(f"[WEIGHTS] Da luu -> '{path}'  ({size_kb:.2f} KB)")
        print(f"[WEIGHTS] Da luu -> '{txt_path}'  ({txt_size:.2f} KB)  [de doc]")
    return path


def _save_npz(model, path: str, feature_names):
    """Luu dang NumPy .npz (binary, nen)."""
    meta = _build_meta(model, feature_names)
    np.savez_compressed(
        path,
        weights      = model.weights,
        bias         = np.array([model.bias]),
        meta_json    = np.array(json.dumps(meta, ensure_ascii=False)),
        loss_total   = np.array(model.loss_history_.get('total',   [])),
        loss_bce     = np.array(model.loss_history_.get('bce',     [])),
        loss_penalty = np.array(model.loss_history_.get('penalty', [])),
    )


def _save_json(model, path: str, feature_names):
    """Luu dang JSON van ban (de doc)."""
    meta = _build_meta(model, feature_names)

    # weights_labeled: dict ro rang ten -> gia tri (sap xep theo |w| giam dan)
    w = model.weights
    names = list(feature_names) if feature_names is not None \
            else [f"feature_{i}" for i in range(len(w))]
    order = list(np.argsort(np.abs(w))[::-1])
    weights_labeled = {
        names[i]: round(float(w[i]), 8) for i in order
    }

    payload = {
        **meta,
        "bias"            : float(model.bias),
        "weights_labeled" : weights_labeled,          # <-- ro rang tung dac trung
        "weights_raw"     : model.weights.tolist(),   # <-- mang raw giu nguyen
        "loss_history" : {
            "total"  : model.loss_history_.get('total',   []),
            "bce"    : model.loss_history_.get('bce',     []),
            "penalty": model.loss_history_.get('penalty', []),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_txt(model, path: str, feature_names):
    """Ghi file .txt voi toan bo trong so ro rang, de doc bang mat thuong."""
    import datetime
    w = model.weights
    b = model.bias
    n = len(w)
    names = list(feature_names) if feature_names is not None \
            else [f"feature_{i}" for i in range(n)]

    # Sap xep theo |w| giam dan
    order = list(np.argsort(np.abs(w))[::-1])

    lh   = getattr(model, 'loss_history_', {})
    ep   = len(lh.get('total', []))
    f_lo = lh['total'][-1]  if lh.get('total')   else None
    f_bc = lh['bce'][-1]    if lh.get('bce')     else None
    f_pe = lh['penalty'][-1]if lh.get('penalty') else None

    sep  = "=" * 72
    sep2 = "-" * 72

    lines = []
    lines.append(sep)
    lines.append("  LOGISTIC REGRESSION — TRONG SO MO HINH (Model Weights)")
    lines.append("  Trang thai: San sang van hanh san pham (Production Ready)")
    lines.append(sep)

    # --- Sieu tham so ---
    lines.append("")
    lines.append("[SIEU THAM SO]")
    lines.append(f"  penalty       = {model.penalty}")
    lines.append(f"  C             = {model.C}")
    lines.append(f"  learning_rate = {model.learning_rate}")
    lines.append(f"  max_iter      = {model.max_iter}")
    lines.append(f"  tol           = {model.tol}")
    lines.append(f"  class_weight  = {model.class_weight}")
    lines.append(f"  init          = {model.init}")
    if model.penalty == 'elasticnet':
        lines.append(f"  l1_ratio      = {model.l1_ratio}")

    # --- Ket qua huan luyen ---
    lines.append("")
    lines.append("[KET QUA HUAN LUYEN]")
    lines.append(f"  So epoch da chay : {ep}")
    if f_lo is not None:
        lines.append(f"  Final total loss : {f_lo:.8f}")
        lines.append(f"  Final BCE loss   : {f_bc:.8f}")
        lines.append(f"  Final penalty    : {f_pe:.8f}")

    # --- Thong ke trong so ---
    lines.append("")
    lines.append("[THONG KE TRONG SO]")
    lines.append(f"  So dac trung  : {n}")
    lines.append(f"  Bias  (theta0): {b:+.8f}")
    lines.append(f"  W  mean       : {np.mean(w):+.8f}")
    lines.append(f"  W  std        : {np.std(w):.8f}")
    lines.append(f"  W  min        : {np.min(w):+.8f}")
    lines.append(f"  W  max        : {np.max(w):+.8f}")
    lines.append(f"  ||W||_1 (L1)  : {np.sum(np.abs(w)):.8f}")
    lines.append(f"  ||W||_2^2(L2) : {np.sum(w**2):.8f}")
    lines.append(f"  So w ~= 0     : {int(np.sum(np.abs(w) < 1e-8))} / {n}")

    # --- Bieu do ASCII thanh ngang ---
    lines.append("")
    lines.append("[BIEU DO TRONG SO (thanh ASCII, sap xep theo |w|)]")
    lines.append(sep2)
    max_abs = float(np.max(np.abs(w))) if np.max(np.abs(w)) > 0 else 1.0
    bar_width = 30
    for i in order:
        name  = names[i]
        val   = float(w[i])
        ratio = abs(val) / max_abs
        bar   = int(ratio * bar_width)
        sign  = "+" if val >= 0 else "-"
        bar_str = ("#" * bar).ljust(bar_width)
        lines.append(f"  {name:<28} [{sign}] {bar_str}  {val:>+12.8f}")
    lines.append(sep2)

    # --- Danh sach day du ---
    lines.append("")
    lines.append("[DANH SACH DAY DU TRONG SO (theo chi so goc)]")
    lines.append(sep2)
    lines.append(f"  {'Rank':>4}  {'Idx':>4}  {'Ten Dac Trung':<28}  "
                 f"{'w':>13}  {'|w|':>12}  {'Odds Ratio':>11}")
    lines.append(f"  {'-'*4}  {'-'*4}  {'-'*28}  {'-'*13}  {'-'*12}  {'-'*11}")
    for rank, i in enumerate(order, start=1):
        name = names[i]
        val  = float(w[i])
        val_clipped = np.clip(val, -500, 500)
        odds = float(np.exp(val_clipped))
        lines.append(f"  {rank:>4}  {i:>4}  {name:<28}  "
                     f"{val:>+13.8f}  {abs(val):>12.8f}  {odds:>11.6f}")
    lines.append(sep2)

    # --- Bias ---
    lines.append("")
    b_clipped = np.clip(b, -500, 500)
    lines.append(f"  BIAS (theta_0) = {b:+.8f}  (Odds: {np.exp(b_clipped):.6f})")
    lines.append("")
    lines.append(sep)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_meta(model, feature_names) -> dict:
    """Xay dung dict metadata chung cho ca hai dinh dang."""
    return {
        "_format_version": _FORMAT_VERSION,
        "n_features"     : int(len(model.weights)),
        "feature_names"  : list(feature_names) if feature_names is not None else None,
        "hyperparameters": {
            "max_iter"    : model.max_iter,
            "learning_rate": model.learning_rate,
            "C"           : model.C,
            "penalty"     : model.penalty,
            "l1_ratio"    : model.l1_ratio,
            "tol"         : model.tol,
            "class_weight": model.class_weight,
            "init"        : model.init,
            "random_state": model.random_state,
        },
        "training_info": {
            "epochs_ran"   : int(len(model.loss_history_.get('total', []))),
            "final_loss"   : float(model.loss_history_['total'][-1])
                             if model.loss_history_.get('total') else None,
            "final_bce"    : float(model.loss_history_['bce'][-1])
                             if model.loss_history_.get('bce') else None,
            "final_penalty": float(model.loss_history_['penalty'][-1])
                             if model.loss_history_.get('penalty') else None,
        },
    }


# ===========================================================================
# 2. TAI TRONG SO
# ===========================================================================

def load_weights(path: str = "weights/best_model_weights.npz", verbose: bool = False):
    """
    Tai trong so tu file va phuc hoi doi tuong LogisticRegression.

    Parameters
    ----------
    path : str, default="weights/best_model_weights.npz"
        Duong dan file (.npz hoac .json).
        Neu co file .txt cung ten, no se duoc doc de lay feature_names.
    verbose : bool, default=False
        In thong bao ra terminal neu True.

    Returns
    -------
    LogisticRegression
        Doi tuong model da duoc phuc hoi voi day du weights, bias,
        sieu tham so va lich su loss.

    Raises
    ------
    FileNotFoundError
        Neu file khong ton tai.
    """
    # Import tai day de tranh circular import
    from model import LogisticRegression

    if not os.path.exists(path):
        raise FileNotFoundError(f"[LOI] Khong tim thay file: '{path}'")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        model = _load_json(path, LogisticRegression)
    else:
        model = _load_npz(path, LogisticRegression)

    n_feat = len(model.weights)
    if verbose:
        print(f"[WEIGHTS] Da tai <- '{path}'  ({n_feat} dac trung)")

        # Goi y neu co file .txt kem theo
        base    = os.path.splitext(path)[0]
        txt_path = base + ".txt"
        if os.path.exists(txt_path):
            print(f"[WEIGHTS] Doc chi tiet trong so tai: '{txt_path}'")

    return model


def load_production_bundle(bundle_path: str = "weights/production_bundle.npz"):
    """
    Nạp toàn bộ pipeline mô hình với bộ trọng số tốt nhất đã kiểm định cho dự án.
    Bao gồm model, scaler, ngưỡng tối ưu tau* và danh sách 30 đặc trưng.
    """
    from model import CreditDefaultInferencePipeline
    return CreditDefaultInferencePipeline.load(bundle_path)


def load_best_pipeline(bundle_path: str = "weights/production_bundle.npz"):
    """Bí danh ngắn gọn cho load_production_bundle."""
    return load_production_bundle(bundle_path)


def _load_npz(path: str, LogisticRegression):
    """Tai tu .npz."""
    data = np.load(path, allow_pickle=True)

    meta_json = str(data["meta_json"])
    meta = json.loads(meta_json)

    _check_version(meta.get("_format_version"))

    hp = meta["hyperparameters"]
    model = LogisticRegression(
        max_iter     = hp["max_iter"],
        learning_rate= hp["learning_rate"],
        C            = hp["C"],
        penalty      = hp["penalty"],
        l1_ratio     = hp["l1_ratio"],
        tol          = hp["tol"],
        class_weight = hp["class_weight"],
        init         = hp["init"],
        random_state = hp["random_state"],
    )
    model.weights = data["weights"].copy()
    model.bias    = float(data["bias"][0])
    model.loss_history_ = {
        "total"  : data["loss_total"].tolist() if "loss_total" in data else [],
        "bce"    : data["loss_bce"].tolist() if "loss_bce" in data else [],
        "penalty": data["loss_penalty"].tolist() if "loss_penalty" in data else [],
    }
    model.feature_names_ = meta.get("feature_names")
    model.training_info_ = meta.get("training_info", {})
    return model


def _load_json(path: str, LogisticRegression):
    """Tai tu .json."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _check_version(payload.get("_format_version"))

    hp = payload["hyperparameters"]
    model = LogisticRegression(
        max_iter     = hp["max_iter"],
        learning_rate= hp["learning_rate"],
        C            = hp["C"],
        penalty      = hp["penalty"],
        l1_ratio     = hp["l1_ratio"],
        tol          = hp["tol"],
        class_weight = hp["class_weight"],
        init         = hp["init"],
        random_state = hp["random_state"],
    )
    model.weights = np.array(payload.get("weights_raw", payload.get("weights", [])))
    model.bias    = float(payload["bias"])
    lh = payload.get("loss_history", {})
    model.loss_history_ = {
        "total"  : lh.get("total",   []),
        "bce"    : lh.get("bce",     []),
        "penalty": lh.get("penalty", []),
    }
    model.feature_names_ = payload.get("feature_names")
    model.training_info_ = payload.get("training_info", {})
    return model


def _check_version(version: str):
    """Canh bao neu phien ban format khong khop."""
    if version != _FORMAT_VERSION:
        import warnings
        warnings.warn(
            f"[WEIGHTS] Phien ban format '{version}' khac voi phien ban hien tai "
            f"'{_FORMAT_VERSION}'. Co the xay ra loi tuong thich.",
            UserWarning,
            stacklevel=3,
        )


# ===========================================================================
# 3. IN TOM TAT TRONG SO
# ===========================================================================

def print_weight_summary(model, top_n: int = 10):
    """
    In tom tat thong ke trong so cua model ra Terminal.

    Parameters
    ----------
    model : LogisticRegression
        Model da co weights.
    top_n : int
        So luong dac trung co trong so lon nhat (theo |w|) duoc in ra.
    """
    if model.weights is None:
        print("[WEIGHTS] Model chua co trong so.")
        return

    w = model.weights
    b = model.bias
    names = getattr(model, "feature_names_", None)

    width = 64
    print("\n" + "=" * width)
    print("  TRONG SO MODEL (Weight Summary)")
    print("=" * width)

    # --- Thong ke tong quan ---
    print(f"  So dac trung    : {len(w)}")
    print(f"  Bias (theta_0)  : {b:+.6f}")
    print(f"  W  mean         : {np.mean(w):+.6f}")
    print(f"  W  std          : {np.std(w):.6f}")
    print(f"  W  min          : {np.min(w):+.6f}")
    print(f"  W  max          : {np.max(w):+.6f}")
    print(f"  ||W||_1  (L1)   : {np.sum(np.abs(w)):.6f}")
    print(f"  ||W||_2^2 (L2)  : {np.sum(w**2):.6f}")
    n_zero = int(np.sum(np.abs(w) < 1e-8))
    print(f"  Trong so ~= 0   : {n_zero} / {len(w)}")

    # --- Sieu tham so ---
    if hasattr(model, "max_iter"):
        print(f"\n  Sieu tham so:")
        print(f"    penalty={model.penalty}, C={model.C}, "
              f"lr={model.learning_rate}, max_iter={model.max_iter}")
        if model.penalty == "elasticnet":
            print(f"    l1_ratio={model.l1_ratio}")
        print(f"    class_weight={model.class_weight}, "
              f"tol={model.tol}, init='{model.init}'")

    # --- Lich su Loss ---
    lh = getattr(model, "loss_history_", {})
    if lh.get("total"):
        epochs = len(lh["total"])
        print(f"\n  Lich su Loss ({epochs} epochs):")
        print(f"    Epoch   1 : total={lh['total'][0]:.6f}  "
              f"bce={lh['bce'][0]:.6f}  penalty={lh['penalty'][0]:.6f}")
        if epochs > 1:
            print(f"    Epoch {epochs:3d} : total={lh['total'][-1]:.6f}  "
                  f"bce={lh['bce'][-1]:.6f}  penalty={lh['penalty'][-1]:.6f}")

    # --- Top N trong so theo |w| ---
    top_n = min(top_n, len(w))
    top_idx = np.argsort(np.abs(w))[::-1][:top_n]
    print(f"\n  Top {top_n} dac trung anh huong nhat (|w| giam dan):")
    print(f"  {'Idx':>5}  {'Ten':<28}  {'w':>12}  {'|w|':>10}")
    print(f"  {'-'*5}  {'-'*28}  {'-'*12}  {'-'*10}")
    for idx in top_idx:
        name = names[idx] if (names and idx < len(names)) else f"feature_{idx}"
        print(f"  {idx:>5}  {name:<28}  {w[idx]:>+12.6f}  {abs(w[idx]):>10.6f}")

    print("=" * width + "\n")


# ===========================================================================
# 4. TIEN ICH BO SUNG
# ===========================================================================

def get_weight_dataframe(model) -> dict:
    """
    Tra ve dict {feature_name: weight} sap xep theo |w| giam dan.
    Co the chuyen thanh Pandas DataFrame bang: pd.DataFrame(get_weight_dataframe(model))

    Parameters
    ----------
    model : LogisticRegression

    Returns
    -------
    dict voi cac key: 'feature', 'weight', 'abs_weight', 'rank'
    """
    if model.weights is None:
        raise ValueError("[LOI] Model chua co trong so.")

    w = model.weights
    names = getattr(model, "feature_names_", None)
    if names is None:
        names = [f"feature_{i}" for i in range(len(w))]

    order = np.argsort(np.abs(w))[::-1]
    return {
        "feature"    : [names[i] for i in order],
        "weight"     : [float(w[i]) for i in order],
        "abs_weight" : [float(abs(w[i])) for i in order],
        "rank"       : list(range(1, len(w) + 1)),
    }


# ===========================================================================
# THỰC THI (chạy trực tiếp: python weights.py)
# Chỉ nạp và hiển thị bộ trọng số tối ưu nhất của dự án, không sinh file liên tục
# ===========================================================================

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 72)
    print("  HỆ THỐNG TRỌNG SỐ MÔ HÌNH — CHỈ NẠP BỘ TRỌNG SỐ TỐI ƯU NHẤT")
    print("=" * 72)

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    bundle_path = os.path.join(weights_dir, "production_bundle.npz")
    best_weights_path = os.path.join(weights_dir, "best_model_weights.npz")

    if os.path.exists(bundle_path):
        from model import CreditDefaultInferencePipeline

        pipeline = CreditDefaultInferencePipeline.load(bundle_path)
        model = pipeline.model
        feature_names = pipeline.feature_names
        threshold = pipeline.threshold

        print(f"  • Nguồn tệp trọng số    : {bundle_path}")
        print(f"  • Số đặc trưng          : {len(feature_names)}")
        print(f"  • Hệ số chệch (Bias)    : {model.bias:+.6f}")
        print(f"  • Ngưỡng tối ưu (tau*)  : {threshold:.4f}")
        print(f"  • Siêu tham số tối ưu   : C={model.C}, penalty='{model.penalty}', class_weight='{model.class_weight}'")

        # Đọc metadata nếu có
        try:
            npz_data = np.load(bundle_path, allow_pickle=True)
            if "meta_json" in npz_data:
                meta = json.loads(str(npz_data["meta_json"]))
                if "test_metrics" in meta:
                    tm = meta["test_metrics"]
                    print(f"  • Hiệu năng Test độc lập: Acc={tm.get('accuracy')*100:.2f}% | Prec={tm.get('precision')*100:.2f}% | Rec={tm.get('recall')*100:.2f}% | F1={tm.get('f1_score'):.4f} | AUC={tm.get('roc_auc'):.4f}")
        except Exception:
            pass

        print("\n  TOP 10 ĐẶC TRƯNG CÓ TRỌNG SỐ LỚN NHẤT TRONG MÔ HÌNH:")
        print("  +" + "-"*6 + "+" + "-"*26 + "+" + "-"*14 + "+" + "-"*14 + "+")
        print(f"  | {'Top':^4} | {'Đặc trưng':<24} | {'Trọng số w':^12} | {'Odds Ratio':^12} |")
        print("  +" + "-"*6 + "+" + "-"*26 + "+" + "-"*14 + "+" + "-"*14 + "+")
        order = np.argsort(np.abs(model.weights))[::-1][:10]
        for rank, idx in enumerate(order, 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            w_val = model.weights[idx]
            odds = np.exp(w_val)
            print(f"  | {rank:^4} | {fname:<24} | {w_val:^+12.4f} | {odds:^12.4f} |")
        print("  +" + "-"*6 + "+" + "-"*26 + "+" + "-"*14 + "+" + "-"*14 + "+")
        print("\n  [XÁC NHẬN] Đã nạp thành công tệp trọng số tốt nhất duy nhất.")
        print("  Không sinh file tạm, không in lặp lại.")
        print("=" * 72)

    elif os.path.exists(best_weights_path):
        model = load_weights(best_weights_path, verbose=False)
        print(f"  • Nguồn tệp đã nạp: {best_weights_path}")
        print(f"  • Số đặc trưng    : {len(model.weights)}")
        print(f"  • Hệ số chệch     : {model.bias:+.6f}")
        print_weight_summary(model, top_n=10)
    else:
        print(f"  [CẢNH BÁO] Chưa tìm thấy tệp trọng số tối ưu tại '{weights_dir}'.")
        print("  Vui lòng chạy 'python run_pipeline.py' để khởi tạo.")
