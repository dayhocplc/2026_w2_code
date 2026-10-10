# -*- coding: utf-8 -*-
"""
ts_stats_extra.py — Các hàm thống kê bổ sung cho bản chỉnh sửa theo phản biện.

Dùng chung cho:
    nested_phi_poland.py      (mục A3, B1, B8 — câu hỏi số 3 của Hội đồng)
    sens_sp_simulation.py     (mục A4 — câu hỏi số 4)
    calibration_poland.py     (mục B5 — hiệu chuẩn, TRIPOD+AI)

Không thay thế ts_evaluate.py; chỉ thêm những thứ ts_evaluate chưa có:
    • AUC nhanh theo hạng (xử lý hạng trùng), khớp đúng Mann–Whitney
    • Kiểm định DeLong ghép cặp (Sun & Xu 2014, bản nhanh)
    • Bootstrap ghép cặp TÍNH ĐẾN NHIỀU LẦN LẶP kiểm định chéo (mục B8)
    • Hiệu chỉnh Holm cho họ phép kiểm (mục B1)
    • Hồi quy logistic 1–2 tham số bằng Newton (hiệu chuẩn, không cần statsmodels)
    • Dịch xác suất theo tỉ lệ hiện mắc (thay cho công thức 3.28 ở mục 4.7.5)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, rankdata


# =====================================================================
# AUC
# =====================================================================

def auc_fast(y, s) -> float:
    """ROC-AUC = thống kê Mann–Whitney chuẩn hoá; hạng trùng lấy hạng trung bình."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - m * (m + 1) / 2.0) / (m * n))


# =====================================================================
# DeLong ghép cặp
# =====================================================================

def delong_paired(y, s1, s2) -> tuple[float, float, float]:
    """→ (AUC1 − AUC2, sai số chuẩn, p hai phía). DeLong 1988, cài đặt theo Sun & Xu 2014."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v10s, v01s = [], [], []
    for s in (s1, s2):
        s = np.asarray(s, dtype=float)
        X, Y = s[pos], s[neg]
        tx, ty = rankdata(X), rankdata(Y)
        tz = rankdata(np.concatenate([X, Y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2.0) / (m * n))
        v10s.append((tz[:m] - tx) / n)          # giá trị đặt chỗ của ca dương
        v01s.append(1.0 - (tz[m:] - ty) / m)    # giá trị đặt chỗ của ca âm
    S = np.cov(np.vstack(v10s)) / m + np.cov(np.vstack(v01s)) / n
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    if var <= 0:
        return float(diff), 0.0, 1.0
    se = float(np.sqrt(var))
    return float(diff), se, float(2.0 * norm.sf(abs(diff) / se))


# =====================================================================
# Bootstrap ghép cặp tính đến nhiều lần lặp (mục B8)
# =====================================================================

def paired_boot_repeats(y, SA, SB, n_boot: int = 2000, seed: int = 20260827):
    """Bootstrap phân tầng cho hiệu AUC trung bình qua các lần lặp.

    SA, SB: mảng (R × n) điểm out-of-fold của R lần lặp kiểm định chéo; mô hình
    tất định có thể truyền một hàng (1 × n), sẽ được dùng chung cho mọi lần lặp.

    Mỗi mẫu bootstrap chọn lại TRẺ (phân tầng theo lớp), rồi tính
        Δ* = trung bình_r [ AUC(SA_r) − AUC(SB_r) ]  trên cùng bộ trẻ đó.
    Cách này giữ đúng cấu trúc phụ thuộc: cùng một trẻ xuất hiện ở mọi lần lặp.
    Khác với ts_external.paired_auc_test (chạy trên điểm đã lấy trung bình qua
    các lần lặp) — chính sự khác biệt này giải thích vì sao hiệu ở Bảng 5.4
    không khớp với hiệu tính từ cột auc_mean của Bảng 5.3.

    → (hiệu quan sát, (CI95 thấp, CI95 cao), p hai phía)
    """
    y = np.asarray(y).astype(int)
    SA = np.atleast_2d(np.asarray(SA, dtype=float))
    SB = np.atleast_2d(np.asarray(SB, dtype=float))
    R = max(SA.shape[0], SB.shape[0])
    if SA.shape[0] == 1:
        SA = np.repeat(SA, R, axis=0)
    if SB.shape[0] == 1:
        SB = np.repeat(SB, R, axis=0)

    def mean_diff(idx):
        yy = y[idx]
        return float(np.mean([auc_fast(yy, SA[r, idx]) - auc_fast(yy, SB[r, idx])
                              for r in range(R)]))

    obs = mean_diff(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        diffs[b] = mean_diff(idx)
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return obs, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), min(p, 1.0)


def delong_over_repeats(y, SA, SB) -> dict:
    """DeLong trên từng lần lặp → trung vị hiệu và trung vị p (đối chiếu với bootstrap)."""
    SA = np.atleast_2d(np.asarray(SA, dtype=float))
    SB = np.atleast_2d(np.asarray(SB, dtype=float))
    R = max(SA.shape[0], SB.shape[0])
    res = [delong_paired(y, SA[min(r, SA.shape[0] - 1)], SB[min(r, SB.shape[0] - 1)])
           for r in range(R)]
    d = np.array([x[0] for x in res])
    p = np.array([x[2] for x in res])
    return {"diff_median": float(np.median(d)), "p_median": float(np.median(p)),
            "p_max": float(p.max()), "p_min": float(p.min())}


# =====================================================================
# Hiệu chỉnh so sánh nhiều lần (mục B1)
# =====================================================================

def holm(pvals: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """Holm–Bonferroni. → danh sách {p, p_adj, reject} theo đúng thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=float)
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return [{"p": float(p[i]), "p_adj": float(adj[i]), "reject": bool(adj[i] <= alpha)}
            for i in range(k)]


# =====================================================================
# Hồi quy logistic nhỏ (hiệu chuẩn)
# =====================================================================

def logistic_newton(x, y, offset=None, with_slope: bool = True,
                    ridge: float = 1e-8, iters: int = 100) -> np.ndarray:
    """Khớp y ~ a (+ b·x) (+ offset). → [a] hoặc [a, b]."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x]) if with_slope else np.ones((len(y), 1))
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta + off)
        W = p * (1.0 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_metrics(y, p, n_bins: int = 10) -> dict:
    """Brier, Brier chuẩn hoá, hệ số chặn hiệu chuẩn (CITL), độ dốc hiệu chuẩn, ECE."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = logit(p)
    brier = float(np.mean((p - y) ** 2))
    prev = float(y.mean())
    brier_ref = prev * (1 - prev)
    a_b = logistic_newton(lp, y, with_slope=True)
    citl = logistic_newton(lp, y, offset=lp, with_slope=False)[0]
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return {"brier": brier, "brier_ref": brier_ref,
            "brier_scaled": 1.0 - brier / brier_ref if brier_ref > 0 else float("nan"),
            "citl": float(citl), "slope": float(a_b[1]), "ece": float(ece)}


def reliability_table(y, p, n_bins: int = 10) -> list[dict]:
    """Bảng độ tin cậy theo phân vị xác suất dự đoán (dùng cho biểu đồ hiệu chuẩn)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            out.append({"bin": b + 1, "n": int(sel.sum()),
                        "p_mean": float(p[sel].mean()),
                        "k_pos": int(y[sel].sum()),
                        "obs_rate": float(y[sel].mean())})
    return out


def prior_shift(p, pi_from: float, pi_to: float) -> np.ndarray:
    """Dịch xác suất hậu nghiệm từ tỉ lệ nền pi_from sang pi_to (giữ nguyên tỉ số khả dĩ).

        logit p' = logit p + ln[pi_to/(1−pi_to)] − ln[pi_from/(1−pi_from)]

    Đây là cách đúng để có xác suất RIÊNG cho từng hồ sơ; công thức (3.28) chỉ
    cho PPV tại một ngưỡng, không cho xác suất theo điểm của từng hồ sơ.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    shift = np.log(pi_to / (1 - pi_to)) - np.log(pi_from / (1 - pi_from))
    return expit(logit(p) + shift)


def fp_per_tp(se: float, sp: float, prev: float) -> float:
    """Số ca báo động nhầm cho mỗi ca đúng ở tỉ lệ hiện mắc prev."""
    tp = se * prev
    return float("inf") if tp <= 0 else (1 - sp) * (1 - prev) / tp


def flagged_rate(se: float, sp: float, prev: float) -> float:
    """Tỉ lệ trẻ vào diện theo dõi ở tỉ lệ hiện mắc prev."""
    return se * prev + (1 - sp) * (1 - prev)
