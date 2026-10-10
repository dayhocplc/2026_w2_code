# -*- coding: utf-8 -*-
"""
run_w2_mamdani_theorem.py — W2-1: kiểm chứng số cho Định lý 1 (tầng Mamdani).

Định lý 1 (phát biểu đầy đủ ở manuscript/W2/W2_proofs.md) tách đơn điệu của đầu ra
Mamdani thành HAI điều kiện độc lập:

  (L) Bổ đề trội ngẫu nhiên: hàm thuộc mỗi đầu vào là phân hoạch Ruspini CÓ THỨ TỰ
      (tổng = 1 và mọi tổng tích luỹ F_j(x) = Σ_{l≤j} μ_l(x) không tăng theo x),
      t-norm tích, bảng luật ĐẦY ĐỦ và đơn điệu, gộp luật cùng kết luận bằng tổng
      chặn  ⇒  vectơ trọng số theo kết luận w(x) là phân phối xác suất, tăng theo
      thứ tự trội ngẫu nhiên bậc nhất (FOSD) theo từng đầu vào.
  (D) Điều kiện giải mờ: ánh xạ D: Δ_M → ℝ (w ↦ đầu ra giải mờ) không giảm theo
      mọi hướng e_{c+1} − e_c trên đơn hình. (D) chỉ phụ thuộc các tập kết luận.

Script này:
  1. kiểm tra điều kiện Ruspini có thứ tự cho bộ hàm thuộc hiện hành và bộ gốc (V1);
  2. kiểm tra (L) trên lưới cho năm cấu hình của Bảng 5.14;
  3. CHỨNG NHẬN CHÍNH XÁC (D) cho giải mờ trọng tâm max–min trên universe rời rạc
     của dự án: D là hàm phân thức tuyến tính từng mảnh trên một sắp xếp đường
     thẳng hữu hạn; tử số đạo hàm theo hướng là affine trong mỗi ô nên cực tiểu
     của nó trên ô đạt tại đỉnh ⇒ duyệt mọi đỉnh của sắp xếp là đủ;
  4. điều kiện (D) giải tích cho mô hình cộng chuẩn (SAM, tích–tổng);
  5. tái lập Bảng 5.14 và thử bảng luật KHÔNG trơn (nhảy THAP → CAO): với tổng chặn
     vẫn đơn điệu, với cực đại thì không (so với Van Broekhoven & De Baets 2009);
  6. thử căng thẳng: phân hoạch Ruspini có thứ tự ngẫu nhiên × bảng luật đơn điệu
     ngẫu nhiên (kể cả không trơn), 2 và 3 đầu vào.

Chạy:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/run_w2_mamdani_theorem.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import config as cfg                                            # noqa: E402
from run_fis_bypass import mf, Mamdani, monotonicity           # noqa: E402

ROOT = os.path.dirname(SRC)
OUT_TXT = os.path.join(ROOT, "out", "reports", "w2_mamdani_theorem.txt")
OUT_JSON = os.path.join(ROOT, "out", "reports", "w2_mamdani_theorem.json")
SEED = 20260925
D_ORDER = ["KHONG_THIEU_HUT", "THIEU_HUT_NHE", "THIEU_HUT_RO"]
L_ORDER = ["DUNG_HAN", "TRE_NHE", "TRE_NANG"]
C_ORDER = ["THAP", "TRUNG_BINH", "CAO"]


# =====================================================================
# 1. Phân hoạch Ruspini có thứ tự
# =====================================================================
def partition_report(mfset, order, lo, hi, n=30001):
    x = np.linspace(lo, hi, n)
    M = np.array([mf(*mfset[k], x) for k in order])
    s = M.sum(0)
    F = np.cumsum(M, 0)[:-1]                       # F_1..F_{K-1}
    inc = np.max(np.diff(F, axis=1)) if len(F) else 0.0
    return {"max_abs_sum_minus_1": float(np.max(np.abs(s - 1))),
            "min_sum": float(s.min()),
            "max_increase_of_cumulative": float(max(inc, 0.0)),
            "ordered_ruspini": bool(np.max(np.abs(s - 1)) < 1e-12 and inc <= 1e-12)}


# =====================================================================
# Hệ Mamdani vectơ hoá, tổng quát số đầu vào
# =====================================================================
class GridMamdani:
    """Đầu vào: danh sách ma trận hàm thuộc (K_i × n_i) đã tính trên lưới; bảng luật
    r: mảng số nguyên shape (K_1, …, K_n) chỉ số kết luận; C: (M × nz) tập kết luận."""

    def __init__(self, z, C, rules, snorm="bounded_sum", tnorm="product"):
        self.z, self.C, self.r = z, C, np.asarray(rules)
        self.snorm, self.tnorm = snorm, tnorm
        self.M = C.shape[0]

    def weights(self, mus):
        """mus: list các vectơ độ thuộc (K_i,) tại một điểm → w (M,)."""
        firing = mus[0]
        for m in mus[1:]:
            firing = (np.multiply.outer(firing, m) if self.tnorm == "product"
                      else np.minimum.outer(firing, m))
        w = np.zeros(self.M)
        for c in range(self.M):
            f = firing[self.r == c]
            if f.size == 0:
                continue
            if self.snorm == "bounded_sum":
                w[c] = min(1.0, f.sum())
            elif self.snorm == "max":
                w[c] = f.max()
            elif self.snorm == "prob_sum":
                w[c] = 1 - np.prod(1 - f)
            else:
                raise ValueError(self.snorm)
        return w

    def defuzz(self, w):
        agg = np.max(np.minimum(w[:, None], self.C), 0)
        return float((self.z * agg).sum() / agg.sum())


def cons_matrix(z):
    return np.array([mf(*cfg.MF_RISK_DOMAIN[k], z) for k in C_ORDER])


def rules_array():
    r = np.zeros((3, 3), int)
    for a, b, c in cfg.RULES_DOMAIN:
        r[D_ORDER.index(a), L_ORDER.index(b)] = C_ORDER.index(c)
    return r


# =====================================================================
# 2. Bổ đề (L): w(x) tăng theo FOSD
# =====================================================================
def fosd_check(sys_, mfd, mfl, nd=101, nl=301):
    """Trả về mức giảm lớn nhất của các đuôi P(c ≥ k) = Σ_{c≥k} w_c khi một đầu vào
    tăng, cùng Σ w lệch 1 lớn nhất. FOSD ⇔ mọi đuôi không giảm (với Σ w = 1)."""
    dg = np.linspace(0, 1, nd)
    lg = np.linspace(0, cfg.LAG_MAX, nl)
    Md = np.array([mf(*mfd[k], dg) for k in D_ORDER]).T
    Ml = np.array([mf(*mfl[k], lg) for k in L_ORDER]).T
    W = np.array([[sys_.weights([Md[i], Ml[j]]) for i in range(nd)] for j in range(nl)])
    tails = np.cumsum(W[..., ::-1], -1)[..., ::-1][..., 1:]     # P(c≥2), P(c≥3)
    drop_d = float(np.max(np.clip(tails[:, :-1] - tails[:, 1:], 0, None)))
    drop_l = float(np.max(np.clip(tails[:-1] - tails[1:], 0, None)))
    return {"max_tail_drop_d": drop_d, "max_tail_drop_lag": drop_l,
            "max_abs_sum_w_minus_1": float(np.max(np.abs(W.sum(-1) - 1)))}


# =====================================================================
# 3. Chứng nhận chính xác điều kiện (D) cho trọng tâm max–min rời rạc
# =====================================================================
def _regime_coeffs(w, C, z):
    """Tại điểm w (trong phần trong một ô), agg_k = hằng hoặc w_c. Trả hệ số
    num(w) = a·w + b, den(w) = c·w + e (w ∈ ℝ^M)."""
    M, nz = C.shape
    clipped = w[:, None] < C                          # min(w_c, C_c(z)) = w_c
    vals = np.where(clipped, w[:, None], C)
    arg = np.argmax(vals, 0)
    a = np.zeros(M); cc = np.zeros(M); b = 0.0; e = 0.0
    for k in range(nz):
        c = arg[k]
        if clipped[c, k]:
            a[c] += z[k]; cc[c] += 1.0
        else:
            b += z[k] * C[c, k]; e += C[c, k]
    return a, b, cc, e


def certify_D(C, z, delta=1e-9, ndir=24):
    """Duyệt mọi đỉnh của sắp xếp đường thẳng {w_c = t}, {w_c = w_c'} trên đơn hình
    Δ_3 (toạ độ (w1, w2), w3 = 1 − w1 − w2). Tại mỗi đỉnh, lấy hệ số của mọi ô kề
    (nhiễu nhỏ theo ndir hướng) và tính tử số đạo hàm theo hướng
        N_v(w) = (a·v)(c·w + e) − (c·v)(a·w + b),
    affine trong ô ⇒ cực tiểu trên ô đạt tại đỉnh. Hướng: v1 = e2 − e1, v2 = e3 − e2.
    Trả min N_v / den² (= đạo hàm theo hướng nhỏ nhất) trên mọi (ô, đỉnh)."""
    assert C.shape[0] == 3
    T = np.unique(np.round(C.ravel(), 12))
    T = T[(T >= 0) & (T <= 1)]
    # đường thẳng dạng α·w1 + β·w2 = γ
    lines = []
    for t in T:
        lines += [(1, 0, t), (0, 1, t), (1, 1, 1 - t)]          # w1=t, w2=t, w3=t
    lines += [(1, -1, 0), (2, 1, 1), (1, 2, 1)]                  # w1=w2, w1=w3, w2=w3
    L = np.array(lines, float)
    # các đỉnh: giao từng cặp đường không song song, nằm trong đơn hình
    verts = []
    A1, B1, G1 = L[:, 0], L[:, 1], L[:, 2]
    for i in range(len(L)):
        det = A1[i] * B1[i + 1:] - B1[i] * A1[i + 1:]
        ok = np.abs(det) > 1e-12
        j = np.arange(i + 1, len(L))[ok]
        d = det[ok]
        x = (G1[i] * B1[j] - B1[i] * G1[j]) / d
        y = (A1[i] * G1[j] - G1[i] * A1[j]) / d
        inside = (x >= -1e-12) & (y >= -1e-12) & (x + y <= 1 + 1e-12)
        verts.append(np.column_stack([x[inside], y[inside]]))
    V = np.unique(np.round(np.vstack(verts), 12), axis=0)
    V = np.clip(V, 0, 1)
    ang = 2 * np.pi * (np.arange(ndir) + 0.5) / ndir
    dirs = np.column_stack([np.cos(ang), np.sin(ang)])
    v_dirs = {"e2-e1": np.array([-1.0, 1.0, 0.0]), "e3-e2": np.array([0.0, -1.0, 1.0])}
    worst = {k: (np.inf, None) for k in v_dirs}
    n_cells_eval = 0
    for P in V:
        w_v = np.array([P[0], P[1], 1 - P[0] - P[1]])
        for u in dirs:
            q = P + delta * u
            wq = np.array([q[0], q[1], 1 - q[0] - q[1]])
            if wq.min() < 0:
                continue
            a, b, cc, e = _regime_coeffs(wq, C, z)
            num = a @ w_v + b; den = cc @ w_v + e
            n_cells_eval += 1
            for k, v in v_dirs.items():
                N = (a @ v) * den - (cc @ v) * num
                g = N / den ** 2
                if g < worst[k][0]:
                    worst[k] = (float(g), w_v.round(6).tolist())
    return {"n_lines": int(len(L)), "n_thresholds": int(len(T)), "n_vertices": int(len(V)),
            "n_cell_vertex_pairs": int(n_cells_eval),
            "min_directional_derivative": {k: v[0] for k, v in worst.items()},
            "argmin": {k: v[1] for k, v in worst.items()}}


def grid_D_check(C, z, N=300, h=1e-7, kind="maxmin"):
    """Kiểm tra độc lập bằng sai phân (không phải chứng minh)."""
    worst = [np.inf, np.inf]
    V = [np.array([-1, 1, 0.]), np.array([0, -1, 1.])]
    for i in range(N + 1):
        for j in range(N + 1 - i):
            w = np.array([i / N, j / N, 1 - i / N - j / N])
            for k, v in enumerate(V):
                w2 = w + h * v
                if w2.min() < 0:
                    continue
                if kind == "maxmin":
                    f = lambda ww: (lambda g: (z * g).sum() / g.sum())(np.max(np.minimum(ww[:, None], C), 0))
                else:
                    f = lambda ww: (lambda g: (z * g).sum() / g.sum())((ww[:, None] * C).sum(0))
                worst[k] = min(worst[k], (f(w2) - f(w)) / h)
    return {"min_fd_e2-e1": float(worst[0]), "min_fd_e3-e2": float(worst[1]), "N": N}


# =====================================================================
# 4. (D) giải tích cho SAM: D(w) = Σ w_c V_c z_c / Σ w_c V_c
# =====================================================================
def sam_condition(C, z):
    V = C.sum(1)
    zc = (C * z).sum(1) / V
    # dấu đạo hàm theo e_{c+1} − e_c tại mức đầu ra y: V_{c+1}(z_{c+1} − y) − V_c(z_c − y),
    # affine theo y ⇒ chỉ cần kiểm tra y ∈ {min z_c, max z_c}.
    rows = []
    for c in range(len(V) - 1):
        for yv in (zc.min(), zc.max()):
            rows.append(float(V[c + 1] * (zc[c + 1] - yv) - V[c] * (zc[c] - yv)))
    return {"areas": V.tolist(), "centroids": zc.tolist(), "min_sign_quantity": min(rows),
            "holds": bool(min(rows) > 0)}


# =====================================================================
# 5. Bảng 5.14 + bảng luật không trơn + t-norm min
# =====================================================================
def grid_violation(sys_, mfd, mfl, step_d=0.01, step_l=0.01, orders=(D_ORDER, L_ORDER)):
    dg = np.arange(0, 1 + 1e-9, step_d)
    lg = np.arange(0, cfg.LAG_MAX + 1e-9, step_l)
    Md = np.array([mf(*mfd[k], dg) for k in orders[0]]).T
    Ml = np.array([mf(*mfl[k], lg) for k in orders[1]]).T
    S = np.array([[sys_.defuzz(sys_.weights([Md[i], Ml[j]])) for i in range(len(dg))]
                  for j in range(len(lg))])
    vd = float(np.clip(S[:, :-1] - S[:, 1:], 0, None).sum())
    vl = float(np.clip(S[:-1] - S[1:], 0, None).sum())
    return vd, vl


# =====================================================================
# 6. Thử căng thẳng ngẫu nhiên
# =====================================================================
def random_ordered_ruspini(rng, K, lo, hi):
    """K tập hình thang/tam giác liền kề, lõi có thứ tự, chỉ chồng lấn kề nhau."""
    pts = np.sort(rng.uniform(lo, hi, 2 * (K - 1)))
    # vùng chuyển tiếp j: [pts[2j], pts[2j+1]]
    cuts = [(pts[2 * j], pts[2 * j + 1]) for j in range(K - 1)]

    def mu(x):
        x = np.asarray(x, float)
        M = np.zeros((K, x.size))
        for j in range(K):
            left = np.ones_like(x) if j == 0 else np.clip((x - cuts[j - 1][0]) /
                                                          max(cuts[j - 1][1] - cuts[j - 1][0], 1e-12), 0, 1)
            right = np.ones_like(x) if j == K - 1 else np.clip((cuts[j][1] - x) /
                                                               max(cuts[j][1] - cuts[j][0], 1e-12), 0, 1)
            M[j] = np.minimum(left, right)
        return M
    return mu


def random_monotone_rules(rng, shape, M):
    """Bảng luật đơn điệu ngẫu nhiên: r = clip(⌊Σ tăng ngẫu nhiên⌋) — cho phép nhảy > 1."""
    grids = np.meshgrid(*[np.sort(rng.uniform(0, 1, k)) for k in shape], indexing="ij")
    wts = rng.dirichlet(np.ones(len(shape)))
    s = sum(w * g for w, g in zip(wts, grids))
    thr = np.sort(rng.uniform(s.min(), s.max(), M - 1))
    return np.searchsorted(thr, s, side="right")


def stress(n_trials=400, seed=SEED):
    rng = np.random.default_rng(seed)
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    C = cons_matrix(z)
    res = {"bounded_sum": [], "max": []}
    nonsmooth = 0
    for t in range(n_trials):
        n_in = 2 if t % 2 == 0 else 3
        Ks = rng.integers(2, 5, n_in)
        mus = [random_ordered_ruspini(rng, int(k), 0, 1) for k in Ks]
        r = random_monotone_rules(rng, tuple(int(k) for k in Ks), 3)
        jump = max(int(np.max(np.abs(np.diff(r, axis=ax)))) for ax in range(n_in))
        nonsmooth += jump > 1
        g = np.linspace(0, 1, 41 if n_in == 2 else 17)
        MU = [m(g) for m in mus]
        for sn in res:
            sys_ = GridMamdani(z, C, r, snorm=sn)
            out = np.empty([len(g)] * n_in)
            for idx in itertools.product(range(len(g)), repeat=n_in):
                out[idx] = sys_.defuzz(sys_.weights([MU[i][:, idx[i]] for i in range(n_in)]))
            v = sum(float(np.clip(-np.diff(out, axis=ax), 0, None).sum()) for ax in range(n_in))
            res[sn].append({"viol": v, "jump": jump, "n_in": n_in})
    summ = {}
    for sn, rows in res.items():
        v = np.array([r["viol"] for r in rows]); j = np.array([r["jump"] for r in rows])
        summ[sn] = {"n": len(rows), "n_violating": int((v > 1e-9).sum()),
                    "max_violation": float(v.max()),
                    "n_violating_nonsmooth": int(((v > 1e-9) & (j > 1)).sum()),
                    "n_violating_smooth": int(((v > 1e-9) & (j <= 1)).sum())}
    summ["n_nonsmooth_rulebases"] = int(nonsmooth)
    return summ


def main():
    t0 = time.time()
    out = {}
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    C = cons_matrix(z)
    r = rules_array()

    out["partitions"] = {
        "MF_D": partition_report(cfg.MF_D, D_ORDER, 0, 1),
        "MF_LAG": partition_report(cfg.MF_LAG, L_ORDER, 0, cfg.LAG_MAX),
        "MF_D_V1": partition_report(cfg.MF_D_V1, D_ORDER, 0, 1),
        "MF_LAG_V1": partition_report(cfg.MF_LAG_V1, L_ORDER, 0, cfg.LAG_MAX),
    }
    out["rule_base"] = {"matrix": r.tolist(),
                        "monotone": bool((np.diff(r, axis=0) >= 0).all() and (np.diff(r, axis=1) >= 0).all()),
                        "max_jump": int(max(np.abs(np.diff(r, axis=0)).max(), np.abs(np.diff(r, axis=1)).max()))}

    configs = [
        ("original MFs + max", cfg.MF_D_V1, cfg.MF_LAG_V1, "max"),
        ("original MFs + bounded sum", cfg.MF_D_V1, cfg.MF_LAG_V1, "bounded_sum"),
        ("Ruspini + max", cfg.MF_D, cfg.MF_LAG, "max"),
        ("Ruspini + probabilistic sum", cfg.MF_D, cfg.MF_LAG, "prob_sum"),
        ("Ruspini + bounded sum", cfg.MF_D, cfg.MF_LAG, "bounded_sum"),
    ]
    out["lemma_fosd"] = {}
    out["table_5_14"] = {}
    for name, md, ml, sn in configs:
        s = GridMamdani(z, C, r, snorm=sn)
        out["lemma_fosd"][name] = fosd_check(s, md, ml)
        vd, vl = grid_violation(s, md, ml)
        # đối chiếu với cài đặt độc lập của run_fis_bypass (Bảng 5.12/5.14 luận văn)
        ref = monotonicity(Mamdani(md, ml, sn))
        out["table_5_14"][name] = {"viol_d": vd, "viol_lag": vl,
                                   "ref_viol_d": ref[0], "ref_viol_lag": ref[1]}

    # t-norm min thay tích (Ruspini + tổng chặn)
    s_min = GridMamdani(z, C, r, snorm="bounded_sum", tnorm="min")
    out["tnorm_min"] = {"fosd": fosd_check(s_min, cfg.MF_D, cfg.MF_LAG),
                        "viol": grid_violation(s_min, cfg.MF_D, cfg.MF_LAG)}

    # bảng luật đơn điệu KHÔNG trơn (nhảy THAP → CAO)
    r_ns = np.array([[0, 0, 2], [0, 2, 2], [2, 2, 2]])

    # Phép kéo theo: cắt (min, Mamdani) vs co giãn (tích, Larsen) × gộp luật × bảng luật.
    # Van Broekhoven & De Baets (2009) dùng CÙNG t-norm cho phép hội và phép kéo theo.
    class ProdImp(GridMamdani):
        def defuzz(self, w):
            agg = np.max(w[:, None] * self.C, 0)
            return float((self.z * agg).sum() / agg.sum())
    impl = []
    for rname, rr in (("smooth", r), ("nonsmooth", r_ns)):
        for conj, imp, sn in (("product", "min", "max"), ("product", "min", "bounded_sum"),
                              ("product", "product", "max"), ("product", "product", "bounded_sum"),
                              ("min", "min", "max"), ("min", "min", "bounded_sum")):
            cls = ProdImp if imp == "product" else GridMamdani
            v = grid_violation(cls(z, C, rr, snorm=sn, tnorm=conj), cfg.MF_D, cfg.MF_LAG)
            impl.append({"rules": rname, "conjunction": conj, "implication": imp,
                         "aggregation": sn, "viol_d": v[0], "viol_lag": v[1]})
    out["implication_variants"] = impl
    out["nonsmooth_rulebase"] = {"matrix": r_ns.tolist()}
    for sn in ("bounded_sum", "max"):
        s = GridMamdani(z, C, r_ns, snorm=sn)
        out["nonsmooth_rulebase"][sn] = {"viol": grid_violation(s, cfg.MF_D, cfg.MF_LAG),
                                         "fosd": fosd_check(s, cfg.MF_D, cfg.MF_LAG)}

    # điều kiện (D)
    tc = time.time()
    out["certificate_maxmin_centroid"] = certify_D(C, z)
    out["certificate_maxmin_centroid"]["sec"] = time.time() - tc
    out["fd_check_maxmin"] = grid_D_check(C, z, kind="maxmin")
    zf = np.linspace(0, 100, 20001)                          # universe gần liên tục
    out["fd_check_maxmin_fine_universe"] = grid_D_check(cons_matrix(zf), zf, N=120, kind="maxmin")
    out["sam_condition"] = sam_condition(C, z)
    out["fd_check_sam"] = grid_D_check(C, z, N=150, kind="sam")
    # phản ví dụ cho (D): kết luận không đối xứng (CAO rất rộng, THAP rất hẹp)
    C_bad = np.array([mf("trimf", [0, 0, 5], z), mf("trimf", [30, 50, 70], z),
                      mf("trimf", [52, 100, 100], z)])
    out["certificate_counterexample_consequents"] = certify_D(C_bad, z)
    # phản ví dụ KHẢ ĐẠT: bảng luật đơn điệu đưa w(x) đi qua vùng D giảm
    # (LAG = 0,2 ⇒ 0,8 khối lượng ở DUNG_HAN; d tăng chuyển khối lượng THAP → TRUNG_BINH
    #  trong khi 0,2 nằm cố định ở CAO)
    r_cx = np.array([[0, 2, 2], [1, 2, 2], [1, 2, 2]])
    cx = {"matrix": r_cx.tolist()}
    for nm, C_ in (("skewed_consequents", C_bad), ("project_consequents", C)):
        s = GridMamdani(z, C_, r_cx, "bounded_sum")
        Ml = np.array([mf(*cfg.MF_LAG[k], 0.2) for k in L_ORDER])
        path = [s.defuzz(s.weights([np.array([mf(*cfg.MF_D[k], d) for k in D_ORDER]), Ml]))
                for d in (0.0, 0.1, 0.2, 0.3, 0.45)]
        cx[nm] = {"viol": grid_violation(s, cfg.MF_D, cfg.MF_LAG), "path_lag0.2": path}
    out["reachable_counterexample"] = cx
    out["stress"] = stress()
    out["runtime_sec"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)

    L = ["W2-1 — ĐỊNH LÝ 1 (tầng Mamdani): kiểm chứng số", ""]
    L.append("1. Phân hoạch Ruspini có thứ tự")
    for k, v in out["partitions"].items():
        L.append(f"  {k:10s} |Σμ−1|max={v['max_abs_sum_minus_1']:.3g} minΣ={v['min_sum']:.3f} "
                 f"tăng F_j max={v['max_increase_of_cumulative']:.3g} → {v['ordered_ruspini']}")
    L.append(f"  Bảng luật: {r.tolist()} đơn điệu={out['rule_base']['monotone']} bước nhảy max={out['rule_base']['max_jump']}")
    L.append("")
    L.append("2. Bổ đề (L): w(x) tăng theo FOSD? (mức giảm đuôi lớn nhất; |Σw−1|)")
    for k, v in out["lemma_fosd"].items():
        L.append(f"  {k:30s} drop_d={v['max_tail_drop_d']:.4f} drop_LAG={v['max_tail_drop_lag']:.4f} "
                 f"|Σw−1|={v['max_abs_sum_w_minus_1']:.3f}")
    L.append("")
    L.append("3. Bảng 5.14 tái lập (tổng mức giảm trên lưới 0,01×0,01; thang 0–100) | cài đặt độc lập")
    for k, v in out["table_5_14"].items():
        L.append(f"  {k:30s} d {v['viol_d']:8.3f} LAG {v['viol_lag']:8.3f} | {v['ref_viol_d']:8.3f} {v['ref_viol_lag']:8.3f}")
    L.append(f"  t-norm min (Ruspini + tổng chặn): vi phạm {out['tnorm_min']['viol']}, FOSD {out['tnorm_min']['fosd']}")
    L.append(f"  Bảng luật không trơn {r_ns.tolist()}: tổng chặn {out['nonsmooth_rulebase']['bounded_sum']['viol']} "
             f"| cực đại {out['nonsmooth_rulebase']['max']['viol']}")
    L.append("  Biến thể phép hội / kéo theo / gộp (Ruspini; vi phạm theo d, LAG):")
    for v in out["implication_variants"]:
        L.append(f"    {v['rules']:9s} hội={v['conjunction']:7s} kéo theo={v['implication']:7s} "
                 f"gộp={v['aggregation']:11s} {v['viol_d']:9.3f} {v['viol_lag']:9.3f}")
    L.append("")
    c = out["certificate_maxmin_centroid"]
    L.append("4. Chứng nhận (D) — trọng tâm max–min, universe rời rạc bước 0,5")
    L.append(f"  {c['n_thresholds']} ngưỡng, {c['n_lines']} đường, {c['n_vertices']} đỉnh, "
             f"{c['n_cell_vertex_pairs']} cặp (ô, đỉnh); {c['sec']:.0f}s")
    L.append(f"  đạo hàm theo hướng nhỏ nhất: {c['min_directional_derivative']}  tại {c['argmin']}")
    L.append(f"  sai phân lưới N=300: {out['fd_check_maxmin']}")
    L.append(f"  universe mịn (20001 điểm), N=120: {out['fd_check_maxmin_fine_universe']}")
    L.append(f"  SAM giải tích: {out['sam_condition']}; sai phân {out['fd_check_sam']}")
    cb = out["certificate_counterexample_consequents"]
    L.append(f"  Phản ví dụ kết luận lệch: đạo hàm nhỏ nhất {cb['min_directional_derivative']} tại {cb['argmin']}")
    for nm in ("skewed_consequents", "project_consequents"):
        v = out["reachable_counterexample"][nm]
        L.append(f"  Phản ví dụ khả đạt, bảng luật {r_cx.tolist()}, {nm}: vi phạm {v['viol']}, "
                 f"đường d∈{{0,.1,.2,.3,.45}} tại LAG=0,2: {[round(t, 2) for t in v['path_lag0.2']]}")
    L.append("")
    L.append(f"5. Thử căng thẳng ngẫu nhiên: {json.dumps(out['stress'], ensure_ascii=False)}")
    L.append(f"Thời gian {out['runtime_sec']:.0f}s")
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
