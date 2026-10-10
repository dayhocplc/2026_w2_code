# -*- coding: utf-8 -*-
"""
run_w2_choquet_equivalence.py — W2-2: kiểm chứng số cho Định lý 2 (tầng Choquet).

Định lý 2 (chứng minh ở manuscript/W2/W2_proofs.md). Với độ đo 2-cộng tính dạng
Möbius (m_i, m_ij), các mệnh đề sau tương đương:
  (G)  Grabisch/Chateauneuf–Jaffray: μ(T ∪ {i}) − μ(T) = m_i + Σ_{j∈T} m_ij ≥ 0
       với mọi i và mọi T ⊆ N∖{i}                       (2^{p−1} bất đẳng thức / nút)
  (C)  dạng đóng: m_i + Σ_{j≠i} min(m_ij, 0) ≥ 0         (p bất đẳng thức)
  (R)  tồn tại s ≥ 0, m⁺ ≥ 0, m⁻ ≥ 0 với m_ij = m⁺_ij − m⁻_ij, m_i = s_i + Σ_j m⁻_ij.
Hơn nữa với phạt λ Σ(m⁺ + m⁻), λ > 0, mọi nghiệm tối ưu thoả m⁺_ij m⁻_ij = 0, nên
bài toán (R) tương đương CHÍNH XÁC bài toán gốc có phạt λ‖m_ij‖₁ dưới (G).

Script:
  1. p = 2..12, 20 000 độ đo ngẫu nhiên (nửa đơn điệu nửa không, gồm ca biên):
     (G) vét cạn ⇔ (C) ⇔ (R) (dựng s từ phần âm);
  2. định nghĩa Choquet bằng sắp xếp + μ(A) = Σ_{B⊆A} m_B trùng công thức Möbius;
  3. độ đo vi phạm (C) ⇒ tìm được x ≤ x' với C(x) > C(x') (tính cần);
  4. nghiệm G3 (POL252, mức mục, λ = 3e-3): kiểm tra bù m⁺ m⁻ = 0 và (G) trên nghiệm
     bằng tìm T xấu nhất (T = {j : m_ij < 0}).

Chạy:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/run_w2_choquet_equivalence.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from run_g3_choquet_graph import ChoquetGraph, orient                  # noqa: E402
from ts_external import load_poland                                   # noqa: E402

ROOT = os.path.dirname(SRC)
OUT_TXT = os.path.join(ROOT, "out", "reports", "w2_choquet_equivalence.txt")
OUT_JSON = os.path.join(ROOT, "out", "reports", "w2_choquet_equivalence.json")
SEED = 20260925
TOL = 1e-12


def grabisch_bruteforce(m, M):
    """min_{i, T⊆N∖{i}} m_i + Σ_{j∈T} M_ij (vét cạn)."""
    p = len(m)
    worst = np.inf
    for i in range(p):
        others = [j for j in range(p) if j != i]
        for r in range(len(others) + 1):
            for T in itertools.combinations(others, r):
                worst = min(worst, m[i] + sum(M[i, j] for j in T))
    return worst


def closed_form(m, M):
    return float(np.min(m + np.minimum(M, 0).sum(1)))      # M đối xứng, đường chéo 0


def reparam_feasible(m, M):
    """(R): chọn m⁺ = max(M,0), m⁻ = max(−M,0) (phân tích tối thiểu); s = m − Σ m⁻.
    (R) khả thi ⇔ tồn tại phân tích nào đó với s ≥ 0; phân tích tối thiểu cho s lớn
    nhất nên (R) ⇔ s_min ≥ 0."""
    s = m - np.maximum(-M, 0).sum(1)
    return float(s.min())


def capacity(m, M, A):
    A = list(A)
    return sum(m[i] for i in A) + sum(M[i, j] for i, j in itertools.combinations(A, 2))


def choquet_sort(x, m, M):
    """Định nghĩa gốc: C(x) = Σ_k (x_(k) − x_(k−1)) μ({(k),…,(p)})."""
    order = np.argsort(x)
    xs = x[order]
    prev, tot = 0.0, 0.0
    for k in range(len(x)):
        tot += (xs[k] - prev) * capacity(m, M, order[k:])
        prev = xs[k]
    return tot


def choquet_mobius(x, m, M):
    p = len(x)
    return float(m @ x + sum(M[i, j] * min(x[i], x[j]) for i, j in itertools.combinations(range(p), 2)))


def random_measure(rng, p, kind):
    m = rng.uniform(0, 1, p)
    M = np.triu(rng.normal(0, 0.4, (p, p)), 1)
    M[rng.uniform(size=(p, p)) < 0.4] = 0.0
    M = np.triu(M, 1); M = M + M.T
    neg = np.minimum(M, 0).sum(1)
    if kind == "monotone":
        m = np.maximum(m, -neg)                             # đẩy về khả thi
    elif kind == "boundary":
        m = -neg                                            # đúng biên (ràng buộc chặt)
        i = rng.integers(p)
        m[i] = max(m[i], 0)
    # kind == "free": để nguyên (thường vi phạm)
    return m, M


def main():
    rng = np.random.default_rng(SEED)
    out = {"equivalence": {}, "choquet_formula": {}, "necessity": {}}
    mism = 0; total = 0
    for p in range(2, 13):
        n = 20000 // 11 if p <= 8 else 300                  # p lớn: vét cạn 2^{p−1}·p đắt
        agree = 0; n_mono = 0
        for t in range(n):
            kind = ("monotone", "boundary", "free")[t % 3]
            m, M = random_measure(rng, p, kind)
            g = grabisch_bruteforce(m, M)
            c = closed_form(m, M)
            r = reparam_feasible(m, M)
            ok = (abs(g - c) < 1e-10) and ((g >= -TOL) == (c >= -TOL) == (r >= -TOL))
            agree += ok; n_mono += (g >= -TOL)
            total += 1; mism += (not ok)
        out["equivalence"][p] = {"n": n, "agree": agree, "n_monotone": n_mono}
    out["equivalence_total"] = {"n": total, "mismatches": mism}

    # công thức Choquet
    err = 0.0
    for t in range(3000):
        p = int(rng.integers(2, 8))
        m, M = random_measure(rng, p, "free")
        x = rng.uniform(0, 1, p)
        err = max(err, abs(choquet_sort(x, m, M) - choquet_mobius(x, m, M)))
    out["choquet_formula"] = {"max_abs_err_sort_vs_mobius": err, "n": 3000}

    # tính cần: vi phạm (C) tại nút i ⇒ tăng x_i trên T xấu nhất làm C giảm
    found = 0; tried = 0
    for t in range(2000):
        p = int(rng.integers(2, 8))
        m, M = random_measure(rng, p, "free")
        viol = m + np.minimum(M, 0).sum(1)
        if viol.min() >= 0:
            continue
        tried += 1
        i = int(np.argmin(viol))
        T = [j for j in range(p) if j != i and M[i, j] < 0]
        x = np.zeros(p); x[T] = 1.0                         # x = 1_T
        x2 = x.copy(); x2[i] = 1.0                          # x' = 1_{T∪{i}} ≥ x
        found += choquet_mobius(x2, m, M) < choquet_mobius(x, m, M) - 1e-12
    out["necessity"] = {"n_nonmonotone_measures": tried, "n_with_decreasing_pair": found}

    # nghiệm G3 thật
    Q, y, _ = load_poland()
    Xf, _, _ = orient(Q / 4.0, y, Q / 4.0)
    g = ChoquetGraph(3e-3, True).fit(Xf, y)
    P = len(g.I_)
    mp = g.theta_[1 + g.p:1 + g.p + P]; mm = g.theta_[1 + g.p + P:]
    Mfull = np.zeros((g.p, g.p)); Mfull[g.I_, g.J_] = g.mij_; Mfull = Mfull + Mfull.T
    worst_T = closed_form(g.m_, Mfull)
    out["g3_solution"] = {"max_mplus_times_mminus": float(np.max(mp * mm)),
                          "n_both_positive": int(np.sum((mp > 1e-10) & (mm > 1e-10))),
                          "closed_form_margin": worst_T,
                          "min_s": float(np.min(g.m_ - (np.zeros(g.p) + np.bincount(g.I_, mm, g.p) + np.bincount(g.J_, mm, g.p)))),
                          "n_edges": int(np.sum(np.abs(g.mij_) > 1e-6)), "converged": g.converged_}

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    L = ["W2-2 — ĐỊNH LÝ 2: (G) Grabisch ⇔ (C) dạng đóng ⇔ (R) tham số hoá lại", ""]
    for p, v in out["equivalence"].items():
        L.append(f"  p={p:2d}: {v['agree']}/{v['n']} khớp ({v['n_monotone']} đơn điệu)")
    L.append(f"  Tổng: {out['equivalence_total']}")
    L.append(f"  Choquet sắp xếp vs Möbius: {out['choquet_formula']}")
    L.append(f"  Tính cần: {out['necessity']}")
    L.append(f"  Nghiệm G3 (POL252, λ=3e-3): {out['g3_solution']}")
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
