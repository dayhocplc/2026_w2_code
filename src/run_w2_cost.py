# -*- coding: utf-8 -*-
"""
run_w2_cost.py — W2-5: chi phí tính toán của ba cách giải cùng một bài toán.

(A) Bài toán lồi có phạt (λ = 3e-3, γ = 1e-3, như G3):
    reparam_lbfgsb   tham số hoá lại, chỉ cận ≥ 0, L-BFGS-B       (đề xuất)
    lincon_trust     biến gốc (m, m⁺, m⁻), p ràng buộc tuyến tính, trust-constr
    lincon_slsqp     như trên, SLSQP
(B) Hợp lý cực đại không phạt (λ = γ = 0) — cùng họ mô hình với choquistic regression:
    reparam_lbfgsb_ml   L-BFGS-B, dung sai chặt
    choquistic_slsqp    tham số hoá gốc (độ đo chuẩn hoá, γ, β), SLSQP (không lồi)

Dữ liệu: POL252 mức mục (p = 25) và mô phỏng p ∈ {5, 10, 15, 20, 30, 40}, n = 500,
sinh từ một độ đo 2-cộng tính đơn điệu thưa ngẫu nhiên. Mỗi phương pháp được
đo thời gian (một luồng BLAS), giá trị mục tiêu, và vi phạm ràng buộc lớn nhất.
Bộ giải chậm bị bỏ ở p lớn hơn khi lần trước vượt ngân sách TIME_CAP giây.

Chạy:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 .venv/Scripts/python.exe src/run_w2_cost.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import run_g3_choquet_graph as G                                   # noqa: E402
from run_g3_choquet_graph import ChoquetGraph, orient              # noqa: E402
from ts_external import load_poland                                # noqa: E402
from w2_common import ChoquetLinCon, Choquistic, logloss_unpenalized  # noqa: E402

ROOT = os.path.dirname(SRC)
OUT_TXT = os.path.join(ROOT, "out", "reports", "w2_cost.txt")
OUT_JSON = os.path.join(ROOT, "out", "reports", "w2_cost.json")
SEED = 20260925
LAM = 3e-3
TIME_CAP = 900.0
P_SIM = [5, 10, 15, 20, 30, 40]


def penalized_objective(model, X, y, lam, gamma):
    """Giá trị mục tiêu (A) tính lại độc lập từ (a, m, m_ij): phạt L1 = λ‖m_ij‖₁
    (bằng λΣ(m⁺+m⁻) ở nghiệm có tính bù)."""
    eta = model.decision_function(X)
    return float(np.mean(np.logaddexp(0, eta) - y * eta) + lam * np.abs(model.mij_).sum()
                 + 0.5 * gamma * model.m_ @ model.m_)


def monotone_margin(model):
    neg = np.zeros(model.p)
    np.add.at(neg, model.I_, np.minimum(model.mij_, 0))
    np.add.at(neg, model.J_, np.minimum(model.mij_, 0))
    return float(np.min(model.m_ + neg))


def simulate(p, n=500, seed=SEED):
    rng = np.random.default_rng(seed + p)
    X = rng.uniform(0, 1, (n, p))
    I, J = ChoquetGraph.pairs(p)
    mij = np.zeros(len(I))
    k = min(len(I), 2 * p)
    idx = rng.choice(len(I), k, replace=False)
    mij[idx] = rng.normal(0, 0.5, k)
    m = rng.uniform(0.2, 1.0, p)
    neg = np.zeros(p)
    np.add.at(neg, I, np.minimum(mij, 0)); np.add.at(neg, J, np.minimum(mij, 0))
    m = np.maximum(m, -neg + 0.05)
    C = X @ m + np.minimum(X[:, I], X[:, J]) @ mij
    S = m.sum() + mij.sum()
    eta = 6 * (C / S - np.median(C / S))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    return X, y


def timed(fn):
    """Thời gian nhỏ nhất của tối đa 3 lần chạy (chỉ lặp khi lần đầu < 5 s)."""
    t = time.perf_counter()
    r = fn()
    best = time.perf_counter() - t
    if best < 5.0:
        for _ in range(2):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
    return r, best


def bench(X, y, skip, old=None):
    """old: các hàng đã có (chế độ W2_COST_REPARAM_ONLY) — chỉ tính lại hai hàng L-BFGS-B."""
    rows = {}
    # (A) phạt
    g, sec = timed(lambda: ChoquetGraph(LAM, True).fit(X, y))
    rows["reparam_lbfgsb"] = {"sec": sec, "obj": penalized_objective(g, X, y, LAM, G.GAMMA),
                              "margin": monotone_margin(g), "converged": g.converged_, "nit": g.al_iters}
    rows["reparam_lbfgsb"].update({"restarts": g.n_restarts_, "kkt_residual": g.kkt_residual_})
    for name, meth in (("lincon_trust", "trust-constr"), ("lincon_slsqp", "SLSQP")):
        if old is not None:
            rows[name] = old[name]
            continue
        if name in skip:
            rows[name] = None
            continue
        mdl, sec = timed(lambda: ChoquetLinCon(LAM, meth).fit(X, y))
        rows[name] = {"sec": sec, "obj": penalized_objective(mdl, X, y, LAM, G.GAMMA),
                      "margin": monotone_margin(mdl), "converged": mdl.converged_, "nit": mdl.nit_,
                      "max_violation": mdl.max_violation_}
    # (B) hợp lý cực đại
    old_gamma = G.GAMMA
    G.GAMMA = 0.0
    try:
        def fit_ml():
            mdl = ChoquetGraph(0.0, True)
            mdl.options = {"maxiter": 50000, "maxfun": 100000, "ftol": 1e-15, "gtol": 1e-10}
            return mdl.fit(X, y)
        g0, sec = timed(fit_ml)
    finally:
        G.GAMMA = old_gamma
    rows["reparam_lbfgsb_ml"] = {"sec": sec, "nll": logloss_unpenalized(g0, X, y),
                                 "margin": monotone_margin(g0), "converged": g0.converged_,
                                 "S": float(g0.m_.sum() + g0.mij_.sum()),
                                 "restarts": g0.n_restarts_, "kkt_residual": g0.kkt_residual_}
    if old is not None:
        rows["choquistic_slsqp"] = old["choquistic_slsqp"]
    elif "choquistic_slsqp" in skip:
        rows["choquistic_slsqp"] = None
    else:
        ch, sec = timed(lambda: Choquistic(maxiter=1000).fit(X, y))
        rows["choquistic_slsqp"] = {"sec": sec, "nll": logloss_unpenalized(ch, X, y),
                                    "converged": ch.converged_, "gamma": float(ch.gamma_),
                                    "max_violation": ch.max_violation_, "nit": ch.nit_}
    return rows


def main():
    t0 = time.time()
    out = {"lambda": LAM, "gamma": G.GAMMA, "time_cap": TIME_CAP, "runs": []}
    Q, y, _ = load_poland()
    Xp, _, _ = orient(Q / 4.0, y, Q / 4.0)
    cases = [("POL252", Xp, y)] + [(f"sim_p{p}", *simulate(p)) for p in sorted(P_SIM)]
    cases.sort(key=lambda c: c[1].shape[1])
    skip = set()
    prev = None
    if os.environ.get("W2_COST_REPARAM_ONLY"):
        with open(OUT_JSON, encoding="utf-8") as fh:
            prev = {r["case"]: r["rows"] for r in json.load(fh)["runs"]}
    for name, X, yy in cases:
        rows = bench(X, yy, skip, None if prev is None else prev[name])
        p = X.shape[1]
        out["runs"].append({"case": name, "n": int(len(yy)), "p": int(p),
                            "n_vars": int(1 + p + p * (p - 1)), "rows": rows})
        for k, v in rows.items():
            if v is not None and v["sec"] > TIME_CAP:
                skip.add(k)
        print(name, p, {k: (round(v["sec"], 2) if v else None) for k, v in rows.items()}, flush=True)
    out["runtime_sec"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)

    L = [f"W2-5 — CHI PHÍ TÍNH TOÁN (λ={LAM}, γ={G.GAMMA}; một luồng; ngân sách {TIME_CAP:.0f}s)", ""]
    L.append("(A) Bài toán lồi có phạt: thời gian (s) | mục tiêu − tốt nhất | biên đơn điệu nhỏ nhất")
    for r in out["runs"]:
        R = r["rows"]
        objs = [R[k]["obj"] for k in ("reparam_lbfgsb", "lincon_trust", "lincon_slsqp") if R[k]]
        best = min(objs)
        cells = []
        for k in ("reparam_lbfgsb", "lincon_trust", "lincon_slsqp"):
            v = R[k]
            cells.append(f"{k} " + ("bỏ (vượt ngân sách ở p nhỏ hơn)" if v is None else
                                    f"{v['sec']:8.2f}s Δobj={v['obj'] - best:.1e} biên={v['margin']:+.1e}"))
        L.append(f"  {r['case']:8s} p={r['p']:2d} biến={r['n_vars']:5d} | " + " | ".join(cells))
    L.append("")
    L.append("(B) Hợp lý cực đại (cùng họ choquistic): thời gian | NLL huấn luyện | S = γ")
    for r in out["runs"]:
        R = r["rows"]
        a = R["reparam_lbfgsb_ml"]; b = R["choquistic_slsqp"]
        s = f"  {r['case']:8s} p={r['p']:2d} | reparam {a['sec']:7.2f}s NLL={a['nll']:.6f} S={a['S']:.2f}"
        s += (" | choquistic bỏ" if b is None else
              f" | choquistic {b['sec']:8.2f}s NLL={b['nll']:.6f} γ={b['gamma']:.2f} hội tụ={b['converged']}")
        L.append(s)
    L.append(f"Thời gian {out['runtime_sec']:.0f}s")
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
