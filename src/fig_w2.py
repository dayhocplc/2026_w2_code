# -*- coding: utf-8 -*-
r"""
fig_w2.py — Hình của bản thảo W2 (đơn điệu theo cấu trúc, hệ tổng hợp mờ hai tầng).

Hình 1  Sơ đồ hai tầng và vị trí của Định lý 1, Định lý 2, Hệ quả 1.
Hình 2  Tầng Mamdani: (a) risk theo d tại vài mức LAG, Ruspini + cực đại (vi phạm);
        (b) như (a) với Ruspini + tổng chặn; (c)(d) đuôi P(c ≥ k) của w(x) theo d.
Hình 3  Điều kiện (D) trên đơn hình: đạo hàm theo hướng nhỏ nhất của D — (a) tập kết
        luận của dự án; (b) tập kết luận lệch (phản ví dụ).
Hình 5  Chi phí: thời gian giải theo số biến (w2_cost.json).
Hình 4  ΔAUC so với Choquet đơn điệu đề xuất trên 4 bộ dữ liệu (w2_external.json).

    .venv/Scripts/python.exe src/fig_w2.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg                                                   # noqa: E402
from figstyle import C, arrow, blank_axes, box, save, setup           # noqa: E402
from run_fis_bypass import mf                                          # noqa: E402
from run_w2_mamdani_theorem import (D_ORDER, L_ORDER, GridMamdani,     # noqa: E402
                                    cons_matrix, rules_array)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "out", "reports")


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    blank_axes(ax)
    ax.set_xlim(-0.02, 1.02)
    box(ax, 0.00, 0.66, 0.15, 0.22, "Questionnaire\nitems $q_1,\\dots,q_{25}$", fc="#F2F2F2", fs=7.5)
    box(ax, 0.00, 0.30, 0.15, 0.22, "Age of child,\nmilestone norms", fc="#F2F2F2", fs=7.5)
    box(ax, 0.20, 0.47, 0.16, 0.24, "Domain features\n$d_i\\in[0,1]$\n$\\mathrm{LAG}_i\\in[0,3]$", fs=7.5)
    box(ax, 0.42, 0.30, 0.25, 0.58, "Layer 1: Mamdani FIS\nper domain\n\nordered Ruspini inputs\nproduct conjunction\nbounded-sum aggregation\nclipping, centroid",
        fc="#E8F1F8", ec=C["xanh"], fs=7.5)
    box(ax, 0.74, 0.30, 0.25, 0.58, "Layer 2: 2-additive\nChoquet logistic\n\n$m_i = s_i + \\Sigma_j\\, m^{-}_{ij}$\n$s,\\ m^{+},\\ m^{-} \\geq 0$\nL-BFGS-B (convex)",
        fc="#FBEFE6", ec=C["donhat"], fs=7.5)
    arrow(ax, (0.15, 0.77), (0.20, 0.65))
    arrow(ax, (0.15, 0.41), (0.20, 0.53))
    arrow(ax, (0.36, 0.59), (0.42, 0.59))
    arrow(ax, (0.67, 0.59), (0.74, 0.59))
    ax.text(0.705, 0.63, "$r_i$", ha="center", fontsize=8)
    ax.text(0.545, 0.14, "Theorem 1: $r_i$ nondecreasing\nin $d_i$ and $\\mathrm{LAG}_i$",
            ha="center", fontsize=7.5, color=C["xanh"])
    ax.text(0.865, 0.14, "Theorem 2: $P(Y=1)$\nnondecreasing in every $r_i$",
            ha="center", fontsize=7.5, color=C["donhat"])
    ax.text(0.50, 0.00, "Corollary 1: the composite score is nondecreasing in every input",
            ha="center", fontsize=8, style="italic")
    save(fig, "W2_fig1_architecture")


def fig2():
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    Cm = cons_matrix(z)
    r = rules_array()
    dg = np.linspace(0, 1, 401)
    Md = np.array([mf(*cfg.MF_D[k], dg) for k in D_ORDER]).T
    lags = [0.0, 0.5, 1.2, 2.0]
    styles = ["-", "--", "-.", ":"]
    cols = [C["xanh"], C["donhat"], C["luc"], C["tim"]]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for col, sn in enumerate(("max", "bounded_sum")):
        s = GridMamdani(z, Cm, r, snorm=sn)
        for lag, ls, cc in zip(lags, styles, cols):
            Ml = np.array([mf(*cfg.MF_LAG[k], lag) for k in L_ORDER])
            W = np.array([s.weights([Md[i], Ml]) for i in range(len(dg))])
            y = np.array([s.defuzz(w) for w in W])
            axs[0, col].plot(dg, y, ls=ls, color=cc, label=f"LAG = {lag:g}")
            if lag == 0.5:
                tail2 = W[:, 1:].sum(1); tail3 = W[:, 2]
                axs[1, col].plot(dg, tail2, color=C["xanh"], ls="-", label=r"$w_2+w_3$")
                axs[1, col].plot(dg, tail3, color=C["donhat"], ls="--", label=r"$w_3$")
                axs[1, col].plot(dg, W.sum(1), color=C["xam"], ls=":", label=r"$\sum_c w_c$")
        axs[0, col].set_ylabel("domain risk $r$ (0–100)")
        axs[1, col].set_xlabel("deficit score $d$")
        axs[1, col].set_ylabel("consequent strength (LAG = 0.5)")
        axs[1, col].set_ylim(-0.03, 1.05)
    axs[0, 0].set_title("(a) Ruspini inputs + max aggregation", loc="left")
    axs[0, 1].set_title("(b) Ruspini inputs + bounded sum", loc="left")
    axs[1, 0].set_title("(c) firing tails, max", loc="left")
    axs[1, 1].set_title("(d) firing tails, bounded sum", loc="left")
    axs[0, 0].legend(loc="upper left")
    axs[1, 1].legend(loc="center right")
    fig.tight_layout()
    save(fig, "W2_fig2_mamdani")


def _min_dir_derivative(Cc, z, N=160, h=1e-6):
    pts, vals = [], []
    V = [np.array([-1, 1, 0.]), np.array([0, -1, 1.])]

    def D(w):
        g = np.max(np.minimum(w[:, None], Cc), 0)
        return (z * g).sum() / g.sum()
    for i in range(N + 1):
        for j in range(N + 1 - i):
            w = np.array([i / N, j / N, 1 - i / N - j / N])
            best = np.inf
            for v in V:
                if (w + h * v).min() >= 0:                  # sai phân tiến
                    best = min(best, (D(w + h * v) - D(w)) / h)
                elif (w - h * v).min() >= 0:                # sai phân lùi ở biên
                    best = min(best, (D(w) - D(w - h * v)) / h)
            pts.append(w); vals.append(best)
    return np.array(pts), np.array(vals)


def fig3():
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    Cp = cons_matrix(z)
    Cb = np.array([mf("trimf", [0, 0, 5], z), mf("trimf", [30, 50, 70], z),
                   mf("trimf", [52, 100, 100], z)])
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.tri as mtri
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    data = [_min_dir_derivative(Cc, z) for Cc in (Cp, Cb)]
    # thang màu cắt ở ±40, tâm 0: giá trị lớn gần đỉnh LOW không chi phối màu
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    for ax, (P, V), ttl in zip(axs, data, ("(a) screening-model consequents", "(b) skewed consequents")):
        ok = np.isfinite(V)
        x = P[ok, 1] + 0.5 * P[ok, 2]
        y = P[ok, 2] * np.sqrt(3) / 2
        Vc = np.clip(V[ok], -39.99, 39.99)
        tri = mtri.Triangulation(x, y)
        sc = ax.tricontourf(tri, Vc, levels=np.linspace(-40, 40, 17), cmap="RdBu", norm=norm, extend="both")
        if (V[ok] < 0).any():
            ax.tricontour(tri, V[ok], levels=[0.0], colors="k", linewidths=1.0)
            ax.text(0.25, 0.30, "D decreasing\n(condition (D) fails)", fontsize=7, ha="center")
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="k", lw=0.8)
        ax.text(-0.04, -0.07, "LOW", fontsize=7); ax.text(0.93, -0.07, "MEDIUM", fontsize=7)
        ax.text(0.45, np.sqrt(3) / 2 + 0.03, "HIGH", fontsize=7)
        ax.set_title(f"{ttl}\nmin = {np.min(V[ok]):.1f}", loc="left", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(sc, ax=axs, shrink=0.75, ticks=[-40, -20, 0, 20, 40])
    cb.ax.set_yticklabels(["≤ −40", "−20", "0", "20", "≥ 40"])
    cb.set_label(r"$\min_k\ \partial D/\partial(e_{k+1}-e_k)$")
    save(fig, "W2_fig3_defuzzifier")


def fig4():
    fn = os.path.join(REP, "w2_cost.json")
    if not os.path.exists(fn):
        print("  bỏ Hình 4 (chưa có w2_cost.json)")
        return
    J = json.load(open(fn, encoding="utf-8"))
    runs = sorted([r for r in J["runs"] if r["case"].startswith("sim")], key=lambda r: r["p"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    spec = [("reparam_lbfgsb", "reparametrized, L-BFGS-B", C["xanh"], "-", "o"),
            ("lincon_trust", "linear constraints, trust-constr", C["donhat"], "--", "s"),
            ("lincon_slsqp", "linear constraints, SLSQP", C["luc"], "-.", "^")]
    for k, lab, cc, ls, mk in spec:
        pts = [(r["n_vars"], r["rows"][k]["sec"]) for r in runs if r["rows"][k]]
        if pts:
            axs[0].plot(*zip(*pts), ls=ls, marker=mk, color=cc, label=lab, ms=4)
    spec2 = [("reparam_lbfgsb_ml", "reparametrized ML, L-BFGS-B", C["xanh"], "-", "o"),
             ("choquistic_slsqp", "choquistic (normalized), SLSQP", C["tim"], "--", "D")]
    for k, lab, cc, ls, mk in spec2:
        pts = [(r["n_vars"], r["rows"][k]["sec"]) for r in runs if r["rows"][k]]
        if pts:
            axs[1].plot(*zip(*pts), ls=ls, marker=mk, color=cc, label=lab, ms=4)
    for ax, t in zip(axs, ("(a) penalized convex problem", "(b) unpenalized maximum likelihood")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of parameters"); ax.set_ylabel("wall-clock time (s)")
        ax.set_title(t, loc="left"); ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "W2_fig5_cost")


def fig5():
    fn = os.path.join(REP, "w2_external.json")
    if not os.path.exists(fn):
        print("  bỏ Hình 5 (chưa có w2_external.json)")
        return
    J = json.load(open(fn, encoding="utf-8"))
    dsets = ["POL252", "PARK195", "BCW683", "HEART303"]
    lab = {"choquet_unconstrained": "2-additive, unconstrained", "additive_monotone": "additive, monotone",
           "choquistic_ml": "choquistic (ML)", "lgbm_monotone": "LightGBM, monotone",
           "linear_l1_logistic": "L1 logistic", "pairwise_l1_logistic": "L1 logistic + pairs"}
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
    for ax, ds in zip(axs, dsets):
        comps = J["datasets"][ds]["comparisons"]
        for yi, c in enumerate(comps):
            lo, hi = c["ci95"]
            col = C["donhat"] if c["reject"] else C["xam"]
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.2)
            ax.plot(c["delta"], yi, "o", color=col, ms=3.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(ds, fontsize=8)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([lab[c["b"]] for c in comps], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    axs[0].invert_yaxis()
    fig.supxlabel("ΔAUC = monotone 2-additive Choquet − comparator (Nadeau–Bengio 95% CI)", fontsize=8)
    fig.tight_layout()
    save(fig, "W2_fig4_auc")


if __name__ == "__main__":
    setup()
    fig1(); fig2(); fig3(); fig4(); fig5()
