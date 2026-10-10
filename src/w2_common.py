# -*- coding: utf-8 -*-
"""
w2_common.py — bộ giải dùng chung cho bài W2 (so sánh W2-3/W2-4/W2-5).

Ba cách giải CÙNG họ mô hình logistic Choquet 2-cộng tính đơn điệu:
  * ChoquetGraph (run_g3_choquet_graph): tham số hoá lại, chỉ còn cận ≥ 0, L-BFGS-B
    — đề xuất của bài.
  * ChoquetLinCon: cùng bài toán lồi nhưng ở biến gốc (m, m⁺, m⁻) với p ràng buộc
    tuyến tính m_i − Σ_j m⁻_ij ≥ 0, giải bằng trust-constr hoặc SLSQP.
  * Choquistic: tham số hoá GỐC của Tehrani et al. (2012) giới hạn ở 2-cộng tính —
    độ đo chuẩn hoá (Σ Möbius = 1), P(y=1) = σ(γ (C_μ(x) − β)), γ ≥ 0, β ∈ [0, 1];
    hợp lý cực đại không phạt, SLSQP. Bài toán KHÔNG lồi (tích γ·μ). Đây là bản cài
    lại của chúng tôi cho lớp 2-cộng tính, không phải mã gốc của tác giả.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

from run_g3_choquet_graph import GAMMA, ChoquetGraph


def _design(X):
    p = X.shape[1]
    I, J = ChoquetGraph.pairs(p)
    Mn = np.minimum(X[:, I], X[:, J])
    A = np.zeros((p, len(I)))
    A[I, np.arange(len(I))] = 1
    A[J, np.arange(len(I))] = 1
    return I, J, Mn, A


class ChoquetLinCon:
    """min  mean logloss(a + Xm + Mn(m⁺ − m⁻)) + λΣ(m⁺+m⁻) + γ/2‖m‖²
       s.t. m_i − Σ_j m⁻_ij ≥ 0, m⁺, m⁻ ≥ 0 (m_i ≥ 0 suy ra)."""

    def __init__(self, lam, method="trust-constr", maxiter=5000):
        self.lam, self.method, self.maxiter = lam, method, maxiter

    def fit(self, X, y):
        n, p = X.shape
        I, J, Mn, A = _design(X)
        P = len(I)
        self.p, self.I_, self.J_ = p, I, J
        yf = y.astype(float)
        lam = self.lam

        def obj(t):
            a, m, mp, mm = t[0], t[1:1 + p], t[1 + p:1 + p + P], t[1 + p + P:]
            eta = a + X @ m + Mn @ (mp - mm)
            f = np.mean(np.logaddexp(0, eta) - yf * eta) + lam * (mp.sum() + mm.sum()) + 0.5 * GAMMA * (m @ m)
            r = (expit(eta) - yf) / n
            g = np.concatenate([[r.sum()], X.T @ r + GAMMA * m, Mn.T @ r + lam, -(Mn.T @ r) + lam])
            return f, g

        nv = 1 + p + 2 * P
        G = np.zeros((p, nv))
        G[:, 1:1 + p] = np.eye(p)
        G[:, 1 + p + P:] = -A
        t0 = np.zeros(nv); t0[1:1 + p] = 0.1
        bounds = [(None, None)] + [(0, None)] * (p + 2 * P)
        tic = time.perf_counter()
        if self.method == "trust-constr":
            from scipy.optimize import Bounds
            lb = np.r_[-np.inf, np.zeros(p + 2 * P)]
            res = minimize(obj, t0, jac=True, method="trust-constr",
                           bounds=Bounds(lb, np.full(nv, np.inf)),
                           constraints=[LinearConstraint(G, 0, np.inf)],
                           options={"maxiter": self.maxiter, "gtol": 1e-8, "xtol": 1e-10})
        else:
            res = minimize(obj, t0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": lambda t: G @ t, "jac": lambda t: G}],
                           options={"maxiter": self.maxiter, "ftol": 1e-12})
        self.sec_ = time.perf_counter() - tic
        t = res.x
        self.nit_ = int(getattr(res, "nit", 0))
        self.converged_ = bool(res.success)
        self.a_, self.m_ = t[0], t[1:1 + p]
        mp, mm = t[1 + p:1 + p + P], t[1 + p + P:]
        self.mij_ = mp - mm
        self.fun_ = float(obj(t)[0])
        self.max_violation_ = float(max(0.0, np.max(-(G @ t)), np.max(-t[1:])))
        return self

    def decision_function(self, X):
        I, J = self.I_, self.J_
        return self.a_ + X @ self.m_ + np.minimum(X[:, I], X[:, J]) @ self.mij_


class Choquistic:
    """Choquistic regression 2-cộng tính, tham số hoá gốc (độ đo chuẩn hoá)."""

    def __init__(self, maxiter=1000, gamma0=5.0, seed=0):
        self.maxiter, self.gamma0, self.seed = maxiter, gamma0, seed

    def fit(self, X, y):
        n, p = X.shape
        I, J, Mn, A = _design(X)
        P = len(I)
        self.p, self.I_, self.J_ = p, I, J
        yf = y.astype(float)
        # t = (γ, β, m(p), m⁺(P), m⁻(P))
        nv = 2 + p + 2 * P

        def unpack(t):
            return t[0], t[1], t[2:2 + p], t[2 + p:2 + p + P], t[2 + p + P:]

        def obj(t):
            gam, beta, m, mp, mm = unpack(t)
            C = X @ m + Mn @ (mp - mm)
            eta = gam * (C - beta)
            f = np.mean(np.logaddexp(0, eta) - yf * eta)
            r = (expit(eta) - yf) / n
            g = np.empty(nv)
            g[0] = r @ (C - beta)
            g[1] = -gam * r.sum()
            g[2:2 + p] = gam * (X.T @ r)
            g[2 + p:2 + p + P] = gam * (Mn.T @ r)
            g[2 + p + P:] = -gam * (Mn.T @ r)
            return f, g

        Gin = np.zeros((p, nv)); Gin[:, 2:2 + p] = np.eye(p); Gin[:, 2 + p + P:] = -A
        geq = np.zeros(nv); geq[2:2 + p] = 1; geq[2 + p:2 + p + P] = 1; geq[2 + p + P:] = -1
        rng = np.random.default_rng(self.seed)
        t0 = np.zeros(nv)
        w = rng.dirichlet(np.ones(p)) if self.seed else np.full(p, 1.0 / p)
        t0[0], t0[2:2 + p] = self.gamma0, w
        t0[1] = float(np.median(X @ w))
        bounds = [(0, None), (0, 1)] + [(0, None)] * (p + 2 * P)
        tic = time.perf_counter()
        res = minimize(obj, t0, jac=True, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": lambda t: Gin @ t, "jac": lambda t: Gin},
                                    {"type": "eq", "fun": lambda t: np.array([geq @ t - 1.0]),
                                     "jac": lambda t: geq[None]}],
                       options={"maxiter": self.maxiter, "ftol": 1e-12})
        self.sec_ = time.perf_counter() - tic
        t = res.x
        self.nit_ = int(res.nit)
        self.converged_ = bool(res.success)
        gam, beta, m, mp, mm = unpack(t)
        self.gamma_, self.beta_, self.m_, self.mij_ = gam, beta, m, mp - mm
        self.fun_ = float(obj(t)[0])
        self.max_violation_ = float(max(0.0, np.max(-(Gin @ t)), abs(geq @ t - 1)))
        return self

    def decision_function(self, X):
        I, J = self.I_, self.J_
        return self.gamma_ * (X @ self.m_ + np.minimum(X[:, I], X[:, J]) @ self.mij_ - self.beta_)


def logloss_unpenalized(model, X, y):
    eta = model.decision_function(X)
    return float(np.mean(np.logaddexp(0, eta) - y * eta))
