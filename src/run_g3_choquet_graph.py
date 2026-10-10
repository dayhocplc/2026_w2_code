# -*- coding: utf-8 -*-
"""
run_g3_choquet_graph.py — Hướng 3: học ĐỒ THỊ TƯƠNG TÁC THƯA, ĐƠN ĐIỆU của
một độ đo mờ 2-cộng tính (Choquet) từ dữ liệu Q-CHAT-25 Ba Lan (n = 252).

Mô hình (dạng Möbius, x ∈ [0,1]^p, đã định hướng theo dấu AUC trong fold huấn luyện):
    C(x) = Σ_i m_i x_i + Σ_{i<j} m_ij min(x_i, x_j)
    P(y=1) = σ(a + C(x))                     (thang đo tự do — xem ghi chú dưới)
Đơn điệu (điều kiện cần và đủ cho 2-cộng tính):
    m_i ≥ 0,   m_i + Σ_j min(m_ij, 0) ≥ 0
Tuyến tính hoá: m_ij = m⁺_ij − m⁻_ij, m⁺, m⁻ ≥ 0,  m_i − Σ_j m⁻_ij ≥ 0.
Mục tiêu: log-loss trung bình + λ Σ(m⁺ + m⁻) + (γ/2) Σ m_i²  (γ = 1e-3 nhỏ, chỉ
để chống phân kỳ thang đo khi dữ liệu gần tách được).
Giải: đổi biến m_i = s_i + Σ_j m⁻_ij (s ≥ 0) ⇒ mọi ràng buộc thành cận ≥ 0, mô hình
thành logistic hệ số không âm trên đặc trưng (x_i, min(x_i,x_j), max(x_i,x_j));
L-BFGS-B có cận; kiểm tra lại ràng buộc gốc ở nghiệm (min_monotone_margin ≥ 0).

Vì sao để thang đo tự do thay vì Σm = 1:
  σ(a + b·C_norm(x)) với C_norm chuẩn hoá (Σm = 1) và b ≥ 0 tương đương CHÍNH XÁC
  với σ(a + C(x)) với m không chuẩn hoá, b = S = Σ m_i + Σ m_ij = C(1,…,1) ≥ 0
  (đơn điệu ⇒ S ≥ 0). Bỏ ràng buộc đẳng thức và tham số b giúp bài toán lồi
  (log-loss lồi theo m, ràng buộc tuyến tính) — nghiệm toàn cục duy nhất về
  thực chất. Báo cáo φ, I sau khi chia cho S nên vẫn là độ đo chuẩn hoá.

Chạy:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/run_g3_choquet_graph.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import config as cfg                                            # noqa: E402
from ts_aggregate import choquet_2additive                      # noqa: E402
from ts_external import (load_poland, QCHAT_ITEMS, QCHAT10_ITEMS,  # noqa: E402
                         domain_scores, to_rows, aggregate)
from p3_common import nadeau_bengio, benjamini_hochberg         # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20260827
ROOT = os.path.dirname(SRC)
OUT_TXT = os.path.join(ROOT, "out", "reports", "g3_choquet_graph.txt")
OUT_JSON = os.path.join(ROOT, "out", "reports", "g3_choquet_graph.json")
OUT_FIG = os.path.join(ROOT, "out", "figures", "g3_stable_graph.png")
LAMBDAS = [1e-2, 3e-3, 1e-3, 3e-4]   # lớn → nhỏ (warm start)
GAMMA = 1e-3
STABLE = 0.70
ZERO = 1e-6
INNER = 3        # số fold CV trong (máy dùng chung, giới hạn 2 tiến trình)
KKT_RESTARTS = 20  # số lần khởi động lại L-BFGS-B tối đa khi KKT chưa đạt (xem fit)


# =====================================================================
# Bộ giải: logistic Choquet 2-cộng tính, thưa + đơn điệu
# =====================================================================
class ChoquetGraph:
    n_fits_total = 0          # bộ đếm toàn tiến trình: số lần fit / số lần phải khởi động lại
    n_restarted_total = 0

    def __init__(self, lam: float, monotone: bool = True, edges: bool = True):
        self.lam, self.monotone, self.edges = lam, monotone, edges

    @staticmethod
    def pairs(p):
        P = list(itertools.combinations(range(p), 2))
        I = np.array([a for a, _ in P], int)
        J = np.array([b for _, b in P], int)
        return I, J

    def _design(self, X):
        return np.minimum(X[:, self.I_], X[:, self.J_])

    def fit(self, X, y, theta0=None):
        """Tham số hoá lại để mọi ràng buộc thành CẬN ĐƠN GIẢN:
            m_i = s_i + Σ_j m⁻_ij,  s_i ≥ 0,  m⁺, m⁻ ≥ 0.
        Khi đó  η = a + Σ s_i x_i + Σ m⁺_ij min(x_i,x_j) + Σ m⁻_ij max(x_i,x_j)
        (vì x_i + x_j − min = max) — chính là dạng khai triển của dự án, và
        đơn điệu ⇔ mọi hệ số của (x, min, max) không âm. Ánh xạ (s,m⁺,m⁻) ↔
        (m,m⁺,m⁻) là song ánh trên miền khả thi nên bài toán tương đương
        CHÍNH XÁC; giải một lần bằng L-BFGS-B có cận, không cần Lagrange."""
        n, p = X.shape
        self.p = p
        self.I_, self.J_ = self.pairs(p)
        P = len(self.I_)
        Mn = self._design(X)
        Mx = np.maximum(X[:, self.I_], X[:, self.J_])
        A = np.zeros((p, P))
        A[self.I_, np.arange(P)] = 1
        A[self.J_, np.arange(P)] = 1
        nv = 1 + p + 2 * P
        if theta0 is None:
            theta0 = np.zeros(nv)
            theta0[1:1 + p] = 0.1
        mono = self.monotone
        lb = 0.0 if mono else None
        ebound = (0.0, None) if self.edges else (0.0, 0.0)
        bounds = [(None, None)] + [(lb, None)] * p + [ebound] * (2 * P)
        lam, gamma = self.lam, GAMMA
        yf = y.astype(float)
        # đơn điệu: biến thứ 2 là s và cột của m⁻ là +max;
        # không ràng buộc: biến thứ 2 là m và cột của m⁻ là −min.
        Fneg = Mx if mono else -Mn

        def obj(t):
            a, u, mp, mm = t[0], t[1:1 + p], t[1 + p:1 + p + P], t[1 + p + P:]
            eta = a + X @ u + Mn @ mp + Fneg @ mm
            m = u + A @ mm if mono else u
            f = np.mean(np.logaddexp(0, eta) - yf * eta)
            f += lam * (mp.sum() + mm.sum()) + 0.5 * gamma * (m @ m)
            r = (expit(eta) - yf) / n
            g = np.empty_like(t)
            g[0] = r.sum()
            g[1:1 + p] = X.T @ r + gamma * m
            g[1 + p:1 + p + P] = Mn.T @ r + lam
            g[1 + p + P:] = Fneg.T @ r + lam + (gamma * (A.T @ m) if mono else 0.0)
            return f, g

        opts = getattr(self, "options", {"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-6})
        res = minimize(obj, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
        # [W2-5] L-BFGS-B có thể dừng giả theo tiêu chí "giảm tương đối của f ≤ ftol" khi
        # gradient chiếu còn lớn (gặp ở mô phỏng p = 30). Kiểm tra KKT bằng gradient chiếu
        # và khởi động lại từ điểm cuối cho tới khi đạt, tối đa KKT_RESTARTS lần.
        lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])

        def pgrad(t):
            g = obj(t)[1]
            pg = g.copy()
            at_lo = t <= lo + 1e-12
            at_hi = t >= hi - 1e-12
            pg[at_lo] = np.minimum(g[at_lo], 0.0)
            pg[at_hi] = np.maximum(g[at_hi], 0.0)
            pg[at_lo & at_hi] = 0.0                     # biến cố định (cạnh = 0)
            return float(np.max(np.abs(pg)))
        self.n_restarts_ = 0
        nit = int(res.nit)
        tol = max(opts.get("gtol", 1e-6), 1e-8)
        while getattr(self, "kkt_restart", True) and pgrad(res.x) > 10 * tol and self.n_restarts_ < KKT_RESTARTS:
            res = minimize(obj, res.x, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
            self.n_restarts_ += 1
            nit += int(res.nit)
        self.kkt_residual_ = pgrad(res.x)
        ChoquetGraph.n_fits_total += 1
        ChoquetGraph.n_restarted_total += int(self.n_restarts_ > 0)
        t = res.x
        self.converged_ = bool(res.success) and self.kkt_residual_ <= 10 * tol
        self.theta_ = t.copy()
        a, u, mp, mm = t[0], t[1:1 + p], t[1 + p:1 + p + P], t[1 + p + P:]
        m = u + A @ mm if mono else u.copy()
        self.a_, self.m_, self.mij_ = a, m, mp - mm
        self.mm_ = mm.copy()
        self.al_iters = nit
        self.pre_repair_violation = float(np.max(np.maximum(0, A @ mm - m))) if mono else 0.0
        # ràng buộc đơn điệu ở nút i "hoạt động" khi có tương tác âm tại nút i và
        # m_i − Σ m⁻ ≈ 0 (chặn thật sự); tách với cận m_i = 0 không có cạnh âm.
        slack = m - A @ mm
        self.active_nodes_ = np.where((A @ mm > ZERO) & (slack < 1e-5 * max(1.0, m.max())))[0]
        self.zero_nodes_ = np.where(m < ZERO)[0]
        # kiểm tra đơn điệu dạng gốc: m_i + Σ_j min(m_ij, 0) ≥ 0
        neg = np.zeros(p)
        np.add.at(neg, self.I_, np.minimum(self.mij_, 0))
        np.add.at(neg, self.J_, np.minimum(self.mij_, 0))
        self.min_monotone_margin_ = float(np.min(m + neg))
        return self

    def decision_function(self, X):
        return self.a_ + X @ self.m_ + self._design(X) @ self.mij_

    def normalized(self):
        S = self.m_.sum() + self.mij_.sum()
        S = S if abs(S) > 1e-12 else 1.0
        half = np.zeros(self.p)
        np.add.at(half, self.I_, 0.5 * self.mij_)
        np.add.at(half, self.J_, 0.5 * self.mij_)
        return (self.m_ + half) / S, self.mij_ / S, S


def orient(Xtr, ytr, Xte):
    """Đảo chiều cột có AUC < 0,5 trong fold huấn luyện (x → 1 − x)."""
    flip = np.array([roc_auc_score(ytr, Xtr[:, j]) < 0.5 for j in range(Xtr.shape[1])])
    Xtr, Xte = Xtr.copy(), Xte.copy()
    Xtr[:, flip] = 1 - Xtr[:, flip]
    Xte[:, flip] = 1 - Xte[:, flip]
    return Xtr, Xte, flip


def fit_path(X, y, monotone, edges=True, lambdas=LAMBDAS):
    models, t0 = [], None
    for lam in lambdas:
        mdl = ChoquetGraph(lam, monotone, edges).fit(X, y, t0)
        t0 = mdl.theta_
        models.append(mdl)
    return models


def choose_lambda(X, y, monotone, seed):
    """λ theo CV trong (INNER fold), chọn AUC trung bình lớn nhất; hoà → λ lớn hơn."""
    skf = StratifiedKFold(INNER, shuffle=True, random_state=seed)
    auc = np.zeros((INNER, len(LAMBDAS)))
    for f, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva, _ = orient(X[tr], y[tr], X[va])
        for k, mdl in enumerate(fit_path(Xtr, y[tr], monotone)):
            auc[f, k] = roc_auc_score(y[va], mdl.decision_function(Xva))
    mean = auc.mean(0)
    k = int(np.argmax(mean + 1e-9 * np.arange(len(LAMBDAS))[::-1]))
    return LAMBDAS[k], mean


def one_fold(fold, tr, te, Q, D, y, fixed_scores):
    t0 = time.time()
    res = {"fold": fold}
    ytr, yte = y[tr], y[te]
    Xi = Q / 4.0
    # ---- điểm cố định (không học)
    for name, s in fixed_scores.items():
        res[name] = roc_auc_score(yte, s[te])
    out_models = {}
    for level, X in (("item", Xi), ("domain", D)):
        Xtr, Xte, flip = orient(X[tr], ytr, X[te])
        res[f"{level}_nflip"] = int(flip.sum())
        # đồ thị đơn điệu thưa
        lam, cvm = choose_lambda(X[tr], ytr, True, SEED + fold)
        g = ChoquetGraph(lam, True).fit(Xtr, ytr)
        res[f"{level}_graph"] = roc_auc_score(yte, g.decision_function(Xte))
        res[f"{level}_graph_lambda"] = lam
        # đồ thị KHÔNG ràng buộc đơn điệu (cùng L1 cạnh, m_i tự do dấu)
        # cùng λ với mô hình đơn điệu → Δ AUC là "giá" của ràng buộc ở cùng mức phạt
        lamu = lam
        gu = ChoquetGraph(lamu, False).fit(Xtr, ytr)
        res[f"{level}_graph_unconstrained"] = roc_auc_score(yte, gu.decision_function(Xte))
        res[f"{level}_graph_unconstrained_lambda"] = lamu
        # cộng tính đơn điệu (cạnh = 0, λ → ∞)
        ad = ChoquetGraph(0.0, True, edges=False).fit(Xtr, ytr)
        res[f"{level}_additive_monotone"] = roc_auc_score(yte, ad.decision_function(Xte))
        # thống kê đơn điệu
        res[f"{level}_n_active_constraints"] = int(len(g.active_nodes_))
        res[f"{level}_n_zero_nodes"] = int(len(g.zero_nodes_))
        res[f"{level}_pre_repair_violation"] = g.pre_repair_violation
        res[f"{level}_min_monotone_margin"] = g.min_monotone_margin_
        res[f"{level}_lbfgs_iters"] = g.al_iters
        res[f"{level}_converged"] = g.converged_
        # mô hình không ràng buộc có vi phạm đơn điệu không?
        negu = np.zeros(X.shape[1])
        np.add.at(negu, gu.I_, np.minimum(gu.mij_, 0))
        np.add.at(negu, gu.J_, np.minimum(gu.mij_, 0))
        res[f"{level}_unconstrained_n_nonmonotone_nodes"] = int(
            np.sum((gu.m_ + negu < -ZERO) | (gu.m_ < -ZERO)))
        phi, I, S = g.normalized()
        res[f"{level}_phi"] = phi.tolist()
        res[f"{level}_I"] = I.tolist()
        res[f"{level}_S"] = float(S)
        res[f"{level}_nedges"] = int(np.sum(np.abs(g.mij_) > ZERO))
        res[f"{level}_nedges_unconstrained"] = int(np.sum(np.abs(gu.mij_) > ZERO))
        res[f"{level}_flip"] = flip.tolist()
    # ---- L1 logistic tuyến tính và L1 logistic tương tác cặp (tích), C theo CV trong
    inner = StratifiedKFold(INNER, shuffle=True, random_state=SEED + fold)
    lin = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=8, cv=inner, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000))
    lin.fit(Xi[tr], ytr)
    res["linear_l1_logistic"] = roc_auc_score(yte, lin.decision_function(Xi[te]))
    pw = make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False),
                       StandardScaler(), LogisticRegressionCV(
        Cs=8, cv=inner, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000))
    pw.fit(Xi[tr], ytr)
    res["pairwise_l1_logistic"] = roc_auc_score(yte, pw.decision_function(Xi[te]))
    res["sec"] = time.time() - t0
    return res


CACHE = os.path.join(__import__("tempfile").gettempdir(), f"g3_choquet_graph_cache_{SEED}_v2")


def _cached_fold(f, tr, te, Q, D, y, fixed):
    warnings.filterwarnings("ignore")
    fn = os.path.join(CACHE, f"fold{f:02d}.json")
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as fh:
            return json.load(fh)
    r = one_fold(f, tr, te, Q, D, y, fixed)
    r = json.loads(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(r, fh)
    return r


def run_folds(splits, Q, D, y, fixed):
    """Chạy song song, lưu đệm từng fold (thư mục tạm) để chạy lại được nếu tiến
    trình con bị hỏng (đã gặp BrokenProcessPool/WinError 6 trên Windows)."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(CACHE, exist_ok=True)
    todo = list(range(len(splits)))
    for attempt in range(4):
        done = {}
        try:
            with ProcessPoolExecutor(int(os.environ.get("G3_JOBS", "2"))) as ex:
                futs = {f: ex.submit(_cached_fold, f, splits[f][0], splits[f][1], Q, D, y, fixed)
                        for f in todo}
                for f, fu in futs.items():
                    try:
                        done[f] = fu.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"fold {f} lỗi ({type(e).__name__}); sẽ chạy lại", flush=True)
        except Exception as e:  # noqa: BLE001
            print("pool lỗi:", type(e).__name__, flush=True)
        todo = [f for f in todo if f not in done and
                not os.path.exists(os.path.join(CACHE, f"fold{f:02d}.json"))]
        print(f"lượt {attempt}: còn {len(todo)} fold", flush=True)
        if not todo:
            break
    return [_cached_fold(f, *splits[f], Q, D, y, fixed) for f in range(len(splits))]


def sanity_checks(D, y):
    """(1) dạng Möbius ≡ dạng dự án C = Σφx − ½ΣI|xi−xj|; (2) bộ giải thoả đơn điệu."""
    rng = np.random.default_rng(SEED)
    p = 4
    m = rng.uniform(0, 1, p)
    I_, J_ = ChoquetGraph.pairs(p)
    mij = rng.normal(0, 0.2, len(I_))
    S = m.sum() + mij.sum()
    names = [f"d{k}" for k in range(p)]
    mdl = ChoquetGraph(0.0)
    mdl.p, mdl.I_, mdl.J_, mdl.m_, mdl.mij_, mdl.a_ = p, I_, J_, m, mij, 0.0
    phi, I, _ = mdl.normalized()
    shap = dict(zip(names, phi))
    inter = {(names[a], names[b]): I[k] for k, (a, b) in enumerate(zip(I_, J_))}
    X = rng.uniform(0, 1, (200, p))
    c1 = mdl.decision_function(X) / S
    c2 = np.array([choquet_2additive(dict(zip(names, x)), shap, inter) for x in X])
    return float(np.max(np.abs(c1 - c2)))


def main():
    t_start = time.time()
    Q, y, meta = load_poland()
    n = len(y)
    dom = domain_scores(Q)
    dnames = list(dom.keys())
    D = np.column_stack([dom[k] for k in dnames])
    rows = to_rows(dom)
    fixed = {
        "raw_sum_25": Q.sum(1),
        "qchat10_sum": Q[:, [i - 1 for i in QCHAT10_ITEMS]].sum(1),
        "shapley_domain_clinical": aggregate(rows, "shapley"),
        "choquet_clinical_interactions": aggregate(rows, "choquet",
                                                   interaction=cfg.INTERACTION_CLINICAL),
    }
    full_auc = {k: float(roc_auc_score(y, v)) for k, v in fixed.items()}
    mob_err = sanity_checks(D, y)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    splits = list(rskf.split(Q, y))
    if os.environ.get("G3_QUICK"):
        splits = splits[:int(os.environ["G3_QUICK"])]
    results = run_folds(splits, Q, D, y, fixed)
    results.sort(key=lambda r: r["fold"])
    n_te = len(splits[0][1]); n_tr = n - n_te

    methods = ["item_graph", "item_graph_unconstrained", "item_additive_monotone",
               "domain_graph", "domain_graph_unconstrained", "domain_additive_monotone",
               "linear_l1_logistic", "pairwise_l1_logistic",
               "raw_sum_25", "qchat10_sum", "shapley_domain_clinical",
               "choquet_clinical_interactions"]
    A = {m: np.array([r[m] for r in results]) for m in methods}
    summary = {m: {"mean": float(A[m].mean()), "sd": float(A[m].std(ddof=1))} for m in methods}

    comps = [
        ("item_graph", "raw_sum_25"), ("item_graph", "shapley_domain_clinical"),
        ("item_graph", "item_additive_monotone"), ("item_graph", "pairwise_l1_logistic"),
        ("item_graph", "qchat10_sum"), ("item_graph", "linear_l1_logistic"),
        ("item_graph", "item_graph_unconstrained"),
        ("domain_graph", "shapley_domain_clinical"), ("domain_graph", "domain_additive_monotone"),
        ("domain_graph", "domain_graph_unconstrained"),
        ("domain_graph", "choquet_clinical_interactions"),
        ("item_graph", "domain_graph"),
    ]
    ctab = []
    for a, b in comps:
        d, t, p, naive, half = nadeau_bengio(A[a], A[b], n_tr, n_te)
        ctab.append({"a": a, "b": b, "delta": d, "ci95": [d - half, d + half], "p_nb": p})
    padj, rej = benjamini_hochberg([c["p_nb"] for c in ctab])
    for c, pa, r in zip(ctab, padj, rej):
        c["p_bh"] = float(pa); c["reject_bh05"] = bool(r)

    # ---- độ ổn định đồ thị mức mục
    I_, J_ = ChoquetGraph.pairs(25)
    Iitem = np.array([r["item_I"] for r in results])           # 50 × 300
    sel = np.abs(Iitem) > ZERO
    freq = sel.mean(0)
    pos = (Iitem > ZERO).sum(0); negc = (Iitem < -ZERO).sum(0)
    edges = []
    for k in np.argsort(-freq):
        if freq[k] < STABLE:
            break
        i, j = I_[k] + 1, J_[k] + 1
        sgn = "+" if pos[k] >= negc[k] else "-"
        nz = Iitem[sel[:, k], k]
        edges.append({
            "items": [int(i), int(j)], "freq": float(freq[k]),
            "sign": sgn, "sign_consistency": float(max(pos[k], negc[k]) / max(1, sel[:, k].sum())),
            "mean_I_when_selected": float(nz.mean()),
            "domains": [QCHAT_ITEMS[i][0], QCHAT_ITEMS[j][0]],
            "desc": [QCHAT_ITEMS[i][1], QCHAT_ITEMS[j][1]],
            "meaning": ("cộng hưởng: cần CẢ HAI cùng có" if sgn == "+"
                        else "dư thừa: MỘT trong hai là đủ"),
        })
    top_any = [{"items": [int(I_[k] + 1), int(J_[k] + 1)], "freq": float(freq[k]),
                "pos": int(pos[k]), "neg": int(negc[k])}
               for k in np.argsort(-freq)[:15]]
    within = [e for e in edges if e["domains"][0] == e["domains"][1]]
    phi_item = np.array([r["item_phi"] for r in results]).mean(0)
    # tỉ lệ cạnh cùng lĩnh vực trong 300 cặp (cơ sở so sánh)
    dom_of = [QCHAT_ITEMS[i + 1][0] for i in range(25)]
    base_within = float(np.mean([dom_of[a] == dom_of[b] for a, b in zip(I_, J_)]))
    sel_within = float(np.sum(sel[:, [dom_of[a] == dom_of[b] for a, b in zip(I_, J_)]]) / max(1, sel.sum()))

    # ---- mức lĩnh vực
    dI_, dJ_ = ChoquetGraph.pairs(len(dnames))
    Idom = np.array([r["domain_I"] for r in results])
    phidom = np.array([r["domain_phi"] for r in results])
    dom_edges = []
    for k, (a, b) in enumerate(zip(dI_, dJ_)):
        pair = (dnames[a], dnames[b])
        clin = cfg.INTERACTION_CLINICAL.get(pair, cfg.INTERACTION_CLINICAL.get(pair[::-1], 0.0))
        dom_edges.append({"pair": list(pair), "freq": float((np.abs(Idom[:, k]) > ZERO).mean()),
                          "mean_I": float(Idom[:, k].mean()),
                          "median_I": float(np.median(Idom[:, k])),
                          "n_pos": int((Idom[:, k] > ZERO).sum()),
                          "n_neg": int((Idom[:, k] < -ZERO).sum()),
                          "clinical_I": clin})
    dom_phi = {dnames[k]: float(phidom[:, k].mean()) for k in range(len(dnames))}

    # ---- đơn điệu
    mono = {}
    for lv in ("item", "domain"):
        act = np.array([r[f"{lv}_n_active_constraints"] for r in results])
        mono[lv] = {
            "folds_with_active_constraint": float((act > 0).mean()),
            "mean_active_nodes": float(act.mean()),
            "mean_zero_nodes": float(np.mean([r[f"{lv}_n_zero_nodes"] for r in results])),
            "max_pre_repair_violation": float(max(r[f"{lv}_pre_repair_violation"] for r in results)),
            "min_monotone_margin": float(min(r[f"{lv}_min_monotone_margin"] for r in results)),
            "unconstrained_folds_nonmonotone": float(np.mean(
                [r[f"{lv}_unconstrained_n_nonmonotone_nodes"] > 0 for r in results])),
            "unconstrained_mean_nonmonotone_nodes": float(np.mean(
                [r[f"{lv}_unconstrained_n_nonmonotone_nodes"] for r in results])),
            "mean_edges": float(np.mean([r[f"{lv}_nedges"] for r in results])),
            "mean_edges_unconstrained": float(np.mean([r[f"{lv}_nedges_unconstrained"] for r in results])),
            "lambda_counts": {str(l): int(sum(r[f"{lv}_graph_lambda"] == l for r in results)) for l in LAMBDAS},
            "n_flipped_mean": float(np.mean([r[f"{lv}_nflip"] for r in results])),
        }

    # ---- đồ thị trên toàn bộ dữ liệu (chỉ để minh hoạ, λ = trung vị các λ được chọn)
    lam_full = float(np.median([r["item_graph_lambda"] for r in results]))
    lam_full = min(LAMBDAS, key=lambda l: abs(np.log(l) - np.log(lam_full)))
    Xf, _, _ = orient(Q / 4.0, y, Q / 4.0)
    gfull = ChoquetGraph(lam_full, True).fit(Xf, y)
    phif, If, Sf = gfull.normalized()
    full_edges = [{"items": [int(I_[k] + 1), int(J_[k] + 1)], "I": float(If[k])}
                  for k in np.argsort(-np.abs(If)) if abs(If[k]) > ZERO]

    runtime = time.time() - t_start
    out = {"seed": SEED, "n": int(n), "n_pos": int(y.sum()), "lambdas": LAMBDAS, "gamma": GAMMA,
           "domains": dnames, "full_sample_auc_fixed": full_auc,
           "mobius_vs_project_formula_maxerr": mob_err,
           "cv_auc": summary, "comparisons": ctab, "stable_edges": edges,
           "top15_edges_any": top_any, "within_domain_share_selected": sel_within,
           "within_domain_share_baseline": base_within,
           "item_phi_mean": {str(i + 1): float(phi_item[i]) for i in range(25)},
           "domain_edges": dom_edges, "domain_phi_mean": dom_phi,
           "monotonicity": mono, "full_data_graph": {"lambda": lam_full, "S": float(Sf),
                                                     "edges": full_edges},
           "runtime_sec": runtime, "folds": results}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    L = []
    L.append("G3 — ĐỒ THỊ TƯƠNG TÁC CHOQUET 2-CỘNG TÍNH THƯA, ĐƠN ĐIỆU (Q-CHAT-25 Ba Lan)")
    L.append(f"n={n} ({int(y.sum())} ASD / {n - int(y.sum())} chứng), 5-fold × 10, seed {SEED}, "
             f"λ ∈ {LAMBDAS} chọn bằng CV trong {INNER}-fold, γ={GAMMA}. Thời gian {runtime:.0f}s")
    L.append(f"Kiểm tra: |Möbius − choquet_2additive dự án| max = {mob_err:.2e}")
    L.append("AUC toàn mẫu (điểm cố định): " + ", ".join(f"{k} {v:.4f}" for k, v in full_auc.items()))
    L.append("")
    L.append("AUC ngoài fold (trung bình ± SD, 50 fold)")
    for m in methods:
        L.append(f"  {m:32s} {summary[m]['mean']:.4f} ± {summary[m]['sd']:.4f}")
    L.append("")
    L.append("So sánh (Nadeau–Bengio, BH trên 12 phép so sánh)")
    for c in ctab:
        L.append(f"  {c['a']:26s} − {c['b']:30s} Δ={c['delta']:+.4f} "
                 f"[{c['ci95'][0]:+.4f};{c['ci95'][1]:+.4f}] p={c['p_nb']:.3f} p_BH={c['p_bh']:.3f}")
    L.append("")
    L.append(f"Cạnh ổn định mức mục (chọn ở ≥ {STABLE:.0%} fold): {len(edges)}")
    for e in edges:
        L.append(f"  Q{e['items'][0]}–Q{e['items'][1]}  f={e['freq']:.2f} dấu {e['sign']} "
                 f"(nhất quán {e['sign_consistency']:.2f}) I≈{e['mean_I_when_selected']:+.4f} "
                 f"[{e['domains'][0]}/{e['domains'][1]}] {e['desc'][0]} | {e['desc'][1]} — {e['meaning']}")
    L.append("15 cạnh hay chọn nhất: " + "; ".join(
        f"Q{t['items'][0]}–Q{t['items'][1]} {t['freq']:.2f}(+{t['pos']}/−{t['neg']})" for t in top_any))
    L.append(f"Tỉ lệ cạnh được chọn nằm trong cùng lĩnh vực: {sel_within:.2f} (cơ sở ngẫu nhiên {base_within:.2f})")
    L.append("")
    L.append("Mức lĩnh vực — I học được (chuẩn hoá) so với bộ lâm sàng")
    for e in dom_edges:
        L.append(f"  {e['pair'][0]:5s}–{e['pair'][1]:5s} f={e['freq']:.2f} mean I={e['mean_I']:+.4f} "
                 f"median={e['median_I']:+.4f} (+{e['n_pos']}/−{e['n_neg']})  lâm sàng {e['clinical_I']:+.2f}")
    L.append("  φ học được: " + ", ".join(f"{k} {v:.3f}" for k, v in dom_phi.items())
             + " | φ lâm sàng: " + ", ".join(f"{k} {cfg.SHAPLEY[k]:.3f}" for k in dnames))
    L.append("")
    L.append("Đơn điệu")
    for lv, d in mono.items():
        L.append(f"  {lv}: " + json.dumps(d, ensure_ascii=False))
    L.append("")
    L.append(f"Đồ thị toàn mẫu (minh hoạ, λ={lam_full}): {len(full_edges)} cạnh: " + "; ".join(
        f"Q{e['items'][0]}–Q{e['items'][1]} {e['I']:+.3f}" for e in full_edges[:20]))
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)

    # ---- hình (tuỳ chọn)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        for i in range(25):
            G.add_node(i + 1, dom=dom_of[i])
        for k in range(len(I_)):
            if freq[k] >= 0.5:
                G.add_edge(int(I_[k] + 1), int(J_[k] + 1), f=freq[k],
                           s=1 if pos[k] >= negc[k] else -1)
        dom_col = {"GTNN": "#4C78A8", "CNXH": "#F58518", "BCHOC": "#54A24B", "DHBT": "#B279A2"}
        pos_xy = nx.circular_layout(sorted(G.nodes, key=lambda i: (dom_of[i - 1], i)))
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_nodes(G, pos_xy, ax=ax, node_color=[dom_col[dom_of[i - 1]] for i in G.nodes],
                               node_size=[3000 * max(phi_item[i - 1], 0.005) for i in G.nodes])
        nx.draw_networkx_labels(G, pos_xy, ax=ax, labels={i: f"Q{i}" for i in G.nodes}, font_size=8)
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos_xy, edgelist=[(u, v)], ax=ax, width=1 + 5 * (d["f"] - 0.5) * 2,
                                   edge_color="#C0392B" if d["s"] > 0 else "#2471A3",
                                   style="solid" if d["f"] >= STABLE else "dashed")
        ax.set_title("Đồ thị tương tác học được (cạnh chọn ≥50% fold; liền: ≥70%)\n"
                     "đỏ = cộng hưởng (+), xanh = dư thừa (−); cỡ nút = φ trung bình", fontsize=9)
        for dname, c in dom_col.items():
            ax.scatter([], [], c=c, label=dname)
        ax.legend(loc="lower right", fontsize=8)
        ax.axis("off")
        os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
        fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    except Exception as ex:  # hình không bắt buộc
        print("Bỏ qua hình:", ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
