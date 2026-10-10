# -*- coding: utf-8 -*-
"""
run_w2_external.py — W2-3 + W2-4: logistic Choquet 2-cộng tính đơn điệu (tham số hoá
lại, L-BFGS-B) trên POL252 và ba bộ ngoài lĩnh vực tự kỷ, so với đối thủ trực tiếp.

Bộ dữ liệu (data/external/p3/, nạp qua p3_common):
  POL252  Q-CHAT-25 Ba Lan, 25 mục 0–4           → x = Q/4
  PARK195 UCI Parkinsons, 22 đo âm học liên tục  → biến đổi phân vị đều (fit trong fold)
  BCW683  Breast Cancer Wisconsin gốc, 9 mục 1–10 → x = (Q − 1)/9
  HEART303 Cleveland, 13 thuộc tính hỗn hợp       → phân vị đều; có thuộc tính định danh
          (cp, restecg, slope, thal) nên giả thiết đơn điệu SAI MỘT PHẦN — ca thử căng.
Mọi biến đổi là đơn điệu theo từng cột, fit trong fold huấn luyện; hướng mỗi cột đặt
theo dấu AUC trong fold huấn luyện (như G3).

Phương pháp (mỗi fold ngoài, 5-fold × 10 lặp):
  choquet_graph          đề xuất: đơn điệu, L1 trên cạnh, λ chọn bằng CV trong 3-fold
  choquet_unconstrained  cùng λ, bỏ ràng buộc đơn điệu
  additive_monotone      hệ số không âm, không cạnh
  choquistic             Tehrani et al. (2012) giới hạn 2-cộng tính: độ đo chuẩn hoá,
                         σ(γ(C_μ − β)), ML không phạt, SLSQP (bản cài lại)
  lgbm_monotone          LightGBM với monotone_constraints = +1 mọi cột (đã định hướng)
  linear_l1_logistic     L1 logistic, C chọn bằng CV trong
  pairwise_l1_logistic   L1 logistic trên tích đôi, C chọn bằng CV trong
So sánh: Nadeau–Bengio, BH trong từng bộ dữ liệu.

Chạy:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/run_w2_external.py
       W2_QUICK=2 … (chỉ 2 fold/bộ để thử)   W2_JOBS=3 (số tiến trình)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, StandardScaler

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import p3_common as P3                                            # noqa: E402
from p3_common import benjamini_hochberg, nadeau_bengio           # noqa: E402
from run_g3_choquet_graph import ChoquetGraph, choose_lambda, orient, ZERO  # noqa: E402
from w2_common import Choquistic, logloss_unpenalized              # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20260925
ROOT = os.path.dirname(SRC)
OUT_TXT = os.path.join(ROOT, "out", "reports", "w2_external.txt")
OUT_JSON = os.path.join(ROOT, "out", "reports", "w2_external.json")
CACHE = os.path.join(tempfile.gettempdir(), f"w2_external_cache_{SEED}_v1")
INNER = 3
N_SLSQP = 10          # số fold đầu (lặp 1–2) chạy choquistic SLSQP tham số hoá gốc
DATASETS = ["POL252", "PARK195", "BCW683", "HEART303"]
METHODS = ["choquet_graph", "choquet_unconstrained", "additive_monotone", "choquistic_ml",
           "lgbm_monotone", "linear_l1_logistic", "pairwise_l1_logistic"]
COMPARE = [("choquet_graph", m) for m in METHODS[1:]]


def load(key):
    ds = P3.REPLICATION_LOADERS[key]()
    return ds.X.astype(float), ds.y.astype(int), ds


def scale(key, Xtr, Xte):
    if key == "POL252":
        return Xtr / 4.0, Xte / 4.0
    if key == "BCW683":
        return (Xtr - 1) / 9.0, (Xte - 1) / 9.0
    qt = QuantileTransformer(n_quantiles=min(100, len(Xtr)), output_distribution="uniform")
    return qt.fit_transform(Xtr), np.clip(qt.transform(Xte), 0, 1)


def nonmonotone_nodes(model):
    neg = np.zeros(model.p)
    np.add.at(neg, model.I_, np.minimum(model.mij_, 0))
    np.add.at(neg, model.J_, np.minimum(model.mij_, 0))
    return int(np.sum((model.m_ + neg < -ZERO) | (model.m_ < -ZERO)))


SOLVER_V = 2   # v2: ChoquetGraph có kiểm KKT + khởi động lại L-BFGS-B (xem run_g3_choquet_graph.fit)


def prepare(key, tr, te, X, y):
    ytr, yte = y[tr], y[te]
    Xtr, Xte = scale(key, X[tr], X[te])
    Xtr, Xte, flip = orient(Xtr, ytr, Xte)
    return Xtr, Xte, ytr, yte, flip


def choquet_part(key, fold, tr, te, X, y):
    """Mọi kết quả phụ thuộc bộ giải ChoquetGraph (tính lại được riêng khi sửa bộ giải)."""
    res = {}
    n0, r0 = ChoquetGraph.n_fits_total, ChoquetGraph.n_restarted_total
    Xtr, Xte, ytr, yte, flip = prepare(key, tr, te, X, y)
    res["n_flip"] = int(flip.sum())
    # λ theo CV trong trên dữ liệu đã chia tỉ lệ (orient lặp lại bên trong, vô hại)
    lam, _ = choose_lambda(Xtr, ytr, True, SEED + fold)
    tic = time.perf_counter()
    g = ChoquetGraph(lam, True).fit(Xtr, ytr)
    res["sec_choquet_graph"] = time.perf_counter() - tic
    res["choquet_graph"] = roc_auc_score(yte, g.decision_function(Xte))
    res["lambda"] = lam
    res["n_edges"] = int(np.sum(np.abs(g.mij_) > ZERO))
    res["n_active"] = int(len(g.active_nodes_))
    res["min_margin"] = g.min_monotone_margin_
    gu = ChoquetGraph(lam, False).fit(Xtr, ytr)
    res["choquet_unconstrained"] = roc_auc_score(yte, gu.decision_function(Xte))
    res["unc_nonmonotone_nodes"] = nonmonotone_nodes(gu)
    res["n_edges_unc"] = int(np.sum(np.abs(gu.mij_) > ZERO))
    ad = ChoquetGraph(0.0, True, edges=False).fit(Xtr, ytr)
    res["additive_monotone"] = roc_auc_score(yte, ad.decision_function(Xte))
    import run_g3_choquet_graph as G
    old = G.GAMMA
    G.GAMMA = 0.0
    try:
        tic = time.perf_counter()
        g0 = ChoquetGraph(0.0, True)
        g0.options = {"maxiter": 50000, "maxfun": 100000, "ftol": 1e-15, "gtol": 1e-10}
        g0.fit(Xtr, ytr)
        res["sec_reparam_ml"] = time.perf_counter() - tic
    finally:
        G.GAMMA = old
    res["reparam_ml_train_logloss"] = logloss_unpenalized(g0, Xtr, ytr)
    res["reparam_ml_auc"] = roc_auc_score(yte, g0.decision_function(Xte))
    res["n_solver_fits"] = ChoquetGraph.n_fits_total - n0
    res["n_solver_restarted"] = ChoquetGraph.n_restarted_total - r0
    res["solver_v"] = SOLVER_V
    return res


def one_fold(key, fold, tr, te, X, y):
    t0 = time.time()
    res = {"dataset": key, "fold": fold}
    res.update(choquet_part(key, fold, tr, te, X, y))
    Xtr, Xte, ytr, yte, _ = prepare(key, tr, te, X, y)
    # Choquistic (tham số hoá gốc, SLSQP) chỉ ở N_SLSQP fold đầu — đủ để kiểm tương
    # đương nghiệm và đo chi phí; SLSQP tốn ~300 s/fold ở p ≈ 25 trên máy dùng chung.
    # Cột AUC "choquistic_ml" (mọi fold) = nghiệm ML cùng họ giải bằng tham số hoá lại
    # (Mệnh đề 4: cùng cực tiểu; kiểm bằng số trên các fold có SLSQP).
    if fold < N_SLSQP:
        ch = Choquistic(maxiter=1000).fit(Xtr, ytr)
        res["choquistic"] = roc_auc_score(yte, ch.decision_function(Xte))
        res["sec_choquistic"] = ch.sec_
        res["choquistic_converged"] = ch.converged_
        res["choquistic_train_logloss"] = logloss_unpenalized(ch, Xtr, ytr)
    # LightGBM đơn điệu
    import lightgbm as lgb
    lg = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                            min_child_samples=10, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, monotone_constraints=[1] * X.shape[1],
                            monotone_constraints_method="advanced",
                            random_state=SEED + fold, verbose=-1, n_jobs=1)
    lg.fit(Xtr, ytr)
    res["lgbm_monotone"] = roc_auc_score(yte, lg.predict_proba(Xte)[:, 1])
    # logistic L1 tuyến tính và tương tác đôi
    inner = StratifiedKFold(INNER, shuffle=True, random_state=SEED + fold)
    lin = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=8, cv=inner, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000))
    lin.fit(Xtr, ytr)
    res["linear_l1_logistic"] = roc_auc_score(yte, lin.decision_function(Xte))
    pw = make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False),
                       StandardScaler(), LogisticRegressionCV(
        Cs=8, cv=inner, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000))
    pw.fit(Xtr, ytr)
    res["pairwise_l1_logistic"] = roc_auc_score(yte, pw.decision_function(Xte))
    res["sec"] = time.time() - t0
    return res


def _cached(key, f, tr, te, X, y):
    warnings.filterwarnings("ignore")
    fn = os.path.join(CACHE, f"{key}_fold{f:02d}.json")
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("solver_v") == SOLVER_V:
            return r
        # đệm cũ (bộ giải v1): chỉ tính lại phần phụ thuộc ChoquetGraph, giữ nguyên phần còn lại;
        # lưu giá trị v1 để báo cáo bản sửa đã thay đổi gì
        r["v1"] = {k: r.get(k) for k in ("choquet_graph", "choquet_unconstrained", "additive_monotone",
                                         "reparam_ml_auc", "reparam_ml_train_logloss", "lambda")}
        r.update(choquet_part(key, f, tr, te, X, y))
        r = json.loads(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        with open(fn, "w", encoding="utf-8") as fh:
            json.dump(r, fh)
        return r
    r = one_fold(key, f, tr, te, X, y)
    r = json.loads(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(r, fh)
    return r


def run_all(jobs):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(CACHE, exist_ok=True)
    tasks = []
    data = {}
    for key in DATASETS:
        X, y, ds = load(key)
        data[key] = (X, y, ds)
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(X, y))
        if os.environ.get("W2_QUICK"):
            splits = splits[:int(os.environ["W2_QUICK"])]
        data[key] += (splits,)
        tasks += [(key, f, tr, te) for f, (tr, te) in enumerate(splits)]
    # việc đắt nhất (p lớn) trước
    tasks.sort(key=lambda t: -data[t[0]][0].shape[1])
    # Chế độ shard (tránh BrokenProcessPool/WinError 6 khi chạy nền trên Windows):
    # W2_SHARD=i/N → tiến trình này chỉ làm các việc có chỉ số ≡ i (mod N), tuần tự, rồi thoát.
    if os.environ.get("W2_SHARD"):
        i, N = (int(v) for v in os.environ["W2_SHARD"].split("/"))
        for k, (key, f, tr, te) in enumerate(tasks):
            if k % N == i:
                _cached(key, f, tr, te, data[key][0], data[key][1])
        print(f"shard {i}/{N} xong", flush=True)
        sys.exit(0)
    for attempt in range(4 if jobs > 1 else 0):
        todo = [t for t in tasks if not os.path.exists(os.path.join(CACHE, f"{t[0]}_fold{t[1]:02d}.json"))]
        print(f"lượt {attempt}: còn {len(todo)} fold", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(jobs) as ex:
                futs = [ex.submit(_cached, k, f, tr, te, data[k][0], data[k][1]) for k, f, tr, te in todo]
                for i, fu in enumerate(futs):
                    try:
                        fu.result()
                    except Exception as e:  # noqa: BLE001
                        print("fold lỗi:", type(e).__name__, e, flush=True)
                    if i % 20 == 0:
                        print(f"  xong {i + 1}/{len(futs)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print("pool lỗi:", type(e).__name__, flush=True)
    out = {}
    for key in DATASETS:
        X, y, ds, splits = data[key]
        out[key] = [_cached(key, f, tr, te, X, y) for f, (tr, te) in enumerate(splits)]
    return out, data


def main():
    t0 = time.time()
    jobs = int(os.environ.get("W2_JOBS", "3"))
    folds, data = run_all(jobs)
    report = {"seed": SEED, "datasets": {}}
    L = ["W2-3/W2-4 — Choquet 2-cộng tính đơn điệu trên 4 bộ dữ liệu, 5-fold × 10", ""]
    for key in DATASETS:
        X, y, ds, splits = data[key]
        R = folds[key]
        n = len(y); n_te = len(splits[0][1]); n_tr = n - n_te
        for r in R:
            r["choquistic_ml"] = r["reparam_ml_auc"]
        A = {m: np.array([r[m] for r in R]) for m in METHODS}
        summ = {m: {"mean": float(A[m].mean()), "sd": float(A[m].std(ddof=1))} for m in METHODS}
        comps = []
        for a, b in COMPARE:
            d, t, p, naive, half = nadeau_bengio(A[a], A[b], n_tr, n_te)
            comps.append({"a": a, "b": b, "delta": d, "ci95": [d - half, d + half], "p_nb": p})
        padj, rej = benjamini_hochberg([c["p_nb"] for c in comps])
        for c, pa, rj in zip(comps, padj, rej):
            c["p_bh"] = float(pa); c["reject"] = bool(rj)
        RS = [r for r in R if "sec_choquistic" in r]          # fold có SLSQP
        ll_gap = np.array([r["reparam_ml_train_logloss"] - r["choquistic_train_logloss"] for r in RS])
        auc_gap = np.array([r["reparam_ml_auc"] - r["choquistic"] for r in RS])
        mono = {
            "folds_active": float(np.mean([r["n_active"] > 0 for r in R])),
            "mean_active": float(np.mean([r["n_active"] for r in R])),
            "min_margin": float(min(r["min_margin"] for r in R)),
            "unc_folds_nonmonotone": float(np.mean([r["unc_nonmonotone_nodes"] > 0 for r in R])),
            "unc_mean_nonmonotone_nodes": float(np.mean([r["unc_nonmonotone_nodes"] for r in R])),
            "mean_edges": float(np.mean([r["n_edges"] for r in R])),
            "mean_edges_unc": float(np.mean([r["n_edges_unc"] for r in R])),
            "mean_flip": float(np.mean([r["n_flip"] for r in R])),
            "solver_fits": int(sum(r["n_solver_fits"] for r in R)),
            "solver_restarted": int(sum(r["n_solver_restarted"] for r in R)),
            "v1_max_abs_auc_change": float(max(
                max(abs(r["v1"][k] - r[k]) for k in ("choquet_graph", "choquet_unconstrained", "additive_monotone"))
                for r in R if "v1" in r)) if any("v1" in r for r in R) else 0.0,
        }
        cost = {
            "sec_choquet_graph_median": float(np.median([r["sec_choquet_graph"] for r in R])),
            "n_slsqp_folds": len(RS),
            "sec_reparam_ml_median": float(np.median([r["sec_reparam_ml"] for r in RS])),
            "sec_choquistic_median": float(np.median([r["sec_choquistic"] for r in RS])),
            "choquistic_converged_frac": float(np.mean([r["choquistic_converged"] for r in RS])),
            "test_auc_gap_reparam_ml_minus_slsqp": {"max_abs": float(np.abs(auc_gap).max()),
                                                    "mean": float(auc_gap.mean())},
            "auc_choquistic_slsqp_mean_subset": float(np.mean([r["choquistic"] for r in RS])),
            "auc_reparam_ml_mean_subset": float(np.mean([r["reparam_ml_auc"] for r in RS])),
            "train_logloss_gap_reparam_minus_choquistic": {
                "median": float(np.median(ll_gap)), "max": float(ll_gap.max()), "min": float(ll_gap.min())},
            "auc_reparam_ml_mean": float(np.mean([r["reparam_ml_auc"] for r in R])),
        }
        report["datasets"][key] = {"n": int(n), "p": int(X.shape[1]), "prevalence": float(y.mean()),
                                   "name": ds.name, "cv_auc": summ, "comparisons": comps,
                                   "monotonicity": mono, "cost": cost}
        L.append(f"[{key}] {ds.name}: n={n}, p={X.shape[1]}, dương={y.mean():.2f}")
        for m in METHODS:
            L.append(f"  {m:24s} {summ[m]['mean']:.4f} ± {summ[m]['sd']:.4f}")
        for c in comps:
            L.append(f"  {c['a']} − {c['b']:24s} Δ={c['delta']:+.4f} [{c['ci95'][0]:+.4f};{c['ci95'][1]:+.4f}] "
                     f"p={c['p_nb']:.3f} p_BH={c['p_bh']:.3f}")
        L.append(f"  đơn điệu: {json.dumps(mono)}")
        L.append(f"  chi phí: {json.dumps(cost)}")
        L.append("")
    report["runtime_sec"] = time.time() - t0
    L.append(f"Thời gian {report['runtime_sec']:.0f}s")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    report["folds"] = folds
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
