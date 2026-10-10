# -*- coding: utf-8 -*-
"""
run_fis_bypass.py — Bảng 5.11 (và kiểm tra lại Bảng 5.12) trên đoàn hệ Ba Lan.

    python src/run_fis_bypass.py
    python src/run_fis_bypass.py --boot 500        # nhanh hơn

MỤC ĐÍCH
--------
Phụ lục B của luận văn dẫn tới kịch bản này nhưng tệp không còn trong dự án; chú
thích trong config.py [REVIEW-P10.3] và Bảng 5.11 lại ghi hai bộ số khác nhau.
Tệp này dựng lại phép đo từ đầu để có MỘT bộ số tái lập được:

  • Bảng 5.11: Nhánh 1 khi d_i đi thẳng vào tầng tổng hợp (bỏ qua tầng 2) so với
    khi d_i đi qua tầng suy diễn Mamdani với LAG ≡ 0 — trên 25 mục và 10 mục
    Q-CHAT-10, với hai bộ Shapley (tiên nghiệm V1 và hậu nghiệm trong cfg).
  • Cơ chế: tỉ lệ quan sát rơi vào vùng phẳng của ánh xạ d → risk khi LAG ≡ 0.
  • Bảng 5.12: tổng mức vi phạm đơn điệu của bề mặt suy diễn cho năm cấu hình
    (thêm 'hàm thuộc nguyên bản + tổng chặn' để chứng minh HAI nguồn là độc lập).

HỆ SUY DIỄN
-----------
Tự cài đặt đúng theo config.py để kết quả không phụ thuộc mô-đun khác:
t-norm tích, gộp luật cùng kết luận bằng cfg.S_NORM, cắt kết luận bằng min (Mamdani),
hợp thành các kết luận khác nhau bằng max, giải mờ trọng tâm trên lưới bước
cfg.RES_RISK. DHBT dùng cfg.RULES_AGELESS khi cfg.DHBT_USE_LAG_TABLE = False.
Hai giá trị kiểm chứng in ở đầu: risk(0, 0) và risk(1, 3) phải bằng
cfg.RISK_FLOOR_MAMDANI và cfg.RISK_CEIL_MAMDANI (11,50 và 88,50). Nếu dự án có
ts_fuzzy.py, nên đối chiếu thêm vài điểm với hàm của ts_fuzzy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import config as cfg
import ts_external
from ts_aggregate import renormalize_for_missing
from ts_stats_extra import auc_fast, delong_paired, paired_boot_repeats

DOMS = ["GTNN", "CNXH", "BCHOC", "DHBT"]
SEED = 20260827


# =====================================================================
# Hàm thuộc và hệ Mamdani
# =====================================================================

def mf(kind: str, p, x):
    x = np.asarray(x, dtype=float)
    if kind == "trimf":
        a, b, c = p
        left = np.ones_like(x) if b == a else (x - a) / (b - a)
        right = np.ones_like(x) if c == b else (c - x) / (c - b)
        y = np.minimum(left, right)
        y = np.where((x < a) | (x > c), 0.0, y)
    elif kind == "trapmf":
        a, b, c, d = p
        left = np.ones_like(x) if b == a else (x - a) / (b - a)
        right = np.ones_like(x) if d == c else (d - x) / (d - c)
        y = np.minimum(np.minimum(left, 1.0), right)
        y = np.where((x < a) | (x > d), 0.0, y)
    else:
        raise ValueError(kind)
    return np.clip(y, 0.0, 1.0)


def s_norm(ws, how):
    ws = np.asarray(ws, dtype=float)
    if how == "max":
        return float(ws.max())
    if how == "prob_sum":
        out = 0.0
        for w in ws:
            out = out + w - out * w
        return float(out)
    if how == "bounded_sum":
        return float(min(1.0, ws.sum()))
    raise ValueError(how)


class Mamdani:
    def __init__(self, mf_d=None, mf_lag=None, s="bounded_sum"):
        self.mf_d = mf_d or cfg.MF_D
        self.mf_lag = mf_lag or cfg.MF_LAG
        self.s = s
        self.z = np.arange(0.0, 100.0 + 1e-9, cfg.RES_RISK)
        self.out = {k: mf(t, p, self.z) for k, (t, p) in cfg.MF_RISK_DOMAIN.items()}

    def _defuzz(self, strength: dict) -> float:
        agg = np.zeros_like(self.z)
        for cons, w in strength.items():
            if w > 0:
                agg = np.maximum(agg, np.minimum(w, self.out[cons]))
        tot = agg.sum()
        return float((self.z * agg).sum() / tot) if tot > 0 else float("nan")

    def _collect(self, pairs):
        by = {}
        for cons, w in pairs:
            by.setdefault(cons, []).append(w)
        return {c: s_norm(ws, self.s) for c, ws in by.items()}

    def risk(self, d: float, lag: float) -> float:
        md = {k: float(mf(t, p, d)) for k, (t, p) in self.mf_d.items()}
        ml = {k: float(mf(t, p, lag)) for k, (t, p) in self.mf_lag.items()}
        pairs = [(c, md[a] * ml[b]) for a, b, c in cfg.RULES_DOMAIN]
        return self._defuzz(self._collect(pairs))

    def risk_ageless(self, d: float) -> float:
        md = {k: float(mf(t, p, d)) for k, (t, p) in self.mf_d.items()}
        return self._defuzz(self._collect([(c, md[a]) for a, c in cfg.RULES_AGELESS]))


# =====================================================================
# Điểm Nhánh 1
# =====================================================================

def weights(shapley):
    phi, _ = renormalize_for_missing(shapley, {}, DOMS)
    return np.array([phi[k] for k in DOMS])


def domain_matrix(Q, items):
    d = ts_external.domain_scores(Q, items)
    return np.column_stack([d[k] for k in DOMS])


def through_fis(D, fis: Mamdani):
    """Đưa từng d_i qua tầng 2 với LAG ≡ 0; chia 100 để về cùng thang [0, 1]."""
    ageless = set(cfg.AGELESS_DOMAINS)
    use_lag_table = getattr(cfg, "DHBT_USE_LAG_TABLE", False)
    cache = {}
    R = np.zeros_like(D)
    for j, dom in enumerate(DOMS):
        for i, v in enumerate(D[:, j]):
            key = (dom in ageless and not use_lag_table, round(float(v), 10))
            if key not in cache:
                cache[key] = (fis.risk_ageless(v) if key[0] else fis.risk(v, 0.0)) / 100.0
            R[i, j] = cache[key]
    return R


# =====================================================================
# Kiểm tra đơn điệu (Bảng 5.12)
# =====================================================================

def monotonicity(fis: Mamdani, step_d=None, step_lag=None):
    """Tổng mức GIẢM của risk khi một biến tăng, biến kia cố định, trên lưới đều.

    Định nghĩa này phải được ghi nguyên văn trong chú thích Bảng 5.12, vì con số
    phụ thuộc bước lưới: bước càng mịn, tổng càng lớn nếu có vi phạm.
    """
    sd = step_d or cfg.RES_D
    sl = step_lag or cfg.RES_LAG
    dg = np.arange(0.0, 1.0 + 1e-9, sd)
    lg = np.arange(0.0, cfg.LAG_MAX + 1e-9, sl)
    S = np.array([[fis.risk(d, l) for d in dg] for l in lg])
    viol_d = float(np.clip(S[:, :-1] - S[:, 1:], 0, None).sum())
    viol_l = float(np.clip(S[:-1, :] - S[1:, :], 0, None).sum())
    return viol_d, viol_l, S.shape


def hr(t):
    print()
    print("=" * 100)
    print(t)
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-mono", action="store_true", help="bỏ phần Bảng 5.12")
    ap.add_argument("--mono-step", type=float, default=None,
                    help="bước lưới cho kiểm tra đơn điệu (mặc định cfg.RES_D / RES_LAG)")
    args = ap.parse_args()

    fis = Mamdani(s=cfg.S_NORM)
    r00, r13 = fis.risk(0.0, 0.0), fis.risk(1.0, cfg.LAG_MAX)
    print(f"Kiểm chứng hệ suy diễn: risk(0,0) = {r00:.2f} (cfg {cfg.RISK_FLOOR_MAMDANI}); "
          f"risk(1,3) = {r13:.2f} (cfg {cfg.RISK_CEIL_MAMDANI})")
    if abs(r00 - cfg.RISK_FLOOR_MAMDANI) > 0.05 or abs(r13 - cfg.RISK_CEIL_MAMDANI) > 0.05:
        print("⚠ Không khớp config — kiểm tra cài đặt trước khi dùng số.")

    Q, y, meta = ts_external.load_poland()
    y = np.asarray(y).astype(int)
    V1 = getattr(cfg, "SHAPLEY_V1", cfg.SHAPLEY)

    hr("CƠ CHẾ — ánh xạ d → risk của tầng 2 khi LAG ≡ 0")
    for d in (0.0, 0.10, 0.25, 0.33, 0.45, 0.60, 0.80, 1.0):
        print(f"  d = {d:4.2f}  →  risk = {fis.risk(d, 0.0):6.2f}   "
              f"(DHBT, bảng không LAG: {fis.risk_ageless(d):6.2f})")

    results, rows_json = {}, []
    hr("BẢNG 5.11 (BẢN SỬA) — bỏ qua tầng 2 so với chạy tầng 2 với LAG ≡ 0")
    print(f"  {'Bộ mục':8s} {'Bộ Shapley':26s} {'Bỏ qua tầng 2':>14s} {'Qua tầng 2':>11s} "
          f"{'Hiệu':>8s} {'CI 95 %':>20s} {'p boot':>7s} {'p DeLong':>9s}")
    print("  " + "-" * 98)
    for tag, items in (("25 mục", None), ("10 mục", ts_external.QCHAT10_ITEMS)):
        D = domain_matrix(Q, items)
        R = through_fis(D, fis)
        flat = [(DOMS[j], float((D[:, j] <= cfg.MF_D["THIEU_HUT_NHE"][1][1] + 1e-12).mean()))
                for j in range(len(DOMS))]
        for sname, sh in (("tiên nghiệm (SHAPLEY_V1)", V1), ("hậu nghiệm (cfg.SHAPLEY)", cfg.SHAPLEY)):
            w = weights(sh)
            s_by, s_fis = D @ w, R @ w
            o, ci, p = paired_boot_repeats(y, s_fis[None], s_by[None], n_boot=args.boot, seed=SEED)
            dl = delong_paired(y, s_fis, s_by)
            a_by, a_fis = auc_fast(y, s_by), auc_fast(y, s_fis)
            print(f"  {tag:8s} {sname:26s} {a_by:14.4f} {a_fis:11.4f} {o:+8.4f} "
                  f"[{ci[0]:+.4f}; {ci[1]:+.4f}] {p:7.3f} {dl[2]:9.4f}")
            rows_json.append({"items": tag, "shapley": sname, "auc_bypass": a_by,
                              "auc_fis": a_fis, "diff": o, "lo": ci[0], "hi": ci[1],
                              "p_boot": p, "p_delong": dl[2],
                              "n_unique_bypass": int(len(np.unique(np.round(s_by, 10)))),
                              "n_unique_fis": int(len(np.unique(np.round(s_fis, 10))))})
        results[tag] = flat
    print(f"  p = 0,000 nghĩa là p < {1 / args.boot:.4f}.")

    hr("TỈ LỆ QUAN SÁT RƠI VÀO VÙNG PHẲNG d ≤ 0,45 (mọi giá trị cho cùng một risk)")
    for tag, flat in results.items():
        print(f"  {tag}: " + "; ".join(f"{k} {v:.0%}" for k, v in flat))
    for r in rows_json:
        if r["shapley"].startswith("hậu"):
            print(f"  {r['items']}: số giá trị điểm phân biệt {r['n_unique_bypass']} → "
                  f"{r['n_unique_fis']} sau tầng 2")

    mono = []
    if not args.no_mono:
        hr("BẢNG 5.12 (KIỂM TRA LẠI) — tổng mức vi phạm đơn điệu trên lưới (điểm, thang 100)")
        step = args.mono_step
        print(f"  Lưới: bước d = {step or cfg.RES_D}, bước LAG = {step or cfg.RES_LAG}; "
              "vi phạm = tổng các lần risk GIẢM khi một biến tăng")
        configs = [
            ("Hàm thuộc nguyên bản + hợp thành cực đại", cfg.MF_D_V1, cfg.MF_LAG_V1, "max"),
            ("Hàm thuộc nguyên bản + tổng chặn", cfg.MF_D_V1, cfg.MF_LAG_V1, "bounded_sum"),
            ("Phân hoạch đơn vị + hợp thành cực đại", cfg.MF_D, cfg.MF_LAG, "max"),
            ("Phân hoạch đơn vị + tổng xác suất", cfg.MF_D, cfg.MF_LAG, "prob_sum"),
            ("Phân hoạch đơn vị + tổng chặn", cfg.MF_D, cfg.MF_LAG, "bounded_sum"),
        ]
        for name, md, ml, s in configs:
            vd, vl, shape = monotonicity(Mamdani(md, ml, s), step, step)
            mono.append({"ten": name, "viol_d": vd, "viol_lag": vl, "grid": list(shape)})
            print(f"  {name:44s} theo d: {vd:9.3f}   theo LAG: {vl:9.3f}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "out", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fis_bypass.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"risk_00": r00, "risk_13": r13, "table_5_11": rows_json,
                   "flat_share": results, "table_5_12": mono,
                   "mono_step": args.mono_step}, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
