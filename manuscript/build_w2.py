# -*- coding: utf-8 -*-
"""
build_w2.py — Dựng bản thảo W2 (International Journal of Approximate Reasoning, Elsevier) từ
W2_main_src.md + kết quả đã chạy, rồi tự kiểm.

    .venv/Scripts/python.exe manuscript/W2/build_w2.py              # dựng + kiểm cục bộ
    .venv/Scripts/python.exe manuscript/W2/build_w2.py --check-doi  # thêm kiểm DOI (Crossref/DataCite)

Đầu vào:  W2_main_src.md, refs_w2.py,
          out/reports/{w2_mamdani_theorem,w2_choquet_equivalence,w2_external,w2_cost,g3_choquet_graph}.json
Đầu ra:   W2_manuscript.md/.docx, W2_selfcheck.txt

Mọi con số trong văn bản sinh từ biểu thức {{...}} (đánh giá trên JSON) hoặc từ bảng sinh tự động;
số thập phân gõ tay còn sót được liệt kê trong phần tự kiểm để rà lại.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REP = os.path.join(ROOT, "out", "reports")
FIG = os.path.join(ROOT, "out", "figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from refs_w2 import REFS  # noqa: E402

PANDOC = os.path.expandvars(r"%LOCALAPPDATA%\Pandoc\pandoc.exe")
MINUS = "−"


def load(fn):
    with open(os.path.join(REP, fn), encoding="utf-8") as f:
        return json.load(f)


MT = load("w2_mamdani_theorem.json")
CE = load("w2_choquet_equivalence.json")
EX = load("w2_external.json")
CO = load("w2_cost.json")
G3 = load("g3_choquet_graph.json")
DSETS = ["POL252", "PARK195", "BCW683", "HEART303"]
DS_NAME = {"POL252": "POL252", "PARK195": "PARK195", "BCW683": "BCW683", "HEART303": "HEART303"}
METHODS = ["choquet_graph", "choquet_unconstrained", "additive_monotone", "choquistic_ml",
           "lgbm_monotone", "linear_l1_logistic", "pairwise_l1_logistic"]
M_NAME = {"choquet_graph": "Monotone 2-additive Choquet (proposed)",
          "choquet_unconstrained": "2-additive Choquet, unconstrained",
          "additive_monotone": "Additive, monotone",
          "choquistic_ml": "Choquistic regression, ML",
          "lgbm_monotone": "LightGBM, monotone constraints",
          "linear_l1_logistic": "L1 logistic",
          "pairwise_l1_logistic": "L1 logistic + pairwise products"}


# ================================================================== định dạng
def num(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if s.lstrip("+-").strip("0.") == "":
        s = f"{0:.{d}f}"
    return s.replace("-", MINUS)


def sci(x, d=1):
    """1.2 × 10⁻¹⁶ dạng văn bản."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{num(m, d)} × 10{sup}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def pval(p):
    return "< 0.001" if p < 0.001 else num(p, 3)


# ---- trợ giúp dùng trong {{...}}
def DSX(ds):
    return EX["datasets"][ds]


def AU(ds, m, d=3):
    return num(DSX(ds)["cv_auc"][m]["mean"], d)


def AUSD(ds, m):
    s = DSX(ds)["cv_auc"][m]
    return f"{num(s['mean'])} ± {num(s['sd'])}"


def CMP(ds, b):
    return next(c for c in DSX(ds)["comparisons"] if c["b"] == b)


def DEL(ds, b):
    c = CMP(ds, b)
    return (f"ΔAUC = {num(c['delta'], 3, True)}, 95% CI {num(c['ci95'][0], 3, True)} to "
            f"{num(c['ci95'][1], 3, True)}, adjusted p {'<' if c['p_bh'] < 0.001 else '='} "
            f"{'0.001' if c['p_bh'] < 0.001 else num(c['p_bh'], 3)}")


def MONO(ds, k):
    return DSX(ds)["monotonicity"][k]


def COST(ds, k):
    return DSX(ds)["cost"][k]


def CRUN(case):
    return next(r for r in CO["runs"] if r["case"] == case)


def max_drop():
    """Mức giảm đỉnh-đáy lớn nhất của risk theo d (LAG cố định), Ruspini + cực đại."""
    import config as cfg
    from run_fis_bypass import mf
    from run_w2_mamdani_theorem import D_ORDER, L_ORDER, GridMamdani, cons_matrix, rules_array
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    s = GridMamdani(z, cons_matrix(z), rules_array(), snorm="max")
    dg = np.arange(0, 1 + 1e-9, 0.01)
    Md = np.array([mf(*cfg.MF_D[k], dg) for k in D_ORDER]).T
    best, arg = 0.0, None
    for lag in np.arange(0, 3 + 1e-9, 0.01):
        Ml = np.array([mf(*cfg.MF_LAG[k], lag) for k in L_ORDER])
        y = np.array([s.defuzz(s.weights([Md[i], Ml])) for i in range(len(dg))])
        drop = float(np.max(np.maximum.accumulate(y) - y))
        if drop > best:
            best, arg = drop, float(lag)
    return {"max": best, "lag": arg}


MAXDROP = max_drop()


def v1_mean_change():
    """Thay đổi lớn nhất của AUC TRUNG BÌNH (50 fold) do bước kiểm KKT, mọi bộ × 3 mô hình Choquet."""
    best = 0.0
    for ds, R in EX["folds"].items():
        for k in ("choquet_graph", "choquet_unconstrained", "additive_monotone"):
            d = np.mean([r[k] - r["v1"][k] for r in R if "v1" in r])
            best = max(best, abs(float(d)))
    return best


def unguarded_p30():
    """Giá trị mục tiêu khi KHÔNG kiểm KKT (L-BFGS-B dừng giả) ở mô phỏng p = 30."""
    import run_g3_choquet_graph as G
    from run_w2_cost import LAM, penalized_objective, simulate
    X, y = simulate(30)
    g = G.ChoquetGraph(LAM, True)
    g.kkt_restart = False
    g.fit(X, y)
    return penalized_objective(g, X, y, LAM, G.GAMMA)


V1CHANGE = v1_mean_change()
UNGUARDED30 = unguarded_p30()


# ================================================================== bảng
def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def table1():
    rows = []
    desc = {"POL252": ("Q-CHAT-25, Polish toddlers", "25 ordinal items (0–4)", "Autism diagnosis (community)"),
            "PARK195": ("UCI Parkinson's voice", "22 continuous acoustic measures", "Parkinson's disease"),
            "BCW683": ("Wisconsin breast cancer (original)", "9 ordinal cytology items (1–10)", "Malignancy"),
            "HEART303": ("Cleveland heart disease", "13 mixed attributes (4 nominal)", "Angiographic disease")}
    for ds in DSETS:
        d = DSX(ds)
        rows.append([ds, desc[ds][0], d["n"], d["p"], d["p"] * (d["p"] - 1) // 2, pct(d["prevalence"], 1),
                     desc[ds][1], desc[ds][2]])
    return ("**Table 1** Data sets. p: number of attributes; pairs: number of pairwise interaction "
            "coefficients of the 2-additive model; prevalence: proportion of positive cases.\n\n"
            + md_table(["Data set", "Source", "n", "p", "Pairs", "Prevalence", "Attributes", "Outcome"],
                       rows, ["l", "l", "r", "r", "r", "r", "l", "l"]))


CONF = [("original MFs + max", "Original (not Ruspini)", "Maximum S_M"),
        ("original MFs + bounded sum", "Original (not Ruspini)", "Bounded sum S_L"),
        ("Ruspini + max", "Ordered Ruspini", "Maximum S_M"),
        ("Ruspini + probabilistic sum", "Ordered Ruspini", "Probabilistic sum S_P"),
        ("Ruspini + bounded sum", "Ordered Ruspini", "Bounded sum S_L")]


def table2():
    rows = []
    for key, part, agg in CONF:
        f = MT["lemma_fosd"][key]
        t = MT["table_5_14"][key]
        rows.append([part, agg, num(f["max_abs_sum_w_minus_1"], 3),
                     num(f["max_tail_drop_d"], 3), num(f["max_tail_drop_lag"], 3),
                     num(t["viol_d"], 1), num(t["viol_lag"], 1)])
    return ("**Table 2** Mamdani layer with product conjunction and clipping implication: which hypotheses of "
            "Lemma 1 hold, and the resulting monotonicity violation. max|Σw − 1|: largest deviation of the "
            "total consequent strength from one; tail decrease: largest decrease of a tail T_k(w(x)) between "
            "neighbouring grid points (zero if and only if w(x) is FOSD-nondecreasing on the grid); violation: "
            "total decrease of the domain risk (0–100 scale) over a 101 × 301 grid when d or LAG increases.\n\n"
            + md_table(["Input partitions", "Rules sharing a consequent", "max|Σw − 1|",
                        "Tail decrease, d", "Tail decrease, LAG", "Violation, d", "Violation, LAG"],
                       rows, ["l", "l", "r", "r", "r", "r", "r"]))


def table3():
    rows = []
    name = {"min": "clipping (min)", "product": "scaling (product)"}
    for v in MT["implication_variants"]:
        rows.append(["smooth" if v["rules"] == "smooth" else "non-smooth",
                     v["conjunction"], name[v["implication"]],
                     "maximum" if v["aggregation"] == "max" else "bounded sum",
                     num(v["viol_d"], 1), num(v["viol_lag"], 1)])
    return ("**Table 3** Monotonicity violation (total decrease over the 101 × 301 grid, 0–100 scale) for "
            "ordered Ruspini partitions, by rule base, conjunction, implication and aggregation of rules sharing a "
            "consequent. The non-smooth rule base jumps from LOW to HIGH between adjacent cells.\n\n"
            + md_table(["Rule base", "Conjunction", "Implication", "Aggregation", "Violation, d", "Violation, LAG"],
                       rows, ["l", "l", "l", "l", "r", "r"]))


def table4():
    rows = []
    for m in METHODS:
        r = [M_NAME[m]]
        for ds in DSETS:
            s = DSX(ds)["cv_auc"][m]
            cell = f"{num(s['mean'])} ({num(s['sd'])})"
            if m != "choquet_graph":
                c = CMP(ds, m)
                if c["reject"]:
                    cell += "*"
            r.append(cell)
        rows.append(r)
    return ("**Table 4** Cross-validated AUC, mean (SD) over 5-fold cross-validation repeated 10 times. "
            "\\* the difference from the proposed model is significant (Nadeau–Bengio corrected t-test, "
            "Benjamini–Hochberg adjustment over six comparisons within the data set, adjusted p < 0.05).\n\n"
            + md_table(["Method"] + DSETS, rows))


def table5():
    rows = []
    for ds in DSETS:
        m = DSX(ds)["monotonicity"]
        c = DSX(ds)["cost"]
        rows.append([ds, pct(m["unc_folds_nonmonotone"]), num(m["unc_mean_nonmonotone_nodes"], 1),
                     pct(m["folds_active"]), num(m["mean_active"], 1),
                     num(m["mean_edges"], 1), num(m["mean_edges_unc"], 1), num(m["mean_flip"], 1),
                     c["n_slsqp_folds"], sci(c["train_logloss_gap_reparam_minus_choquistic"]["max"]),
                     num(c["test_auc_gap_reparam_ml_minus_slsqp"]["max_abs"], 4)])
    return ("**Table 5** Monotonicity diagnostics over the 50 outer folds and agreement with choquistic regression. "
            "Unconstrained: folds in which the unconstrained 2-additive model violates condition (C) at one or more "
            "nodes, and the mean number of such nodes; active: folds in which a constraint of the monotone model is "
            "binding at a node with negative interactions, and the mean number of such nodes; edges: mean number of "
            "nonzero interactions; flipped: mean number of attributes reversed by the orientation step. The last three "
            "columns compare, on the folds where both were run, the unpenalized maximum-likelihood fit obtained by "
            "the reparametrized solver with choquistic regression fitted by SLSQP in its original parametrization: "
            "largest difference in training log-loss (reparametrized minus SLSQP; negative favours the "
            "reparametrized solver) and largest absolute difference in test AUC.\n\n"
            + md_table(["Data set", "Unconstrained: folds non-monotone", "Non-monotone nodes",
                        "Monotone: folds active", "Active nodes", "Edges", "Edges, unconstrained",
                        "Flipped", "SLSQP folds", "Max Δ log-loss", "Max |ΔAUC|"], rows))


def table6():
    rows = []
    runs = sorted(CO["runs"], key=lambda r: r["p"])
    for r in runs:
        R = r["rows"]
        objs = [R[k]["obj"] for k in ("reparam_lbfgsb", "lincon_trust", "lincon_slsqp") if R[k]]
        best = min(objs)

        def cell(k):
            v = R[k]
            if v is None:
                return "—"
            return f"{num(v['sec'], 2)}"

        def gap(k):
            v = R[k]
            return "—" if v is None else sci(max(v["obj"] - best, 0.0)) if v["obj"] - best > 1e-15 else "0"
        a = R["reparam_lbfgsb_ml"]; b = R["choquistic_slsqp"]
        rows.append([r["case"].replace("sim_p", "Simulated, p = ").replace("POL252", "POL252, p = 25"),
                     r["n_vars"], cell("reparam_lbfgsb"), cell("lincon_trust"), cell("lincon_slsqp"),
                     gap("lincon_trust"), gap("lincon_slsqp"),
                     num(a["sec"], 2), "—" if b is None else num(b["sec"], 2),
                     "—" if b is None else sci(a["nll"] - b["nll"]) if abs(a["nll"] - b["nll"]) > 1e-15 else "0"])
    return ("**Table 6** Wall-clock time in seconds (one thread). Penalized problem (λ = 0.003, γ = 0.001): "
            "reparametrized L-BFGS-B versus linear constraints with trust-constr or SLSQP, and the excess objective "
            "value of the latter over the best of the three. Maximum likelihood: reparametrized L-BFGS-B versus "
            "choquistic regression (normalized parametrization, SLSQP), and the difference in training log-loss "
            "(negative favours the reparametrized solver). —: not run after exceeding the 900 s budget at a smaller "
            "size.\n\n"
            + md_table(["Problem", "Parameters", "L-BFGS-B", "trust-constr", "SLSQP", "Excess, trust-constr",
                        "Excess, SLSQP", "ML: L-BFGS-B", "ML: choquistic SLSQP", "ML: Δ log-loss"], rows))


# ================================================================== biểu thức {{...}}
EXPR = re.compile(r"\{\{(.+?)\}\}", flags=re.S)
GENERATED: list[str] = []
TABLES = {"TABLE1": table1, "TABLE2": table2, "TABLE3": table3, "TABLE4": table4, "TABLE5": table5,
          "TABLE6": table6}


def render(text):
    env = dict(MT=MT, CE=CE, EX=EX, CO=CO, G3=G3, num=num, sci=sci, pct=pct, pval=pval, AU=AU, AUSD=AUSD,
               DEL=DEL, CMP=CMP, MONO=MONO, COST=COST, CRUN=CRUN, MAXDROP=MAXDROP, V1CHANGE=V1CHANGE, UNGUARDED30=UNGUARDED30, DSX=DSX, np=np,
               max=max, min=min, abs=abs, len=len, round=round, sum=sum, int=int)

    def rep(m):
        expr = m.group(1).strip()
        if expr in TABLES:
            return TABLES[expr]()
        v = eval(expr, env)  # noqa: S307 — biểu thức trong tệp nguồn của chính tác giả
        s = v if isinstance(v, str) else (num(v) if isinstance(v, float) else str(v))
        GENERATED.append(s)
        return s
    return EXPR.sub(rep, text)


# ================================================================== trích dẫn
CITE = re.compile(r"\s?\[\[([a-z0-9_,\s]+)\]\]")


def comp(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


def number_citations(text, order):
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        for k in keys:
            if k not in REFS:
                raise KeyError(f"Khoá trích dẫn không có trong refs_w2.py: {k}")
            if k not in order:
                order.append(k)
        return " [" + comp([order.index(k) + 1 for k in keys]) + "]"
    return CITE.sub(rep, text)


def references_md(order):
    return "\n".join(f"[{i}] {REFS[k][0]}" + (f" https://doi.org/{REFS[k][1]}" if REFS[k][1] else "")
                     + "\n" for i, k in enumerate(order, 1))


def check_dois(order):
    bad, n = [], 0
    H = {"User-Agent": "w2-build (mailto:azauto.vn@gmail.com)"}
    for k in order:
        s, doi, src = REFS[k]
        if not doi:
            continue
        n += 1
        try:
            if src == "DC":
                a = json.load(urllib.request.urlopen(f"https://api.datacite.org/dois/{doi}", timeout=30))["data"]["attributes"]
                year, fam = str(a.get("publicationYear")), a["creators"][0].get("familyName") or a["creators"][0]["name"].split(",")[0]
            else:
                m = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers=H),
                                                     timeout=30))["message"]
                d = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
                year, fam = str(d), (m.get("author") or [{"family": ""}])[0].get("family", "")
            probs = []
            if year not in s:
                probs.append(f"năm {year} không có trong trích dẫn")
            norm = s.lower().replace("é", "e").replace("ź", "z").replace("ü", "u")
            famn = fam.lower().replace("é", "e").replace("ź", "z").replace("ü", "u")
            if famn and famn.split()[-1] not in norm:
                probs.append(f"họ tác giả đầu '{fam}' không khớp")
            if probs:
                bad.append(f"{k}: " + "; ".join(probs))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{k}: không tra được DOI {doi} ({e})")
    return n, bad


# ================================================================== tự kiểm
def words(s):
    s = re.sub(r"\[[\d,\s–]+\]", "", s)
    s = re.sub(r"^\|.*\|$", "", s, flags=re.M)
    s = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", "", s)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    return len(re.findall(r"[A-Za-z0-9α-ωΑ-Ωλ][\w\-–.·'’]*", s))


def section(md, name):
    m = re.search(rf"^# {re.escape(name)}\s*$(.*?)(?=^# )", md, flags=re.S | re.M)
    return m.group(1) if m else ""


FIG_FILES = ["W2_fig1_architecture.png", "W2_fig2_mamdani.png", "W2_fig3_defuzzifier.png",
             "W2_fig4_auc.png", "W2_fig5_cost.png"]


def place_figures(md):
    head, sep, tail = md.partition("# References")
    legends = dict(re.findall(r"^\*\*Fig\. (\d)\*\* (.*?)$", tail, flags=re.M))
    paras = head.split("\n\n")
    for i, fn in enumerate(FIG_FILES, 1):
        idx = next(k for k, p in enumerate(paras)
                   if re.search(rf"Fig\. {i}(?![0-9])", p) and not p.lstrip().startswith(("|", "**Table", "![", "#", "**Fig")))
        paras[idx] += f"\n\n![]({os.path.join(FIG, fn).replace(os.sep, '/')}){{width=16cm}}\n\n**Fig. {i}** {legends[str(i)]}"
    return "\n\n".join(paras) + sep + tail


def main() -> int:
    check_doi = "--check-doi" in sys.argv
    src = open(os.path.join(HERE, "W2_main_src.md"), encoding="utf-8").read()
    for key, fn in (("RESULTS", "W2_results_src.md"), ("RESULTS_EXT", "W2_results_ext_src.md"), ("DISCUSSION", "W2_discussion_src.md"),
                    ("CONCLUSIONS", "W2_conclusions_src.md")):
        part = open(os.path.join(HERE, fn), encoding="utf-8").read().strip()
        src = src.replace("{{" + key + "}}", part)
    body = render(src)
    order: list[str] = []
    body = number_citations(body, order)
    main_md = place_figures(body.replace("{REFERENCES}", references_md(order)))
    open(os.path.join(HERE, "W2_manuscript.md"), "w", encoding="utf-8").write(main_md)

    rep = ["W2 — TỰ KIỂM BẢN THẢO", ""]
    title = re.search(r'title: "(.*?)"', src).group(1)
    rep.append(f"Tiêu đề: {len(title.split())} từ, {len(title)} ký tự")
    hl = re.findall(r"^- (.*)$", section(main_md, "Highlights"), flags=re.M)
    rep.append(f"Highlights: {len(hl)} mục (IJAR: 3–5, ≤ 85 ký tự): " +
               ", ".join(f"{len(h)}{'' if len(h) <= 85 else ' VƯỢT'}" for h in hl))
    ab = section(main_md, "Abstract")
    aw = words(ab.split("**Keywords:**")[0])
    rep.append(f"Tóm tắt: {aw} từ (Elsevier khuyến nghị ≤ 250) {'OK' if aw <= 250 else 'VƯỢT'}")
    kw = re.search(r"\*\*Keywords:\*\* (.*)", main_md).group(1)
    nkw = len([k for k in kw.split(";") if k.strip()])
    rep.append(f"Từ khoá: {nkw} (Elsevier: 1–7) {'OK' if 1 <= nkw <= 7 else 'KIỂM TRA'}")
    secs = ["1. Introduction", "2. Preliminaries", "3. The Mamdani layer", "4. The Choquet layer",
            "5. The two-layer system", "6. Experimental design", "7. Results", "8. Discussion", "9. Conclusions"]
    tot = 0
    for s in secs:
        w = words(section(main_md, s))
        tot += w
        rep.append(f"  {s}: {w} từ")
    rep.append(f"Thân bài (1→9): {tot} từ")
    rep.append(f"Tài liệu được trích: {len(order)} / {len(REFS)}; không dùng: " + (", ".join(k for k in REFS if k not in order) or "không"))
    for n in FIG_FILES:
        rep.append(f"  hình {n}: {'có' if os.path.exists(os.path.join(FIG, n)) else 'THIẾU'}")
    pre = main_md.split("# References")[0]
    for pat, lim in [(r"Fig\. (\d)", 5), (r"Table (\d)", 6)]:
        used = sorted(set(int(x) for x in re.findall(pat, pre)))
        rep.append(f"Tham chiếu {pat}: {used} {'OK' if used == list(range(1, lim + 1)) else 'KIỂM TRA'}")
    body_only = pre.split("# 1. Introduction")[1]
    seen = list(dict.fromkeys(int(x) for x in re.findall(r"Fig\. (\d)", body_only)))
    rep.append(f"Thứ tự nhắc hình lần đầu: {seen} {'OK' if seen == sorted(seen) else 'KIỂM TRA'}")
    firstt = list(dict.fromkeys(int(x) for x in re.findall(r"Table (\d)", body_only)))
    rep.append(f"Thứ tự nhắc bảng lần đầu: {firstt} {'OK' if firstt == sorted(firstt) else 'KIỂM TRA'}")
    for pat, name in [(r"Lemma (\d)", "Lemma"), (r"Theorem (\d)", "Theorem"), (r"Proposition (\d)", "Proposition"),
                      (r"Corollary (\d)", "Corollary"), (r"Algorithm (\d)", "Algorithm")]:
        defined = re.findall(rf"\*\*{name} (\d)", pre)
        used = sorted(set(re.findall(pat, pre)))
        rep.append(f"{name}: định nghĩa {defined}; được nhắc {used} "
                   f"{'OK' if set(used) <= set(defined) and defined == sorted(defined) else 'KIỂM TRA'}")
    eqs = re.findall(r"\\tag\{(\d+)\}", pre)
    rep.append(f"Phương trình đánh số: {eqs} {'OK' if eqs == [str(i) for i in range(1, len(eqs) + 1)] else 'KIỂM TRA'}")
    raw = EXPR.sub(" ", src)
    raw = re.sub(r"^\|.*\|$", "", raw, flags=re.M)
    raw = re.sub(r"\$\$.*?\$\$", " ", raw, flags=re.S)
    hand = re.findall(r"(?<![\w.])−?\d+\.\d+(?!\w|\.\d)", raw)
    rep.append(f"Số sinh tự động: {len(GENERATED)}; số thập phân gõ tay: {len(hand)} → {sorted(set(hand))}")
    if check_doi:
        n, bad = check_dois(order)
        rep.append(f"Kiểm DOI (Crossref/DataCite): {n} DOI; lỗi: {len(bad)}")
        rep += ["  " + b for b in bad]
    left = re.findall(r"\{\{.*?\}\}|\[\[[^\]]+\]\]|\{REFERENCES\}", main_md)
    rep.append(f"Placeholder còn sót: {left if left else 'không'}")
    todo = re.findall(r"\*\[[^\]]+\]\*", main_md)
    rep.append(f"Mục chờ tác giả điền ({len(todo)}): " + " | ".join(t[2:-2][:60] for t in todo))
    r = subprocess.run([PANDOC, "W2_manuscript.md", "-o", "W2_manuscript.docx", "--resource-path", FIG],
                       cwd=HERE, capture_output=True, text=True, encoding="utf-8")
    rep.append(f"pandoc W2_manuscript.docx: {'OK' if r.returncode == 0 else 'LỖI ' + r.stderr[:300]}"
               + (f" (cảnh báo: {r.stderr.strip()[:200]})" if r.returncode == 0 and r.stderr.strip() else ""))
    out = "\n".join(rep)
    open(os.path.join(HERE, "W2_selfcheck.txt"), "w", encoding="utf-8").write(out + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
