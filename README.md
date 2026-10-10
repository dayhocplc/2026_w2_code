# Monotonicity by construction in two-layer fuzzy aggregation — code

Code accompanying the manuscript *"Monotonicity by construction in two-layer fuzzy aggregation:
a stochastic-dominance condition for clipped Mamdani inference and convex learning of monotone
2-additive Choquet integrals"* (submitted to the International Journal of Approximate Reasoning).

It reproduces every number, table and figure of the paper.

## Layout

```
src/                         analysis code (Python 3.12)
  run_w2_mamdani_theorem.py  Mamdani layer: Lemma 1 checks, Tables 2–3, certificate (Algorithm 1),
                             reachable counterexample, random stress test
  run_w2_choquet_equivalence.py  Theorem 2: brute-force check of (G) <=> (C) <=> (R), Möbius formula,
                             necessity, complementarity at a fitted solution
  run_w2_external.py         Choquet layer on POL252, PARK195, BCW683, HEART303 (Tables 1, 4, 5; Fig. 4)
  run_w2_cost.py             solver cost study (Table 6, Fig. 5)
  fig_w2.py                  Figures 1–5
  w2_common.py               linear-constraint solvers and choquistic regression (2-additive re-implementation)
  run_g3_choquet_graph.py    ChoquetGraph: the proposed bound-constrained L-BFGS-B learner
  run_fis_bypass.py          reference Mamdani implementation (cross-check of Table 2)
  config.py, ts_*.py, p3_common.py, figstyle.py   shared model constants, loaders and statistics
tests/test_w2.py             fast sanity tests (pytest)
out/reports/                 result files produced for the paper (JSON + TXT)
out/figures/                 figures (PNG 300 dpi + PDF)
manuscript/                  build script: renders the manuscript from W2_main_src.md and the JSON results
data/                        NOT included — see data/README.md
```

Comments inside the analysis scripts are partly in Vietnamese (the project language); function
names, docstrings of the new W2 code paths and all outputs used in the paper are documented here.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
python scripts/fetch_data.py                    # UCI data sets + checksum check
# POL252: download manually, see data/README.md
```

## Reproduce

Run from the package root (set `PYTHONIOENCODING=utf-8` on Windows):

```bash
python src/run_w2_mamdani_theorem.py        # ~15 min, single process
python src/run_w2_choquet_equivalence.py    # ~2 min
W2_JOBS=4 python src/run_w2_external.py     # 200 outer folds; ~1–3 h depending on cores
OMP_NUM_THREADS=1 python src/run_w2_cost.py # ~30–60 min (SLSQP / trust-constr dominate)
python src/fig_w2.py
python manuscript/build_w2.py               # renders manuscript/W2_manuscript.md (+ .docx if pandoc is installed)
pytest -q tests                             # < 1 min
```

`run_w2_external.py` caches each outer fold in the system temporary directory
(`w2_external_cache_20260925_v1`); delete it to force a clean rerun. `W2_QUICK=2` runs only the first
two folds per data set.

Seeds: 20260925 for all W2 scripts (20260827 for `run_g3_choquet_graph.py`, whose POL252 domain-level
results are quoted in the case study). Timings in Table 6 depend on hardware; the ratios, objective values
and constraint margins do not.

## Key implementation facts

* **Mamdani layer.** Product conjunction, rules sharing a consequent aggregated by the bounded sum,
  clipping (min) implication, maximum across different consequents, centroid on a 0.5-step grid of
  [0, 100]. `GridMamdani` in `run_w2_mamdani_theorem.py` is vectorized and supports any number of inputs;
  it reproduces the independent implementation `run_fis_bypass.Mamdani` to three decimals.
* **Certificate (Algorithm 1).** `certify_D` enumerates all vertices of the line arrangement
  {w_c = t} ∪ {w_c = w_c'} on the 2-simplex and evaluates the affine numerator of the directional
  derivative in every incident cell.
* **Choquet layer.** `ChoquetGraph(lam, monotone=True)` solves
  min logloss(a + Σ s_i x_i + Σ m⁺_ij min(x_i,x_j) + Σ m⁻_ij max(x_i,x_j)) + λΣ(m⁺+m⁻) + γ/2‖m‖²
  over s, m⁺, m⁻ ≥ 0 with SciPy's L-BFGS-B; `monotone=False` gives the unconstrained comparator.
  `ChoquetLinCon` (trust-constr / SLSQP) and `Choquistic` (normalized parametrization, SLSQP) are in
  `w2_common.py`.

## Licence

*[licence to be chosen by the authors, e.g. MIT]*
