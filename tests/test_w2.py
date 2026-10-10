# -*- coding: utf-8 -*-
"""Fast sanity tests for the W2 code paths (pytest -q tests)."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as cfg                                                      # noqa: E402
from run_fis_bypass import Mamdani, mf, monotonicity                      # noqa: E402
from run_g3_choquet_graph import ChoquetGraph                             # noqa: E402
from run_w2_choquet_equivalence import (choquet_mobius, choquet_sort,     # noqa: E402
                                        closed_form, grabisch_bruteforce, random_measure)
from run_w2_mamdani_theorem import (D_ORDER, L_ORDER, GridMamdani,        # noqa: E402
                                    cons_matrix, fosd_check, rules_array)


def test_lemma1_bounded_sum_is_fosd():
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    s = GridMamdani(z, cons_matrix(z), rules_array(), snorm="bounded_sum")
    r = fosd_check(s, cfg.MF_D, cfg.MF_LAG, nd=41, nl=61)
    assert r["max_tail_drop_d"] < 1e-12 and r["max_tail_drop_lag"] < 1e-12
    assert r["max_abs_sum_w_minus_1"] < 1e-12


def test_max_aggregation_breaks_fosd():
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    s = GridMamdani(z, cons_matrix(z), rules_array(), snorm="max")
    r = fosd_check(s, cfg.MF_D, cfg.MF_LAG, nd=41, nl=61)
    assert r["max_tail_drop_d"] > 1e-3


def test_gridmamdani_matches_reference():
    z = np.arange(0, 100 + 1e-9, cfg.RES_RISK)
    s = GridMamdani(z, cons_matrix(z), rules_array(), snorm="bounded_sum")
    ref = Mamdani(s="bounded_sum")
    for d, lag in [(0.0, 0.0), (0.3, 0.7), (0.6, 1.8), (1.0, 3.0)]:
        w = s.weights([np.array([mf(*cfg.MF_D[k], d) for k in D_ORDER]),
                       np.array([mf(*cfg.MF_LAG[k], lag) for k in L_ORDER])])
        assert abs(s.defuzz(w) - ref.risk(d, lag)) < 1e-9


def test_project_system_monotone_on_coarse_grid():
    vd, vl, _ = monotonicity(Mamdani(s="bounded_sum"), 0.05, 0.05)
    assert vd < 1e-9 and vl < 1e-9


def test_theorem2_equivalence_small():
    rng = np.random.default_rng(0)
    for p in (3, 4, 5):
        for kind in ("monotone", "boundary", "free"):
            for _ in range(30):
                m, M = random_measure(rng, p, kind)
                assert abs(grabisch_bruteforce(m, M) - closed_form(m, M)) < 1e-10
                x = rng.uniform(0, 1, p)
                assert abs(choquet_sort(x, m, M) - choquet_mobius(x, m, M)) < 1e-12


def test_choquetgraph_is_monotone():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (300, 6))
    y = (X[:, 0] + np.minimum(X[:, 1], X[:, 2]) - 0.5 * np.maximum(X[:, 3], X[:, 4])
         + rng.normal(0, 0.3, 300) > 0.6).astype(int)
    g = ChoquetGraph(1e-3, True).fit(X, y)
    assert g.min_monotone_margin_ >= -1e-12
    for _ in range(200):
        a = rng.uniform(0, 1, 6)
        b = np.minimum(1, a + rng.uniform(0, 0.3, 6))
        assert g.decision_function(b[None])[0] >= g.decision_function(a[None])[0] - 1e-12
