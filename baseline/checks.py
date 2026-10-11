"""Robustness and calibration checks used by replicate.py (figures 6 and 7).

multistart: refit every candidate from random starting points, to show that a poor chi2 is the
            model's fault and not the optimiser's.
loro_calibration: leave one run out, refit M2/M4 on the other four, predict the held-out run with
            the same procedure as predict.py, and count how often the truth lands in the 90% interval.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from model_fit import (CANDIDATES, NOISE_B, SPECIES, Conditions, bic_weights, bounds, draw_params,
                       fit_model, simulate)

N_STARTS = {"M2": 6, "M4": 6, "M1": 3, "M3": 3, "M5": 3, "M6": 3}
N_DRAWS = 200


def _workers():
    return max(1, min(4, os.cpu_count() or 1))


def _one_start(args):
    model, data, x0 = args
    try:
        return model, float(fit_model(model, data, x0=x0)["chi2"])
    except Exception:
        return model, float("nan")


def multistart(data, seed=7):
    """chi2 reached from random starts: log10 k(75 C) in [-4.5, -0.5], Ea in [35, 85] kJ/mol."""
    rng = np.random.default_rng(seed)
    jobs = []
    for m, n in N_STARTS.items():
        names = CANDIDATES[m].param_names
        for _ in range(n):
            x0 = [rng.uniform(-4.5, -0.5) if nm.startswith("k") else rng.uniform(35, 85) for nm in names]
            jobs.append((m, data, x0))
    with ProcessPoolExecutor(max_workers=_workers()) as ex:
        res = list(ex.map(_one_start, jobs))
    return pd.DataFrame(res, columns=["model", "chi2"])


def _predict_run(fits, w, g, rng):
    """90% interval for one run, same recipe as predict.py (mixture, parameter draw, noise)."""
    T, A0, B0 = (float(g[c].iloc[0]) for c in ("temperature_C", "A0_M", "B0_M"))
    times = np.sort(g["time_min"].unique())
    models = list(fits)
    probs = np.array([w[m] for m in models])
    draws = {sp: [] for sp in SPECIES}
    tries = 0
    while len(draws["A"]) < N_DRAWS and tries < 3 * N_DRAWS:
        tries += 1
        m = models[rng.choice(len(models), p=probs)]
        try:
            sim = simulate(m, Conditions(T, A0, B0, times), draw_params(m, fits[m], rng))
        except Exception:
            continue
        for sp in SPECIES:
            v = sim[sp].to_numpy()
            rel, ab = NOISE_B[sp]
            draws[sp].append(np.clip(v + np.hypot(rel * v, ab) * rng.standard_normal(len(v)), 0, None))
    out = {}
    for sp in SPECIES:
        lo, hi = np.percentile(np.array(draws[sp]), [5, 95], axis=0)
        out[sp] = (dict(zip(times, lo)), dict(zip(times, hi)))
    return out


def _one_fold(args):
    held, data, seed = args
    train = data[data["run_id"] != held]
    fits = {m: fit_model(m, train) for m in ("M2", "M4")}
    w = bic_weights(fits)
    g = data[data["run_id"] == held]
    pred = _predict_run(fits, w, g, np.random.default_rng(seed))
    rows = []
    for sp in SPECIES:
        lo, hi = pred[sp]
        gg = g[(g["species"] == sp) & (~g["cens"])]          # censored Q has no number to check
        for t, y in zip(gg["time_min"], gg["value"]):
            rows.append({"run": held, "species": sp, "inside": bool(lo[t] <= y <= hi[t])})
    return rows


def loro_calibration(data, seed=11):
    runs = sorted(data["run_id"].unique())
    jobs = [(r, data, seed + i) for i, r in enumerate(runs)]
    with ProcessPoolExecutor(max_workers=_workers()) as ex:
        folds = list(ex.map(_one_fold, jobs))
    return pd.DataFrame([row for f in folds for row in f])
