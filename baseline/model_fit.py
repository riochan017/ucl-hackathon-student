"""Shared System B fitting code used by predict.py, suggest.py and replicate.py."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import log_ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from candidates import CANDIDATES, Conditions, simulate  # noqa: E402

DATA_B = os.path.join(ROOT, "data", "system_B", "measurements.csv")
HELDOUT = os.path.join(ROOT, "data", "heldout_conditions.csv")
SUBMISSION = os.path.join(ROOT, "submission")

# Noise model from data/README.md: sigma = sqrt((rel * value)^2 + abs^2)
NOISE_B = {"A": (0.03, 3.5e-3), "B": (0.03, 3.5e-3), "P": (0.03, 3.5e-3), "Q": (0.15, 2.2e-5)}
LOQ = 3.0e-5
SPECIES = ["A", "B", "P", "Q"]

# Starting point for the fit: best values from the earlier 20-start search.
# k entries are log10(k at 75 C); Ea entries are kJ/mol.
START = {
    "M2": [np.log10(0.0304), 66.2, np.log10(0.0201), 57.9, np.log10(1.94e-5), 59.8],
    "M4": [np.log10(0.0304), 66.2, np.log10(0.0201), 57.9, np.log10(2.38e-5), 59.2],
}


def ensure_submission_dir():
    os.makedirs(SUBMISSION, exist_ok=True)
    os.makedirs(os.path.join(SUBMISSION, "figures"), exist_ok=True)


def load_b():
    """System B data. Censored (<LOQ) rows are kept and flagged in the 'cens' column."""
    b = pd.read_csv(DATA_B)
    b["cens"] = b["censored"].astype(str).str.lower().eq("true")
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    return b


def bounds(model):
    names = CANDIDATES[model].param_names
    lo = np.array([-6.0 if n.startswith("k") else 20.0 for n in names])
    hi = np.array([2.0 if n.startswith("k") else 130.0 for n in names])
    return lo, hi


def to_params(model, x):
    names = CANDIDATES[model].param_names
    return {n: (10 ** x[i] if n.startswith("k") else x[i]) for i, n in enumerate(names)}


def make_resid(model, data):
    """Misfit in noise-widths for every measurement; censored Q uses the normal tail probability."""
    n_tot = len(data)

    def resid(x):
        p = to_params(model, x)
        out = []
        try:
            for _, g in data.groupby("run_id"):
                T, A0, B0 = g["temperature_C"].iloc[0], g["A0_M"].iloc[0], g["B0_M"].iloc[0]
                times = np.sort(g["time_min"].unique())
                sim = simulate(model, Conditions(T, A0, B0, times), p).set_index("time_min")
                for sp in SPECIES:
                    gg = g[g["species"] == sp]
                    s = sim.loc[gg["time_min"].values, sp].to_numpy()
                    y = gg["value"].to_numpy()
                    c = gg["cens"].to_numpy()
                    rel, ab = NOISE_B[sp]
                    sigma = np.hypot(rel * np.where(c, s, y), ab)
                    r_obs = (s - np.nan_to_num(y)) / sigma
                    r_cen = np.sqrt(-2 * log_ndtr((LOQ - s) / sigma) + 1e-12)
                    out.append(np.where(c, r_cen, r_obs))
            return np.concatenate(out)
        except Exception:
            return np.full(n_tot, 1e3)

    return resid


def fit_model(model, data):
    lo, hi = bounds(model)
    sol = least_squares(make_resid(model, data), np.array(START[model]),
                        bounds=(lo, hi), max_nfev=300)
    chi2 = 2 * sol.cost
    cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    cov = (cov + cov.T) / 2
    return {"x": sol.x, "chi2": chi2, "bic": chi2 + len(sol.x) * np.log(len(data)), "cov": cov}


def fit_all(data, models=("M2", "M4")):
    return {m: fit_model(m, data) for m in models}


def bic_weights(fits):
    bic = np.array([f["bic"] for f in fits.values()])
    w = np.exp(-0.5 * (bic - bic.min()))
    w /= w.sum()
    return dict(zip(fits.keys(), w))


def draw_params(model, fit, rng):
    """One plausible parameter set, drawn from the fit's (Laplace) uncertainty."""
    lo, hi = bounds(model)
    x = np.clip(rng.multivariate_normal(fit["x"], fit["cov"]), lo, hi)
    return to_params(model, x)
