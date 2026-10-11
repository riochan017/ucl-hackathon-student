"""just predict -> submission/predictions.csv: M2/M4 mixture with 90% prediction intervals."""
import os

import numpy as np
import pandas as pd

from model_fit import (HELDOUT, NOISE_B, SPECIES, SUBMISSION, Conditions, bic_weights,
                       draw_params, ensure_submission_dir, fit_all, load_b, simulate)

N_DRAWS = 300


def main():
    ensure_submission_dir()
    data = load_b()
    fits = fit_all(data)
    w = bic_weights(fits)
    print("[predict] chi2:", {m: round(float(f["chi2"]), 1) for m, f in fits.items()},
          "weights:", {m: round(float(v), 3) for m, v in w.items()})

    rng = np.random.default_rng(42)
    models = list(fits)
    probs = np.array([w[m] for m in models])
    cond = pd.read_csv(HELDOUT)
    rows = []
    for rid, g in cond.groupby("run_id"):
        T, A0, B0 = (float(g[c].iloc[0]) for c in ("temperature_C", "A0_M", "B0_M"))
        times = np.unique(g["time_min"].to_numpy(float))
        draws = {sp: [] for sp in SPECIES}
        tries = 0
        while len(draws["A"]) < N_DRAWS and tries < 3 * N_DRAWS:
            tries += 1
            m = models[rng.choice(len(models), p=probs)]
            try:
                sim = simulate(m, Conditions(T, A0, B0, times), draw_params(m, fits[m], rng))
            except Exception:
                continue          # a wild parameter draw made the solver fail; skip it
            for sp in SPECIES:
                v = sim[sp].to_numpy()
                rel, ab = NOISE_B[sp]
                noisy = v + np.hypot(rel * v, ab) * rng.standard_normal(len(v))
                draws[sp].append(np.clip(noisy, 0, None))
        for sp in SPECIES:
            d = np.array(draws[sp])
            med = np.median(d, axis=0)
            lo, hi = np.percentile(d, [5, 95], axis=0)
            for i, t in enumerate(times):
                rows.append({"run_id": rid, "time_min": round(float(t), 4), "species": sp,
                             "prediction": f"{med[i]:.6g}", "lower90": f"{lo[i]:.6g}",
                             "upper90": f"{hi[i]:.6g}"})

    df = pd.DataFrame(rows)
    out = os.path.join(SUBMISSION, "predictions.csv")
    df.to_csv(out, index=False)
    print(f"[predict] wrote {len(df)} predictions -> {out}")


if __name__ == "__main__":
    main()
