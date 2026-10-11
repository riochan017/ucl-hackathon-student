"""just suggest -> submission/campaign.yaml: 7 runs chosen to separate M2 from M4."""
import os

import numpy as np
import yaml

from model_fit import (Conditions, SUBMISSION, ensure_submission_dir, fit_all, load_b,
                       simulate, to_params)

# (run_id, T in C, A0, B0): B in excess of A so that A is depleted, which stops M4's
# Q formation (I + A -> Q) but not M2's (I -> Q).
DESIGN = [
    ("E1", 120, 0.50, 1.5), ("E2", 120, 0.25, 1.0),
    ("E3",  85, 0.50, 1.0), ("E4",  85, 0.25, 1.5),
    ("E5",  50, 0.50, 1.5), ("E6",  50, 0.25, 1.0),
    ("E7", 100, 0.50, 0.5),
]
N_SAMPLES = 6


def main():
    ensure_submission_dir()
    fits = fit_all(load_b())
    p2 = to_params("M2", fits["M2"]["x"])
    grid = np.linspace(0, 960, 961)

    experiments = []
    for rid, T, A0, B0 in DESIGN:
        q = simulate("M2", Conditions(T, A0, B0, grid), p2)["Q"].to_numpy()
        tau = grid[np.argmax(q >= 0.9 * q[-1])]          # time for Q to reach 90% of its final value
        t_first = max(0.3 * tau, 5.0)
        t_last = min(960.0, max(8.0 * tau, 60.0))
        times = np.unique(np.round(np.geomspace(t_first, t_last, N_SAMPLES)))
        experiments.append({"run_id": rid, "temperature_C": float(T), "A0_M": float(A0),
                            "B0_M": float(B0), "sample_times_min": [float(t) for t in times],
                            "assay": ["A", "B", "P", "Q"]})

    # Check the budget with the same cost model the scorer uses.
    total_g = sum(e["A0_M"] * 5 / 1000 * 250 for e in experiments)
    n_samples = sum(len(e["sample_times_min"]) for e in experiments)
    assert 3 <= len(experiments) <= 10 and total_g <= 5.0 and n_samples <= 60, "budget exceeded"

    campaign = {
        "rationale": ("M2 and M4 fit the released data equally well (chi2 155.0 vs 155.6) and differ only "
                      "in whether Q's formation consumes A. All released runs have A in large excess, so "
                      "A acts as a constant. Runs here put B in excess of A (B:A of 1 to 6, A0 of 0.25-0.5 M) "
                      "so that A is depleted, stopping M4's Q formation but not M2's; the models then differ "
                      "by about 5 noise-widths in Q. Three temperatures (50, 85, 120 C) constrain Ea3, the "
                      "weakest parameter, and sample times scale with each run's timescale. "
                      f"Total {total_g:.2f} g A, {len(experiments)} runs, {n_samples} samples."),
        "experiments": experiments,
    }
    out = os.path.join(SUBMISSION, "campaign.yaml")
    with open(out, "w") as fh:
        yaml.safe_dump(campaign, fh, sort_keys=False, width=100)
    print(f"[suggest] wrote {len(experiments)} experiments ({total_g:.4f} g A, {n_samples} samples) -> {out}")


if __name__ == "__main__":
    main()
