"""just replicate -> submission/figures/*.png (every figure in the presentation)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from model_fit import (CANDIDATES, HELDOUT, SUBMISSION, Conditions, bic_weights,
                       ensure_submission_dir, fit_all, load_b, simulate, to_params)

FIG = os.path.join(SUBMISSION, "figures")


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[replicate] wrote", path)


def fig_gap(b):
    """Model-free evidence: B consumed minus (P + Q). Q below LOQ counted as 0."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rid, g in b.groupby("run_id"):
        t = g.pivot_table(index="time_min", columns="species", values="value")
        gap = (g["B0_M"].iloc[0] - t["B"]) - (t["P"] + t["Q"].fillna(0))
        ax.plot(gap.index, gap.values, "o-", ms=3, label=f"{rid} ({g['temperature_C'].iloc[0]:.0f} C)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("B consumed - (P + Q)  [mol/L]")
    ax.set_title("Material missing from B, P and Q: a hidden intermediate")
    ax.legend(fontsize=8)
    save(fig, "fig1_gap.png")


def fig_fits(b, fits):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    p2, p4 = to_params("M2", fits["M2"]["x"]), to_params("M4", fits["M4"]["x"])
    for j, (rid, g) in enumerate(b.groupby("run_id")):
        T, A0, B0 = g["temperature_C"].iloc[0], g["A0_M"].iloc[0], g["B0_M"].iloc[0]
        tf = np.linspace(0, g["time_min"].max(), 300)
        s2 = simulate("M2", Conditions(T, A0, B0, tf), p2)
        s4 = simulate("M4", Conditions(T, A0, B0, tf), p4)
        for sp, col in [("B", "C1"), ("P", "C2")]:
            gg = g[g["species"] == sp]
            axes[0, j].plot(gg["time_min"], gg["value"], "o", color=col, ms=4)
            axes[0, j].plot(tf, s2[sp], "-", color=col)
            axes[0, j].plot(tf, s4[sp], "--", color="k", lw=0.8)
        axes[0, j].plot(tf, s2["I"], ":", color="C4")
        axes[0, j].set_title(f"{rid}: {T:.0f} C")
        gq = g[(g["species"] == "Q") & (~g["cens"])]
        axes[1, j].plot(gq["time_min"], gq["value"], "o", color="C3")
        axes[1, j].plot(tf, s2["Q"], "-", color="C3")
        axes[1, j].plot(tf, s4["Q"], "--", color="k", lw=0.8)
        axes[1, j].axhline(3e-5, color="gray", ls=":")
        axes[1, j].set_xlabel("time (min)")
    axes[0, 0].set_ylabel("mol/L: B orange, P green, hidden I purple\n(solid = M2, dashed = M4)")
    axes[1, 0].set_ylabel("Q (mol/L); dotted = LOQ")
    save(fig, "fig2_fits.png")


def fig_bookkeeping(b):
    """Which models can hold B-derived material back? Illustrative parameters: the zeros hold for any."""
    p = {"k1": 0.04, "Ea1": 58, "k2": 0.006, "Ea2": 52, "k3": 7e-4, "Ea3": 66}
    cnd = Conditions(75.0, 1.0, 0.25, np.linspace(0, 720, 200))
    models = list(CANDIDATES)
    col1, col2 = [], []
    for m in models:
        s = simulate(m, cnd, p)
        used = 0.25 - s["B"]
        col1.append(max(float((used - (s["P"] + s["Q"])).abs().max()), 1e-17))
        col2.append(max(float((used - s["P"]).abs().max()), 1e-17))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    ax.semilogy(x - 0.08, col1, "o", ms=9, label="max |B used - (P+Q)|")
    ax.semilogy(x + 0.08, col2, "s", ms=9, label="max |B used - P|")
    ax.axhline(0.10, color="r", ls="--", label="gap seen in the data (~0.10)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("largest mismatch (mol/L); 1e-16 = zero")
    ax.set_title("Only M2 and M4 can leave B-derived material unaccounted for")
    ax.legend(fontsize=8)
    save(fig, "fig3_bookkeeping.png")


def fig_predictions():
    pred_path = os.path.join(SUBMISSION, "predictions.csv")
    if not os.path.exists(pred_path):
        print("[replicate] predictions.csv missing (run `just predict` first); skipping fig4")
        return
    p = pd.read_csv(pred_path)
    q = p[p["species"] == "Q"]
    last = q[q["time_min"] == q.groupby("run_id")["time_min"].transform("max")].sort_values("run_id")
    h = pd.read_csv(HELDOUT).drop_duplicates("run_id").set_index("run_id")
    labels = [f"{r}\nB:A={h.loc[r, 'B0_M'] / h.loc[r, 'A0_M']:.2f}" for r in last["run_id"]]
    y = last["prediction"].to_numpy(float)
    lo, hi = last["lower90"].to_numpy(float), last["upper90"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.errorbar(range(len(y)), y, yerr=[y - lo, hi - y], fmt="o", capsize=4)
    ax.axhline(3e-5, color="gray", ls=":", label="LOQ")
    ax.set_yscale("log")
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Q at end of run (mol/L), 90% interval")
    ax.set_title("Held-out Q predictions: wide where M2 and M4 disagree (A depleted)")
    ax.legend()
    save(fig, "fig4_predictions.png")


def fig_campaign(fits):
    camp_path = os.path.join(SUBMISSION, "campaign.yaml")
    if not os.path.exists(camp_path):
        print("[replicate] campaign.yaml missing (run `just suggest` first); skipping fig5")
        return
    exps = yaml.safe_load(open(camp_path))["experiments"]
    p2, p4 = to_params("M2", fits["M2"]["x"]), to_params("M4", fits["M4"]["x"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.ravel()
    grid = np.geomspace(2, 960, 300)
    for ax, e in zip(axes, exps):
        cnd = Conditions(e["temperature_C"], e["A0_M"], e["B0_M"], grid)
        q2 = simulate("M2", cnd, p2)["Q"].to_numpy()
        q4 = simulate("M4", cnd, p4)["Q"].to_numpy()
        ts = np.array(e["sample_times_min"])
        c = Conditions(e["temperature_C"], e["A0_M"], e["B0_M"], ts)
        ax.plot(grid, q2, label="M2")
        ax.plot(grid, q4, "--", label="M4")
        ax.plot(ts, simulate("M2", c, p2)["Q"].to_numpy(), "k|", ms=14, label="samples")
        ax.axhline(3e-5, color="gray", ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2e-3)
        ax.set_title(f"{e['run_id']}: {e['temperature_C']:.0f} C, A={e['A0_M']}, B={e['B0_M']}", fontsize=9)
        ax.set_xlabel("time (min)")
    for ax in axes[len(exps):]:
        ax.axis("off")
    axes[0].set_ylabel("Q (mol/L)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Proposed campaign: where M2 and M4 predict different Q, and when we sample")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig5_campaign.png")


def main():
    ensure_submission_dir()
    b = load_b()
    fits = fit_all(b)
    w = bic_weights(fits)
    print("[replicate] M2/M4 chi2:", {m: round(float(f["chi2"]), 1) for m, f in fits.items()},
          "BIC weights:", {m: round(float(v), 3) for m, v in w.items()})
    fig_gap(b)
    fig_fits(b, fits)
    fig_bookkeeping(b)
    fig_predictions()
    fig_campaign(fits)


if __name__ == "__main__":
    main()
