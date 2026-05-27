"""
make_figures.py
===============
Generates the two data-driven figures of the article directly from the
measured result CSVs produced by the experiment scripts. No values are
hand-set; every point comes from results/*.csv.

Outputs (PDF, vector, ready for LaTeX includegraphics):
  results/fig_trust_trace.pdf   -- smoothed trust score vs time
                                   (from results/trace_A2_seed0.csv)
  results/fig_scaling.pdf       -- CPU% and O1 KB/s vs #xApps
                                   (from results/scaling_results.csv)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"


def fig_trust_trace():
    df = pd.read_csv(os.path.join(RESULTS, "trace_A2_seed0.csv"))
    # the A2 malicious xApp and one benign xApp for contrast
    mal = df[df.kind == "A2"].sort_values("t")
    ben = df[df.xapp == "benign_sched"].sort_values("t")

    # thresholds: re-derive from the run's governor by reading the
    # smoothed benign distribution -- but the simplest faithful source
    # is the detector itself. We recompute them the same way the
    # detector does: percentiles of benign smoothed scores in the
    # pre-attack window.
    pre = df[df.t < 600]
    s_pre = pre[pre.kind == "benign"].score_smoothed.values
    theta_w = float(np.percentile(s_pre, 99.0))
    theta_t = float(np.percentile(s_pre, 99.9))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(ben.t, ben.score_smoothed, color="#2a8a3e", lw=1.0,
            label="benign xApp")
    ax.plot(mal.t, mal.score_smoothed, color="#c0392b", lw=1.2,
            label="malicious xApp (A2)")
    ax.axhline(theta_w, ls="--", color="#2c6fbb", lw=0.8)
    ax.axhline(theta_t, ls="--", color="#c0392b", lw=0.8)
    ax.text(mal.t.max(), theta_w, r"$\theta_w$", color="#2c6fbb",
            fontsize=7, va="bottom", ha="right")
    ax.text(mal.t.max(), theta_t, r"$\theta_t$", color="#c0392b",
            fontsize=7, va="bottom", ha="right")
    ax.axvline(600, ls=":", color="0.5", lw=0.8)
    ax.text(600, ax.get_ylim()[1], "attack on", fontsize=6.5,
            color="0.4", ha="center", va="bottom")
    ax.set_xlabel("time since start of operation (s)", fontsize=8)
    ax.set_ylabel(r"smoothed trust score $\bar{s}$", fontsize=8)
    ax.set_xlim(0, mal.t.max())
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.9)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    out = os.path.join(RESULTS, "fig_trust_trace.pdf")
    fig.savefig(out)
    plt.close(fig)

    # report the crossing times actually observed
    mal_after = mal[mal.t >= 600]
    cross_w = mal_after[mal_after.score_smoothed >= theta_w].t.min()
    cross_t = mal_after[mal_after.score_smoothed >= theta_t].t.min()
    print(f"  trust trace: theta_w={theta_w:.3f} theta_t={theta_t:.3f} "
          f"cross_w at t={cross_w:.0f}s cross_t at t={cross_t:.0f}s")
    return {"theta_w": theta_w, "theta_t": theta_t,
            "cross_w": float(cross_w), "cross_t": float(cross_t)}


def fig_scaling():
    df = pd.read_csv(os.path.join(RESULTS, "scaling_results.csv"))

    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    ax1.plot(df.n_xapps, df.cpu_pct, "o-", color="#2c6fbb", lw=1.2,
             ms=3.5, label="CPU")
    ax1.set_xlabel("number of xApps monitored", fontsize=8)
    ax1.set_ylabel("CPU (% of one core)", color="#2c6fbb", fontsize=8)
    ax1.set_xticks([0, 40, 80, 120, 160])
    ax1.tick_params(axis="y", labelcolor="#2c6fbb", labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_ylim(0, max(2.0, df.cpu_pct.max() * 1.3))
    ax1.set_xlim(0, df.n_xapps.max() * 1.05)
    ax1.grid(True, lw=0.3, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df.n_xapps, df.o1_kbps, "s--", color="#c0392b", lw=1.2,
             ms=3.0, label="O1 telemetry")
    ax2.set_ylabel("O1 telemetry (KB/s)", color="#c0392b", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#c0392b", labelsize=7)
    ax2.set_ylim(0, df.o1_kbps.max() * 1.2)

    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [l.get_label() for l in lines],
               fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=0.3)
    out = os.path.join(RESULTS, "fig_scaling.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  scaling: CPU {df.cpu_pct.iloc[0]:.2f}-{df.cpu_pct.iloc[-1]:.2f}%"
          f"  O1 {df.o1_kbps.iloc[0]:.1f}-{df.o1_kbps.iloc[-1]:.1f} KB/s")


def fig_roc():
    df = pd.read_csv(os.path.join(RESULTS, "roc_results.csv"))
    colors = {"A1": "#2c6fbb", "A2": "#e08a1e", "A3": "#c0392b"}
    markers = {"A1": "o", "A2": "s", "A3": "^"}

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for kind in ["A1", "A2", "A3"]:
        d = df[df.mal_class == kind].sort_values("fpr")
        ax.plot(d.fpr, d.tpr, marker=markers[kind], color=colors[kind],
                lw=1.2, ms=3.5, label=kind)
        # mark the operating point used in the article (99th pct)
        op = d[d.percentile == 99.0]
        if len(op):
            ax.scatter(op.fpr, op.tpr, s=55, facecolors="none",
                       edgecolors=colors[kind], linewidths=1.4,
                       zorder=5)
    ax.set_xlabel("false positive rate", fontsize=8)
    ax.set_ylabel("true positive rate", fontsize=8)
    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(0.93, 1.005)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="lower right", title="attack class",
              title_fontsize=7, framealpha=0.9)
    ax.text(0.022, 0.945,
            "circles: operating\npoint (99th pct.)",
            fontsize=6, color="0.35")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(RESULTS, "fig_roc.pdf"))
    plt.close(fig)
    aucs = {k: float(np.trapezoid(
                df[df.mal_class == k].sort_values("fpr").tpr,
                df[df.mal_class == k].sort_values("fpr").fpr))
            for k in ["A1", "A2", "A3"]}
    print(f"  roc: partial AUC over swept range "
          f"A1={aucs['A1']:.3f} A2={aucs['A2']:.3f} "
          f"A3={aucs['A3']:.3f}")


def fig_importance():
    df = pd.read_csv(os.path.join(RESULTS, "importance_results.csv"))
    # order features by the three behavioural dimensions
    order = ["kpm_sub_rate", "kpm_ran_funcs", "kpm_volume_kbps",
             "rc_ctrl_rate", "rnti_coverage", "prb_delta_mean",
             "ho_trigger_rate", "pred_entropy", "pred_kl_div"]
    labels = {r["feature"]: r["label"]
              for _, r in df.iterrows()}
    classes = ["A1", "A2", "A3"]
    colors = {"A1": "#2c6fbb", "A2": "#e08a1e", "A3": "#c0392b"}

    piv = (df.pivot(index="feature", columns="mal_class",
                    values="share")
             .reindex(order))
    y = np.arange(len(order))
    h = 0.26

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for i, kind in enumerate(classes):
        ax.barh(y + (1 - i) * h, piv[kind].values, height=h,
                color=colors[kind], label=kind)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[f] for f in order], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("share of detector attention", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, max(0.5, float(np.nanmax(piv.values)) * 1.15))
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="lower right", title="attack class",
              title_fontsize=7, framealpha=0.9)
    # dimension brackets
    for y0, y1, txt in [(-0.4, 2.4, "read"),
                        (2.6, 5.4, "write"),
                        (6.6, 8.4, "inference")]:
        ax.text(ax.get_xlim()[1] * 0.99, (y0 + y1) / 2, txt,
                rotation=90, va="center", ha="right",
                fontsize=6, color="0.45")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(RESULTS, "fig_importance.pdf"))
    plt.close(fig)
    for kind in classes:
        top = piv[kind].idxmax()
        print(f"  importance {kind}: top = {labels[top]} "
              f"({piv[kind].max():.2f})")


def main():
    import matplotlib.ticker  # noqa
    info = fig_trust_trace()
    fig_scaling()
    fig_roc()
    fig_importance()
    print("figures written to results/fig_*.pdf")
    return info


if __name__ == "__main__":
    main()
