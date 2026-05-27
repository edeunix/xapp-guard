"""
run_roc_experiment.py
=====================
Produces the ROC data for the article (new Fig. 5a).

The trust governor flags an xApp when its smoothed score crosses the
WATCH threshold theta_w. theta_w is calibrated at commissioning as a
percentile of the benign smoothed-score distribution. Sweeping that
percentile from low to high traces the operating curve of the
detector: a low percentile gives an aggressive detector (high TPR,
high FPR), a high percentile gives a conservative one.

For each attack class and each percentile we:
  * run the full commission + operation pipeline (same as the
    detection experiment),
  * recompute the trust state under the swept threshold,
  * measure TPR (malicious epochs during the attack) and FPR
    (all benign-xApp epochs).

Every point is measured -- no curve is fitted or hand-drawn.

Output: results/roc_results.csv
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.xapp_sim import build_scenario
from xapp_guard.detector import (XAppGuard, ExponentialSmoother,
                                 FeatureExtractor)

EPOCH          = 1.0
WARMUP_SECONDS = 600
RUN_SECONDS    = 1800
ATTACK_ON      = 600.0
ATTACK_OFF     = 1500.0
N_SEEDS        = 5
ALPHA          = 0.2
N_ESTIMATORS   = 200

# WATCH percentiles to sweep. Lower percentile -> lower threshold ->
# more aggressive detector.
PERCENTILES = [80, 85, 90, 93, 95, 97, 98, 99, 99.5, 99.9]


def run_scored_stream(seed: int, kind: str):
    """Run one commission+operation and return, per operational epoch,
    the smoothed score, the xApp kind, and the attacking flag.

    The detector is committed once; we then *reuse* the smoothed
    scores and apply different thresholds in post-processing. This is
    legitimate because theta_w only affects the governor's labelling,
    not the score itself -- so one run yields the whole ROC curve.
    """
    rng = np.random.default_rng(seed)
    xapps = build_scenario(rng, kind,
                           attack_on=WARMUP_SECONDS + ATTACK_ON,
                           attack_off=WARMUP_SECONDS + ATTACK_OFF)

    guard = XAppGuard(alpha=ALPHA, n_estimators=N_ESTIMATORS, seed=seed)

    # commissioning phase
    commission = []
    for step in range(WARMUP_SECONDS):
        t = step * EPOCH
        for xa in xapps:
            commission.append(xa.emit(t))
    # train the forest but skip threshold calibration -- we sweep it
    guard.commission(commission, calibrate=False)

    # also score the commissioning telemetry so we can derive
    # percentile thresholds from the benign smoothed-score
    # distribution, exactly as the real governor does.
    fe = FeatureExtractor()
    Xc = np.vstack([fe.vector(t) for t in commission])
    sc = guard.scorer.score_batch(Xc)
    cal_sm = ExponentialSmoother(alpha=ALPHA)
    cal_bar = np.array([cal_sm.update(t["xapp"], float(s))
                        for t, s in zip(commission, sc)])

    # operational phase
    rows = []
    for step in range(WARMUP_SECONDS, WARMUP_SECONDS + RUN_SECONDS):
        t = step * EPOCH
        for out in guard.step_epoch([xa.emit(t) for xa in xapps]):
            rows.append((out["score_smoothed"], out["kind"],
                         out["attacking"]))
    op = pd.DataFrame(rows, columns=["s_bar", "kind", "attacking"])
    return op, cal_bar


def main():
    os.makedirs("results", exist_ok=True)
    classes = sys.argv[1:] if len(sys.argv) > 1 else ["A1", "A2", "A3"]
    t0 = time.time()

    for kind in classes:
        rows = []
        per_pct = {p: {"tpr": [], "fpr": []} for p in PERCENTILES}
        for seed in range(N_SEEDS):
            op, cal_bar = run_scored_stream(seed, kind)
            mal = op[op.kind == kind]
            ben = op[op.kind == "benign"]
            mal_attack = mal[mal.attacking]
            for p in PERCENTILES:
                theta_w = float(np.percentile(cal_bar, p))
                tpr = float((mal_attack.s_bar >= theta_w).mean())
                fpr = float((ben.s_bar >= theta_w).mean())
                per_pct[p]["tpr"].append(tpr)
                per_pct[p]["fpr"].append(fpr)
        for p in PERCENTILES:
            rows.append({
                "mal_class": kind, "percentile": p,
                "tpr": np.mean(per_pct[p]["tpr"]),
                "tpr_sd": np.std(per_pct[p]["tpr"]),
                "fpr": np.mean(per_pct[p]["fpr"]),
                "fpr_sd": np.std(per_pct[p]["fpr"]),
            })
        pd.DataFrame(rows).to_csv(f"results/roc_raw_{kind}.csv",
                                  index=False)
        print(f"  {kind}: swept {len(PERCENTILES)} thresholds "
              f"x {N_SEEDS} seeds")

    # ---- assemble from whatever roc_raw_*.csv files exist ----------
    parts = []
    for kind in ["A1", "A2", "A3"]:
        path = f"results/roc_raw_{kind}.csv"
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df.to_csv("results/roc_results.csv", index=False)

    auc = {}
    for kind in df.mal_class.unique():
        d = df[df.mal_class == kind].sort_values("fpr")
        auc[kind] = float(np.trapezoid(d.tpr, d.fpr))
    summary = {
        "config": {"percentiles": PERCENTILES, "n_seeds": N_SEEDS,
                   "alpha": ALPHA, "n_estimators": N_ESTIMATORS},
        "auc_partial": auc,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open("results/roc_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== ROC sweep done ===")
    print(df.to_string(index=False))
    print(f"\nwall-clock: {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
