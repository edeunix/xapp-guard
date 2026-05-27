"""
run_detection_experiment.py
===========================
Runs the detection-accuracy experiment (article Table II / Fig. 4).

For each malicious class A1, A2, A3:
  * build a scenario: 3 benign xApps + 1 malicious variant
  * commission xApp-Guard on a benign-only warm-up window
  * stream 30 min of telemetry (1 s epochs) and run the detector
  * repeat over N independent seeds
  * score the detector against ground truth and aggregate

Metrics (all measured, none hand-written):
  TPR  : among epochs where the malicious xApp is *actually* attacking,
         fraction flagged as WATCH/THROTTLE/QUARANTINE.
  FPR  : among epochs of *benign* xApps, fraction flagged.
  latency: seconds from attack activation to first sustained flag
           (first epoch t* >= attack_on such that the malicious xApp
            is flagged at t* and stays flagged for >= 3 consecutive
            epochs).

Outputs:
  results/detection_results.csv   -- per-class aggregate metrics
  results/trace_<seed>.csv        -- one full per-epoch trace for plots
  results/summary.json            -- machine-readable summary
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.xapp_sim import build_scenario
from xapp_guard.detector import XAppGuard, WATCH, THROTTLE, QUARANTINE

FLAGGED = {WATCH, THROTTLE, QUARANTINE}

# ---- experiment configuration (article Sec. V) -----------------------
EPOCH          = 1.0          # s
RUN_SECONDS    = 1800         # 30 min
ATTACK_ON      = 600.0        # attack active minutes 10..25
ATTACK_OFF     = 1500.0
WARMUP_SECONDS = 600          # commissioning window (benign only)
N_SEEDS        = 5            # independent runs per class
ALPHA          = 0.2
THETA_W        = 0.55
THETA_T        = 0.70
N_ESTIMATORS   = 200


def run_one(seed: int, kind: str) -> pd.DataFrame:
    """One full run for one malicious class. Returns the per-epoch df.

    The run has two contiguous phases drawn from the *same* xApp
    processes and the same RNG stream, exactly as a real deployment is
    first commissioned and then operated:

      * commissioning phase  [0, WARMUP_SECONDS)
            all xApps behave benignly; xApp-Guard trains its Isolation
            Forest and calibrates its thresholds on this telemetry.
      * operational phase    [WARMUP_SECONDS, WARMUP_SECONDS+RUN_SECONDS)
            the malicious xApp activates its attack inside the window
            [WARMUP_SECONDS+ATTACK_ON, WARMUP_SECONDS+ATTACK_OFF);
            xApp-Guard scores every epoch. Only this phase is scored.

    Because both phases come from one stream, the distribution the
    detector is calibrated on matches the distribution it operates on
    -- there is no artificial calibration/operation mismatch.
    """
    rng = np.random.default_rng(seed)

    # attack is shifted by WARMUP_SECONDS so it falls in the
    # operational phase only.
    xapps = build_scenario(rng, kind,
                           attack_on=WARMUP_SECONDS + ATTACK_ON,
                           attack_off=WARMUP_SECONDS + ATTACK_OFF)

    # ---- commissioning phase: collect benign telemetry -------------
    commission_tele = []
    for step in range(WARMUP_SECONDS):
        t = step * EPOCH
        for xa in xapps:
            commission_tele.append(xa.emit(t))

    guard = XAppGuard(alpha=ALPHA, theta_w=THETA_W, theta_t=THETA_T,
                      n_estimators=N_ESTIMATORS, seed=seed)
    guard.commission(commission_tele)

    # ---- operational phase: score every epoch ----------------------
    rows = []
    for step in range(WARMUP_SECONDS, WARMUP_SECONDS + RUN_SECONDS):
        t = step * EPOCH
        tele_epoch = [xa.emit(t) for xa in xapps]
        for out in guard.step_epoch(tele_epoch):
            # re-base time so t=0 is the start of the operational phase
            out["t"] = out["t"] - WARMUP_SECONDS * EPOCH
            out["seed"] = seed
            out["mal_class"] = kind
            rows.append(out)
    return pd.DataFrame(rows)


def score_run(df: pd.DataFrame, kind: str) -> dict:
    """Compute TPR, FPR, detection latency for one run."""
    mal = df[df.kind == kind]
    ben = df[df.kind == "benign"]

    # TPR: malicious xApp during the attack window
    mal_attack = mal[mal.attacking]
    tpr = (mal_attack.trust_state.isin(FLAGGED)).mean()

    # FPR: every benign-xApp epoch
    fpr = (ben.trust_state.isin(FLAGGED)).mean()

    # latency: first sustained flag after ATTACK_ON
    mal_after = mal[mal.t >= ATTACK_ON].sort_values("t").reset_index(drop=True)
    flagged = mal_after.trust_state.isin(FLAGGED).values
    latency = np.nan
    for i in range(len(flagged) - 2):
        if flagged[i] and flagged[i+1] and flagged[i+2]:
            latency = mal_after.t.iloc[i] - ATTACK_ON
            break
    return {"tpr": tpr, "fpr": fpr, "latency": latency}


def main():
    os.makedirs("results", exist_ok=True)
    # optional single-class mode: python run_detection_experiment.py A1
    classes = sys.argv[1:] if len(sys.argv) > 1 else ["A1", "A2", "A3"]
    t_wall0 = time.time()

    for kind in classes:
        per = []
        for seed in range(N_SEEDS):
            df = run_one(seed, kind)
            metrics = score_run(df, kind)
            per.append(metrics)
            print(f"  {kind} seed={seed}: "
                  f"TPR={metrics['tpr']:.3f} "
                  f"FPR={metrics['fpr']:.3f} "
                  f"lat={metrics['latency']:.1f}s")
            if kind == "A2" and seed == 0:
                df.to_csv("results/trace_A2_seed0.csv", index=False)
        # persist this class's raw per-seed metrics immediately
        pd.DataFrame(per).assign(mal_class=kind).to_csv(
            f"results/raw_{kind}.csv", index=False)

    # ---- aggregate from whatever raw_*.csv files exist --------------
    agg_rows = []
    for kind in ["A1", "A2", "A3"]:
        path = f"results/raw_{kind}.csv"
        if not os.path.exists(path):
            continue
        m = pd.read_csv(path)
        agg_rows.append({
            "mal_class": kind,
            "tpr": m.tpr.mean(), "tpr_sd": m.tpr.std(ddof=0),
            "fpr": m.fpr.mean(), "fpr_sd": m.fpr.std(ddof=0),
            "latency_s": m.latency.mean(),
            "latency_sd": m.latency.std(ddof=0)})

    if not agg_rows:
        return
    agg = pd.DataFrame(agg_rows)
    overall = {"tpr": float(agg.tpr.mean()),
               "fpr": float(agg.fpr.mean())}
    agg.to_csv("results/detection_results.csv", index=False)

    summary = {
        "config": {
            "epoch_s": EPOCH, "run_seconds": RUN_SECONDS,
            "attack_on": ATTACK_ON, "attack_off": ATTACK_OFF,
            "warmup_seconds": WARMUP_SECONDS, "n_seeds": N_SEEDS,
            "alpha": ALPHA, "theta_w": THETA_W, "theta_t": THETA_T,
            "n_estimators": N_ESTIMATORS,
        },
        "per_class": agg.to_dict(orient="records"),
        "aggregate": overall,
        "wall_clock_s": round(time.time() - t_wall0, 1),
    }
    with open("results/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Detection results (mean over "
          f"{N_SEEDS} seeds) ===")
    print(agg.to_string(index=False))
    print(f"\nAggregate TPR={overall['tpr']:.3f}  "
          f"FPR={overall['fpr']:.3f}")
    print(f"wall-clock: {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
