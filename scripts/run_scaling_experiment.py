"""
run_scaling_experiment.py
=========================
Measures the resource footprint of xApp-Guard as a function of the
number of xApps it monitors (article Fig. 5).

What is measured -- all real, none hand-written:

  cpu_pct  : the CPU cost of the xApp-Guard scoring loop, expressed as
             a percentage of one core. We time how long the detector
             spends per epoch (feature extraction + Isolation Forest
             scoring + smoothing + governor) and divide by the 1 s
             epoch budget. cpu_pct = 100 * t_detector_per_epoch / 1.0.
             Timed with time.process_time() so only CPU work counts,
             not wall-clock sleeping.

  o1_kbps  : the O1 telemetry volume xApp-Guard must ingest. Each xApp
             emits one 9-feature record per epoch; we serialise the
             records exactly as they would travel on O1 (JSON, the
             encoding FlexRIC's O1 exporter uses) and measure the byte
             count, then divide by the epoch to get KB/s.

The experiment replays synthetic xApp telemetry: the first three
xApps are the benign archetypes, the rest are independent benign
clones with small profile jitter, so the per-xApp cost is realistic
and not dominated by one archetype.

Output: results/scaling_results.csv
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.xapp_sim import XApp, BENIGN_ARCHETYPES
from xapp_guard.detector import XAppGuard

EPOCH          = 1.0
WARMUP_SECONDS = 300       # commissioning window
MEASURE_EPOCHS = 200       # epochs timed for the per-epoch CPU average
N_REPEATS      = 3         # repeats per xApp-count point
XAPP_COUNTS    = [5, 10, 20, 40, 80, 160]
N_ESTIMATORS   = 200


def make_xapps(n: int, rng: np.random.Generator) -> list[XApp]:
    """Build n benign xApps: cycle the three archetypes."""
    archetypes = list(BENIGN_ARCHETYPES.keys())
    xapps = []
    for i in range(n):
        arch = archetypes[i % len(archetypes)]
        xapps.append(XApp(f"xapp_{i:03d}", arch, "benign", rng))
    return xapps


def o1_record_bytes(tele: dict) -> int:
    """Bytes of one xApp's O1 telemetry record, as it travels on O1.

    FlexRIC's O1/VES exporter ships counter records as JSON. We
    serialise exactly the 9 feature fields plus the xApp id and the
    timestamp -- the ground-truth label fields are stripped because
    they never travel on a real O1 link.
    """
    from xapp_guard.detector import FEATURES
    record = {"xapp": tele["xapp"], "t": tele["t"]}
    for f in FEATURES:
        record[f] = round(float(tele[f]), 4)
    return len(json.dumps(record).encode("utf-8"))


def measure_point(n_xapps: int, repeat: int) -> dict:
    """One (n_xapps, repeat) measurement."""
    rng = np.random.default_rng(1000 * repeat + n_xapps)
    xapps = make_xapps(n_xapps, rng)

    # ---- commission on a benign warm-up ---------------------------
    guard = XAppGuard(n_estimators=N_ESTIMATORS, seed=repeat)
    commission = []
    for step in range(WARMUP_SECONDS):
        t = step * EPOCH
        for xa in xapps:
            commission.append(xa.emit(t))
    guard.commission(commission)

    # ---- timed operational epochs ---------------------------------
    cpu_times = []
    o1_bytes_per_epoch = []
    for step in range(MEASURE_EPOCHS):
        t = (WARMUP_SECONDS + step) * EPOCH
        tele_epoch = [xa.emit(t) for xa in xapps]
        # O1 ingest volume for this epoch
        o1_bytes_per_epoch.append(
            sum(o1_record_bytes(tl) for tl in tele_epoch))
        # CPU cost of the detector for this epoch
        c0 = time.process_time()
        guard.step_epoch(tele_epoch)
        cpu_times.append(time.process_time() - c0)

    cpu_per_epoch = float(np.mean(cpu_times))         # seconds
    cpu_pct = 100.0 * cpu_per_epoch / EPOCH           # % of one core
    o1_kbps = float(np.mean(o1_bytes_per_epoch)) / 1024.0 / EPOCH
    return {
        "n_xapps": n_xapps, "repeat": repeat,
        "cpu_per_epoch_ms": cpu_per_epoch * 1e3,
        "cpu_pct": cpu_pct,
        "o1_kbps": o1_kbps,
        "o1_kbps_per_xapp": o1_kbps / n_xapps,
    }


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    t0 = time.time()
    for n in XAPP_COUNTS:
        for r in range(N_REPEATS):
            m = measure_point(n, r)
            rows.append(m)
            print(f"  n={n:3d} rep={r}: "
                  f"CPU={m['cpu_pct']:.2f}%/core  "
                  f"O1={m['o1_kbps']:.1f} KB/s  "
                  f"({m['cpu_per_epoch_ms']:.2f} ms/epoch)")

    df = pd.DataFrame(rows)
    agg = (df.groupby("n_xapps")
             .agg(cpu_pct=("cpu_pct", "mean"),
                  cpu_pct_sd=("cpu_pct", "std"),
                  o1_kbps=("o1_kbps", "mean"),
                  o1_kbps_sd=("o1_kbps", "std"),
                  o1_kbps_per_xapp=("o1_kbps_per_xapp", "mean"))
             .reset_index())
    agg.to_csv("results/scaling_results.csv", index=False)

    summary = {
        "config": {
            "warmup_seconds": WARMUP_SECONDS,
            "measure_epochs": MEASURE_EPOCHS,
            "n_repeats": N_REPEATS,
            "xapp_counts": XAPP_COUNTS,
            "n_estimators": N_ESTIMATORS,
        },
        "per_count": agg.to_dict(orient="records"),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open("results/scaling_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Scaling results (mean over "
          f"{N_REPEATS} repeats) ===")
    print(agg.to_string(index=False))
    print(f"\nwall-clock: {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
