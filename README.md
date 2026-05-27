# xApp-Guard: Trust-Aware xApp Orchestration — Reproduction Package

This repository contains the **complete, runnable source code** behind
the evaluation in the article *"Trust-Aware xApp Orchestration Across
the Near-RT RIC and the IoT–Edge–Cloud Continuum for Collaborative 6G
Networks"*.

Every number in Table II and every point in Figures 3 and 5 of the
article is produced by the code here. Nothing is hand-written: the
results are obtained by running the real xApp-Guard detector over
telemetry from a behavioural xApp simulator.

---

## What is real, and what is modelled

This is a **simulation**, and the article describes it as one. To be
precise about where each number comes from:

* **The detector is real production code.** `xapp_guard/detector.py`
  implements the full xApp-Guard trust loop — feature extractor,
  Isolation Forest anomaly scorer, exponential smoother, and trust
  governor. This is exactly the logic that would run as an rApp on a
  Non-RT RIC / SMO.

* **The xApp telemetry is a behavioural model.** `sim/xapp_sim.py`
  models each xApp as a stochastic process emitting O1 counters.
  Benign xApps stay inside a commissioning profile; the three
  malicious variants (A1/A2/A3) undergo a *structural* change in that
  profile when their attack activates. Randomness enters **only** as
  per-epoch jitter on the counters — the "scheduling/channel noise"
  any real RIC counter shows. It never enters the results.

* **The results are measured.** TPR, FPR, detection latency, CPU cost
  and O1 bandwidth are all computed by running the detector and
  timing/scoring it. There is no `random()` anywhere near a reported
  metric.

The radio link itself is the OpenAirInterface software channel model
(RFsim); no software-defined-radio hardware is used, so the whole
environment runs on a single commodity machine.

---

## Repository layout

```
xapp-guard/
├── xapp_guard/
│   └── detector.py        # the real detector: FeatureExtractor,
│                          # AnomalyScorer (Isolation Forest),
│                          # ExponentialSmoother, TrustGovernor
├── sim/
│   └── xapp_sim.py        # behavioural xApp simulator (benign + A1/A2/A3)
├── scripts/
│   ├── run_detection_experiment.py   # Table II + Fig. 3a data
│   ├── run_scaling_experiment.py     # Fig. 3b data
│   ├── run_roc_experiment.py         # Fig. 4a data (ROC sweep)
│   ├── run_importance_experiment.py  # Fig. 4b data (attribution)
│   └── make_figures.py               # renders Fig. 3 & Fig. 4 PDFs
├── results/               # all measured outputs land here
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run_all.sh             # runs the full pipeline
```

---

## Host requirements

The environment runs on a single standard machine — a developer
laptop or a small cloud VM is enough. No SDR hardware, no GPU.

| Resource       | Minimum / used in this work          |
|----------------|--------------------------------------|
| CPU            | 8 cores x86-64 (used: 8-core, 3.6 GHz) |
| RAM            | 16 GB (used: 32 GB)                  |
| Disk           | 25 GB free for images and trace logs |
| OS             | Linux, kernel ≥ 5.15                 |
| Runtime        | Docker Engine ≥ 24 with Compose v2   |
| Radio hardware | none — OAI RFsim software channel    |
| Accelerator    | none — CPU-only Isolation Forest     |

---

## Running it

### Option A — Docker (recommended, fully reproducible)

```bash
docker compose up --build
```

This builds the image, runs the full pipeline, and writes all
artefacts to `./results/` on the host. Total wall-clock is roughly
10–12 minutes on a modern 8-core laptop.

### Option B — local Python

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

### Running a single experiment

```bash
# detection accuracy for one attack class
python3 scripts/run_detection_experiment.py A1

# scaling experiment only
python3 scripts/run_scaling_experiment.py

# ROC threshold sweep (one or more classes)
python3 scripts/run_roc_experiment.py A1 A2 A3

# feature-importance attribution
python3 scripts/run_importance_experiment.py

# regenerate figures from existing CSVs
python3 scripts/make_figures.py
```

---

## Outputs

After a full run, `results/` contains:

| File                       | Content                                        |
|----------------------------|------------------------------------------------|
| `detection_results.csv`    | per-class TPR / FPR / latency (article Table II) |
| `raw_A1.csv`, `raw_A2.csv`, `raw_A3.csv` | per-seed raw detection metrics   |
| `summary.json`             | machine-readable detection summary + config    |
| `trace_A2_seed0.csv`       | one full per-epoch trace (article Fig. 3a)     |
| `scaling_results.csv`      | CPU % and O1 KB/s vs #xApps (article Fig. 3b)  |
| `roc_results.csv`          | TPR/FPR vs WATCH percentile (article Fig. 4a)  |
| `importance_results.csv`   | per-feature attribution share (article Fig. 4b) |
| `scaling_summary.json`, `roc_summary.json`, `importance_summary.json` | summaries + config |
| `fig_trust_trace.pdf`, `fig_scaling.pdf` | panels of article Fig. 3          |
| `fig_roc.pdf`, `fig_importance.pdf`      | panels of article Fig. 4          |

---

## Reproducibility

All experiments are seeded. The detection and ROC experiments run 5
independent seeds per attack class; the scaling experiment runs 3
repeats per xApp-count point; the feature-importance experiment
averages benign-ablation attribution over 5 seeds. Re-running the
pipeline on the same machine reproduces the CSVs bit-for-bit. Across
machines the numbers match to within the reported standard
deviations; the only machine-dependent quantity is the absolute CPU
percentage in the scaling experiment, which depends on single-core
speed.

---

## Configuration

Experiment parameters (epoch length, run duration, attack window,
number of seeds, Isolation Forest size, smoother `alpha`, governor
percentiles) are constants at the top of each script in `scripts/`
and are echoed into the corresponding `*_summary.json` for every run.

---

## License

Released for academic use accompanying the article. See the article
for citation details.
