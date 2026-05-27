"""
run_importance_experiment.py
============================
Produces the feature-importance data for the article (new Fig. 5b).

For each attack class we ask: which of the nine O1 features does the
Isolation Forest actually rely on to flag that attack? We answer it
with *permutation importance*, the standard model-agnostic attribution
method and the same principle the article refers to as SHAP-style
attribution.

Procedure, per attack class:
  1. commission xApp-Guard exactly as in the detection experiment;
  2. collect the malicious xApp's feature vectors during the attack
     window (these are the vectors the detector must score as
     anomalous);
  3. measure the baseline mean anomaly score on those vectors;
  4. for each feature j, permute column j across the attack vectors
     (destroying that feature's information while keeping its
     marginal distribution) and re-score;
  5. importance_j = baseline_score - permuted_score. A large drop
     means the detector leans heavily on feature j for this attack.

Importances are averaged over seeds and over several permutations,
then normalised per class so they sum to 1 -- giving the share of the
detector's attention each feature receives.

Every number is measured by re-scoring with the real detector; no
attribution is assumed.

Output: results/importance_results.csv
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.xapp_sim import build_scenario
from xapp_guard.detector import XAppGuard, FeatureExtractor, FEATURES

EPOCH          = 1.0
WARMUP_SECONDS = 600
RUN_SECONDS    = 1800
ATTACK_ON      = 600.0
ATTACK_OFF     = 1500.0
N_SEEDS        = 5
# (ablation importance is deterministic; no permutation count needed)
ALPHA          = 0.2
N_ESTIMATORS   = 200

# human-readable short names for the nine features (article Sec. IV)
FEATURE_LABELS = {
    "kpm_sub_rate":    "KPM sub. rate",
    "kpm_ran_funcs":   "KPM RAN funcs",
    "kpm_volume_kbps": "KPM volume",
    "rc_ctrl_rate":    "RC ctrl rate",
    "rnti_coverage":   "RNTI coverage",
    "prb_delta_mean":  "PRB delta",
    "ho_trigger_rate": "HO trig. rate",
    "pred_entropy":    "pred. entropy",
    "pred_kl_div":     "pred. KL div.",
}


def attack_vectors(seed: int, kind: str):
    """Commission the detector and return (guard, X_ramp, X_benign):
    the trained guard, the malicious xApp's feature matrix during the
    attack ramp, and a matrix of benign feature vectors collected
    during commissioning (the reference 'normal' behaviour).
    """
    rng = np.random.default_rng(seed)
    a_on = WARMUP_SECONDS + ATTACK_ON
    xapps = build_scenario(rng, kind,
                           attack_on=a_on,
                           attack_off=WARMUP_SECONDS + ATTACK_OFF)
    guard = XAppGuard(alpha=ALPHA, n_estimators=N_ESTIMATORS, seed=seed)

    fe = FeatureExtractor()
    mal_name = f"mal_{kind}"

    commission = []
    X_benign = []
    for step in range(WARMUP_SECONDS):
        t = step * EPOCH
        for xa in xapps:
            tele = xa.emit(t)
            commission.append(tele)
            # the malicious xApp's *benign* commissioning behaviour is
            # the right 'normal' reference for that same xApp.
            if tele["xapp"] == mal_name:
                X_benign.append(fe.vector(tele))
    guard.commission(commission, calibrate=False)

    ramp_end = a_on + 120.0
    X_ramp = []
    for step in range(WARMUP_SECONDS, WARMUP_SECONDS + RUN_SECONDS):
        t = step * EPOCH
        for xa in xapps:
            tele = xa.emit(t)
            if (tele["xapp"] == mal_name and tele["attacking"]
                    and t < ramp_end):
                X_ramp.append(fe.vector(tele))
    return guard, np.array(X_ramp), np.array(X_benign)


def ablation_importance(guard, X_attack, X_benign):
    """Benign-ablation importance of each feature.

    For each feature j we replace the attack vectors' column j with
    the benign reference value of that feature (its commissioning
    mean) and re-score. If the anomaly score collapses toward normal,
    feature j is what made the attack detectable. This answers
    'which feature drives detection' correctly -- unlike permuting
    within the attack population, which leaves the anomalous feature
    anomalous.

    importance_j = base_attack_score - score_with_feature_j_normalised
    """
    base = float(guard.scorer.score_batch(X_attack).mean())
    benign_mean = X_benign.mean(axis=0)
    drops = np.zeros(X_attack.shape[1])
    for j in range(X_attack.shape[1]):
        Xa = X_attack.copy()
        Xa[:, j] = benign_mean[j]          # ablate feature j to normal
        ablated = float(guard.scorer.score_batch(Xa).mean())
        drops[j] = base - ablated          # positive => feature matters
    return base, drops


def main():
    os.makedirs("results", exist_ok=True)
    classes = ["A1", "A2", "A3"]
    t0 = time.time()
    rows = []

    for kind in classes:
        per_seed = []
        for seed in range(N_SEEDS):
            guard, X, X_benign = attack_vectors(seed, kind)
            _, drops = ablation_importance(guard, X, X_benign)
            per_seed.append(drops)
        drops = np.mean(per_seed, axis=0)
        drops_sd = np.std(per_seed, axis=0)
        # clip tiny negative noise to 0, then normalise to a share
        drops_pos = np.clip(drops, 0.0, None)
        share = drops_pos / (drops_pos.sum() + 1e-12)
        for j, feat in enumerate(FEATURES):
            rows.append({
                "mal_class": kind,
                "feature": feat,
                "label": FEATURE_LABELS[feat],
                "importance": float(drops[j]),
                "importance_sd": float(drops_sd[j]),
                "share": float(share[j]),
            })
        top = FEATURES[int(np.argmax(drops))]
        print(f"  {kind}: top feature = {FEATURE_LABELS[top]} "
              f"(share {share[np.argmax(drops)]:.2f})")

    df = pd.DataFrame(rows)
    df.to_csv("results/importance_results.csv", index=False)

    summary = {
        "config": {"n_seeds": N_SEEDS,
                   "method": "benign-ablation",
                   "alpha": ALPHA, "n_estimators": N_ESTIMATORS},
        "top_feature": {
            k: df[df.mal_class == k].sort_values(
                "importance", ascending=False).iloc[0]["feature"]
            for k in classes},
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open("results/importance_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Feature importance (share of detector attention) ===")
    piv = df.pivot(index="label", columns="mal_class", values="share")
    print(piv.round(3).to_string())
    print(f"\nwall-clock: {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
