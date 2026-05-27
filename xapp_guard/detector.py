"""
detector.py
===========
The xApp-Guard runtime trust loop. This is the *real* detector code:
the same logic would run as an rApp on the Non-RT RIC / SMO.

Pipeline (per the article, Sec. IV):
  1. FeatureExtractor   -- assembles the 9-D behavioural vector from
                           O1 telemetry, one vector per xApp per epoch.
  2. AnomalyScorer      -- unsupervised Isolation Forest, trained on a
                           window of known-benign vectors, returns a
                           normalised anomaly score s in [0,1].
  3. ExponentialSmoother-- s_bar_t = alpha*s_t + (1-alpha)*s_bar_{t-1}.
  4. TrustGovernor      -- maps s_bar into {HEALTHY, WATCH, THROTTLE,
                           QUARANTINE} via thresholds theta_w, theta_t.

No randomness is used anywhere in this file. Given the same telemetry
stream the detector is fully deterministic, except for the Isolation
Forest's training, whose RNG seed is fixed and reported.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest

FEATURES = [
    "kpm_sub_rate", "kpm_ran_funcs", "kpm_volume_kbps",
    "rc_ctrl_rate", "rnti_coverage", "prb_delta_mean",
    "ho_trigger_rate", "pred_entropy", "pred_kl_div",
]

# Trust states
HEALTHY, WATCH, THROTTLE, QUARANTINE = "HEALTHY", "WATCH", "THROTTLE", "QUARANTINE"


class FeatureExtractor:
    """Turns a telemetry dict into the ordered 9-D feature vector."""
    @staticmethod
    def vector(tele: dict) -> np.ndarray:
        return np.array([tele[f] for f in FEATURES], dtype=float)


class AnomalyScorer:
    """
    Unsupervised Isolation Forest. Trained once on commissioning data,
    then queried per epoch. The raw sklearn score_samples() is mapped
    to [0,1] where 1 == most anomalous, using the training-set score
    range so the mapping is stable and reproducible.
    """
    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.iforest = IsolationForest(
            n_estimators=n_estimators, contamination="auto",
            random_state=seed)
        self._lo = None
        self._hi = None
        self._mu = None
        self._sd = None

    def fit(self, X_benign: np.ndarray) -> None:
        # standardise features (Isolation Forest is scale-robust but
        # standardising makes the [0,1] mapping interpretable)
        self._mu = X_benign.mean(axis=0)
        self._sd = X_benign.std(axis=0) + 1e-9
        Xs = (X_benign - self._mu) / self._sd
        self.iforest.fit(Xs)
        raw = self.iforest.score_samples(Xs)   # higher = more normal
        # anomaly score a = -raw  (higher = more anomalous)
        a = -raw
        self._lo, self._hi = float(a.min()), float(a.max())

    def score(self, x: np.ndarray) -> float:
        return float(self.score_batch(x.reshape(1, -1))[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised scoring of several feature vectors at once.

        Semantically identical to calling score() on each row: the
        Isolation Forest is stateless at inference time, so scoring a
        batch produces exactly the same per-row scores as scoring rows
        one by one. Batching only removes Python-call overhead.
        """
        Xs = (X - self._mu) / self._sd
        a = -self.iforest.score_samples(Xs)
        s = (a - self._lo) / (self._hi - self._lo + 1e-12)
        return np.clip(s, 0.0, 1.0)


class ExponentialSmoother:
    """s_bar_t = alpha*s_t + (1-alpha)*s_bar_{t-1}, per xApp."""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def update(self, xapp: str, s: float) -> float:
        prev = self._state.get(xapp, s)
        cur = self.alpha * s + (1 - self.alpha) * prev
        self._state[xapp] = cur
        return cur


@dataclass
class TrustGovernor:
    """Maps a smoothed score to a discrete trust state + A1 policy."""
    theta_w: float = 0.55     # WATCH threshold
    theta_t: float = 0.70     # THROTTLE threshold
    quarantine_margin: float = 0.15

    def state(self, s_bar: float) -> str:
        if s_bar < self.theta_w:
            return HEALTHY
        if s_bar < self.theta_t:
            return WATCH
        if s_bar < self.theta_t + self.quarantine_margin:
            return THROTTLE
        return QUARANTINE


class XAppGuard:
    """End-to-end loop tying the four components together."""
    def __init__(self, alpha: float = 0.2, theta_w: float = 0.55,
                 theta_t: float = 0.70, n_estimators: int = 200,
                 seed: int = 0):
        self.scorer = AnomalyScorer(n_estimators=n_estimators, seed=seed)
        self.smoother = ExponentialSmoother(alpha=alpha)
        self.governor = TrustGovernor(theta_w=theta_w, theta_t=theta_t)
        self.extractor = FeatureExtractor()

    def commission(self, benign_telemetry: list[dict],
                   calibrate: bool = True,
                   watch_pct: float = 99.0,
                   throttle_pct: float = 99.9) -> None:
        """Train the Isolation Forest on known-benign telemetry.

        If `calibrate` is set, the WATCH/THROTTLE thresholds are derived
        from the distribution of *smoothed* benign scores observed on
        the commissioning data: theta_w is the `watch_pct` percentile
        and theta_t the `throttle_pct` percentile. This is what an
        operator does at commissioning -- it fixes the benign false
        positive rate by construction rather than guessing thresholds.
        """
        X = np.vstack([self.extractor.vector(t) for t in benign_telemetry])
        self.scorer.fit(X)
        if not calibrate:
            return
        # replay the commissioning telemetry through the *smoother* to
        # get the distribution of smoothed scores a benign xApp yields,
        # then set thresholds at the requested percentiles.
        cal_smoother = ExponentialSmoother(alpha=self.smoother.alpha)
        s_raw = self.scorer.score_batch(X)
        s_bar = []
        for tele, s in zip(benign_telemetry, s_raw):
            s_bar.append(cal_smoother.update(tele["xapp"], float(s)))
        s_bar = np.array(s_bar)
        self.governor.theta_w = float(np.percentile(s_bar, watch_pct))
        self.governor.theta_t = float(np.percentile(s_bar, throttle_pct))

    def step(self, tele: dict) -> dict:
        """Process one telemetry sample, return detector output."""
        return self.step_epoch([tele])[0]

    def step_epoch(self, tele_list: list[dict]) -> list[dict]:
        """Process all xApps' telemetry for a single epoch at once.

        This is causal and online: it uses only telemetry from the
        current epoch. Scoring is batched purely for speed; the result
        for each xApp is identical to calling step() on it individually
        because the Isolation Forest is stateless at inference and the
        per-xApp smoother is updated in a deterministic order.
        """
        X = np.vstack([self.extractor.vector(t) for t in tele_list])
        scores = self.scorer.score_batch(X)
        out = []
        for tele, s in zip(tele_list, scores):
            s = float(s)
            s_bar = self.smoother.update(tele["xapp"], s)
            state = self.governor.state(s_bar)
            out.append({
                "t": tele["t"], "xapp": tele["xapp"],
                "score": s, "score_smoothed": s_bar,
                "trust_state": state,
                "kind": tele["kind"], "attacking": tele["attacking"],
            })
        return out
