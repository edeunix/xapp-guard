"""
xapp_sim.py
===========
Behavioural simulator for O-RAN xApps running on a Near-RT RIC.

This module produces a stream of *O1 telemetry counters* for each xApp,
epoch by epoch (1 epoch = 1 s). It does NOT produce detection results --
those are computed by the xApp_guard package by running the real
detector on the telemetry generated here.

Design philosophy
-----------------
Each xApp is modelled as a stochastic process whose *expected* counter
values are governed by a behavioural profile. Benign xApps stay inside
their commissioning profile; malicious xApps undergo a structural change
in that profile when their attack is activated. The per-epoch jitter
around the expected value is Gaussian/Poisson noise -- this is the
"channel/scheduling noise" any real RIC counter exhibits, and it is the
only place randomness enters. The detector never sees the ground-truth
label; it only sees counters.

The three malicious classes match the threat taxonomy A1/A2/A3:

  A1  KPM exfiltration       -> read-dimension counters inflate
  A2  slice-targeted QoS deg -> write-dimension counters skew
  A3  intra-xApp ML poisoning-> inference-dimension distribution drifts

Counters emitted per epoch (the 9 features used by the detector):
  f1 kpm_sub_rate        E2SM-KPM subscription requests / s
  f2 kpm_ran_funcs       distinct RAN functions subscribed
  f3 kpm_volume_kbps     aggregate KPM data volume   (KB/s)
  f4 rc_ctrl_rate        E2SM-RC control messages / s
  f5 rnti_coverage       fraction of active UEs targeted (0..1)
  f6 prb_delta_mean      mean |delta PRB| per control msg
  f7 ho_trigger_rate     handover triggers per UE / s
  f8 pred_entropy        Shannon entropy of prediction outputs (bits)
  f9 pred_kl_div         KL divergence of prediction histogram vs baseline
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

FEATURES = [
    "kpm_sub_rate", "kpm_ran_funcs", "kpm_volume_kbps",
    "rc_ctrl_rate", "rnti_coverage", "prb_delta_mean",
    "ho_trigger_rate", "pred_entropy", "pred_kl_div",
]

# Number of handover-target cells used to build the prediction histogram
N_HO_TARGETS = 8


@dataclass
class BenignProfile:
    """Commissioning-time behavioural profile of a benign xApp."""
    kpm_sub_rate: float        # subscriptions / s
    kpm_ran_funcs: float       # distinct RAN functions
    kpm_volume_kbps: float     # KB/s
    rc_ctrl_rate: float        # control msgs / s
    rnti_coverage: float       # 0..1
    prb_delta_mean: float      # PRBs
    ho_trigger_rate: float     # triggers / UE / s
    # prediction histogram: probability mass over N_HO_TARGETS cells.
    # benign xApps predict roughly uniformly over their serving cells.
    pred_hist: np.ndarray = field(
        default_factory=lambda: np.ones(N_HO_TARGETS) / N_HO_TARGETS)


# Three benign xApp archetypes used in the experiments.
# Numbers are chosen to be representative of FlexRIC/OAI E2 counter
# magnitudes; what matters scientifically is that benign != malicious
# in *structure*, and that the detector is never told which is which.
BENIGN_ARCHETYPES = {
    "handover_pred": BenignProfile(
        kpm_sub_rate=2.0, kpm_ran_funcs=2.0, kpm_volume_kbps=14.0,
        rc_ctrl_rate=1.2, rnti_coverage=0.35, prb_delta_mean=3.0,
        ho_trigger_rate=0.08),
    "slice_sched": BenignProfile(
        kpm_sub_rate=3.0, kpm_ran_funcs=3.0, kpm_volume_kbps=22.0,
        rc_ctrl_rate=4.5, rnti_coverage=0.55, prb_delta_mean=6.0,
        ho_trigger_rate=0.02),
    "kpm_reporter": BenignProfile(
        kpm_sub_rate=4.0, kpm_ran_funcs=3.0, kpm_volume_kbps=30.0,
        rc_ctrl_rate=0.1, rnti_coverage=0.10, prb_delta_mean=1.0,
        ho_trigger_rate=0.0),
}


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0); p = p / p.sum()
    q = np.clip(q, 1e-12, 1.0); q = q / q.sum()
    return float((p * np.log2(p / q)).sum())


class XApp:
    """
    A simulated xApp. `kind` is one of:
        'benign'  -- stays inside its profile for the whole run
        'A1','A2','A3' -- malicious; attack active in [attack_on, attack_off]
    The ground-truth label is stored only for *scoring* the detector
    afterwards; it is never exposed in the emitted telemetry.
    """

    def __init__(self, name: str, archetype: str, kind: str,
                 rng: np.random.Generator,
                 attack_on: float = 600.0, attack_off: float = 1500.0):
        self.name = name
        self.kind = kind
        self.rng = rng
        self.attack_on = attack_on
        self.attack_off = attack_off
        self.base = BENIGN_ARCHETYPES[archetype]
        # commissioning baseline: the benign prediction histogram,
        # frozen at construction time. f9 (KL divergence) is always
        # measured against this fixed reference.
        self.commission_baseline = self.base.pred_hist.copy()

    # ----- helpers -------------------------------------------------
    def _is_attacking(self, t: float) -> bool:
        return (self.kind != "benign"
                and self.attack_on <= t < self.attack_off)

    def _noisy(self, mu: float, rel_sigma: float = 0.12,
               nonneg: bool = True) -> float:
        """Gaussian multiplicative jitter -- the only randomness."""
        v = mu * (1.0 + self.rng.normal(0.0, rel_sigma))
        return max(0.0, v) if nonneg else v

    def _pois(self, mu: float) -> float:
        return float(self.rng.poisson(max(mu, 1e-6)))

    # ----- main telemetry step ------------------------------------
    def emit(self, t: float) -> dict:
        """Return the 9 O1 counters for this xApp at epoch time t (s)."""
        b = self.base
        attacking = self._is_attacking(t)

        # ----- READ dimension (A1 inflates these) -----------------
        m_sub  = b.kpm_sub_rate
        m_func = b.kpm_ran_funcs
        m_vol  = b.kpm_volume_kbps
        if attacking and self.kind == "A1":
            # exfiltration: broaden subscription scope + volume.
            # ramp up over ~30 s so it is not a step the detector
            # could catch trivially on a single epoch.
            ramp = min(1.0, (t - self.attack_on) / 30.0)
            m_sub  *= (1.0 + 3.0 * ramp)
            m_func *= (1.0 + 1.5 * ramp)
            m_vol  *= (1.0 + 5.0 * ramp)

        # ----- WRITE dimension (A2 skews these) -------------------
        m_ctrl = b.rc_ctrl_rate
        m_cov  = b.rnti_coverage
        m_prb  = b.prb_delta_mean
        if attacking and self.kind == "A2":
            # slice-targeted QoS degradation: more control msgs, but
            # each one still individually small; coverage skews toward
            # the victim slice's UEs; PRB deltas grow modestly.
            ramp = min(1.0, (t - self.attack_on) / 40.0)
            m_ctrl *= (1.0 + 2.2 * ramp)
            m_cov   = min(0.98, m_cov * (1.0 + 0.9 * ramp))
            m_prb  *= (1.0 + 1.4 * ramp)

        # ----- INFERENCE dimension (A3 drifts the histogram) ------
        if attacking and self.kind == "A3":
            # ML poisoning: prediction mass slowly concentrates on a
            # target cell. The per-epoch change is small; the attack
            # is visible only in the aggregate distribution. We model
            # a moderate-intensity poisoning campaign (final bias 0.70
            # of the prediction mass onto the attacker-favoured cell),
            # ramped over 90 s so no single epoch is a step change.
            ramp = min(1.0, (t - self.attack_on) / 90.0)
            hist = self.base.pred_hist.copy()
            target = 0  # attacker-favoured cell
            bias = 0.70 * ramp
            hist = (1 - bias) * hist
            hist[target] += bias
            hist = hist / hist.sum()
        else:
            # benign / non-A3: small dirichlet jitter around profile
            hist = self.rng.dirichlet(self.base.pred_hist * 200.0)

        m_ho = b.ho_trigger_rate

        # measured prediction-distribution statistics.
        # f8 (entropy) is intrinsic to the current histogram.
        # f9 (KL divergence) is measured against the *commissioning
        #     baseline* -- the benign prediction histogram frozen at
        #     commissioning time. A self-adapting baseline would chase
        #     a slow poisoning campaign and go blind to it; freezing
        #     the reference is what makes A3 observable at all.
        pred_entropy = _entropy(hist)
        pred_kl      = _kl(hist, self.commission_baseline)

        return {
            "t":               t,
            "xapp":            self.name,
            "kind":            self.kind,            # ground truth, for scoring only
            "attacking":       bool(attacking),       # ground truth, for scoring only
            "kpm_sub_rate":    self._noisy(m_sub),
            "kpm_ran_funcs":   max(1.0, self._noisy(m_func, 0.05)),
            "kpm_volume_kbps": self._noisy(m_vol),
            "rc_ctrl_rate":    self._pois(m_ctrl),
            "rnti_coverage":   float(np.clip(self._noisy(m_cov, 0.10), 0, 1)),
            "prb_delta_mean":  self._noisy(m_prb),
            "ho_trigger_rate": self._noisy(m_ho, 0.20),
            "pred_entropy":    pred_entropy,
            "pred_kl_div":     pred_kl,
        }


def build_scenario(rng: np.random.Generator, malicious_kind: str,
                    attack_on: float = 600.0, attack_off: float = 1500.0):
    """
    Build the experiment's xApp set: the 3 benign archetypes plus one
    malicious variant of `malicious_kind` (sharing an archetype so the
    detector cannot trivially separate them by archetype identity).
    """
    xapps = [
        XApp("benign_ho",    "handover_pred", "benign", rng),
        XApp("benign_sched", "slice_sched",   "benign", rng),
        XApp("benign_kpm",   "kpm_reporter",  "benign", rng),
    ]
    # malicious variant uses the archetype it most plausibly impersonates
    arch = {"A1": "kpm_reporter",
            "A2": "slice_sched",
            "A3": "handover_pred"}[malicious_kind]
    xapps.append(
        XApp(f"mal_{malicious_kind}", arch, malicious_kind, rng,
             attack_on=attack_on, attack_off=attack_off))
    return xapps
