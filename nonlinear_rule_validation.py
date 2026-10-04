"""
nonlinear_rule_validation.py: validation of the non-linearity rule against known
linear and non-linear data-generating processes (reviewer comment 15a).

The rule uses only a feature's values and its feature attributions (FAs), so it
is validated on exact FAs computed from a known log-hazard function, without
fitting a model. The features cover the cases named by the reviewer, in which a
weak correlation between a feature and its FAs does not mean non-linearity, plus
a linear effect as the reference case:

  linear         0.6 x                         should not be selected
  nonmonotonic   0.4 (x^2 - 1)                 should be selected
  categorical    4 categories coded 0-3,       should be selected (it then enters the
                 effects 0, 0.6, -0.3, 0.3     model as category indicators); its
                                               effects are uncorrelated with the codes
  interaction    -0.4 x + 0.8 x z (z binary,   should not be selected: x is linear within
                 plus 0.3 z)                   each level of z, with slopes -0.4 and +0.4,
                                               so its overall correlation is near zero
  noisy          0.02 x                        should not be selected (weak linear effect,
                                               small relative to the estimation error)

Each case except the linear one gives a weak correlation between the feature and
its FAs. The rule is Recommender.nonlinear from your recommender.py, applied
unchanged, with the threshold on the gain in R2 set in recommender.py
(min_delta_r2).

FAs are the exact Shapley values of the log-hazard function against the mean
patient: for a main effect f(x), f(x) - f(x_ref); for the product g*x*z,
x receives g/2 (x - x_ref)(z + z_ref) and z the symmetric share. Gaussian noise
with SD --noise is added to every FA to represent estimation error, as with
KernelSHAP; for the weak effect this noise is large relative to the signal.

Each replicate draws new patients and new noise.

Usage:  python nonlinear_rule_validation.py --reps 200
"""
import argparse
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CANDIDATES = ["linear", "nonmonotonic", "categorical", "interaction", "noisy"]
EXPECTED = {"linear": "not selected", "nonmonotonic": "selected", "categorical": "selected",
            "interaction": "not selected", "noisy": "not selected"}
CAT_EFFECT = np.array([0.0, 0.6, -0.3, 0.3])


def exact_attributions(X):
    """Shapley values of the log-hazard function against the mean patient.
    A constant shift of a feature's FAs does not affect the rule (its regressions
    have an intercept), so categorical FAs are centred at their mean."""
    r = X.mean()
    phi = pd.DataFrame(index=X.index)
    phi["linear"] = 0.6 * (X["linear"] - r["linear"])
    phi["nonmonotonic"] = 0.4 * (X["nonmonotonic"] ** 2 - r["nonmonotonic"] ** 2)
    cat = CAT_EFFECT[X["categorical"].astype(int).to_numpy()]
    phi["categorical"] = cat - cat.mean()
    phi["noisy"] = 0.02 * (X["noisy"] - r["noisy"])
    g = 0.8                                                   # product term g * interaction * z
    phi["interaction"] = (-0.4 * (X["interaction"] - r["interaction"])
                          + g / 2 * (X["interaction"] - r["interaction"]) * (X["z"] + r["z"]))
    phi["z"] = 0.3 * (X["z"] - r["z"]) + g / 2 * (X["z"] - r["z"]) * (X["interaction"] + r["interaction"])
    return phi


def one_replicate(rng, n, noise):
    from recommender import Recommender
    X = pd.DataFrame({f: rng.normal(size=n) for f in ("linear", "nonmonotonic", "interaction", "noisy")})
    X["categorical"] = rng.integers(0, 4, n).astype(float)
    X["z"] = (rng.random(n) < 0.5).astype(float)
    phi = exact_attributions(X) + rng.normal(0, noise, (n, X.shape[1]))
    with contextlib.redirect_stdout(io.StringIO()):
        selected, tab = Recommender().nonlinear(phi[X.columns], X, CANDIDATES)
    gains = tab.set_index("feature")["delta_r2"]
    return {f: (f in selected, float(gains.loc[f])) for f in CANDIDATES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=5000, help="patients per replicate")
    ap.add_argument("--noise", type=float, default=0.1, help="SD of the estimation error added to the FAs")
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    reps = [one_replicate(rng, a.n, a.noise) for _ in range(a.reps)]
    rows = [dict(feature=f, expected=EXPECTED[f],
                 selected_pct=100 * np.mean([r[f][0] for r in reps]),
                 gain_in_r2_median=np.median([r[f][1] for r in reps])) for f in CANDIDATES]
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv("nonlinear_rule_validation_summary.csv", index=False)
    from recommender import Recommender
    print(f"{a.reps} replicates of {a.n} patients; noise SD {a.noise}; "
          f"threshold on the gain in R2 (min_delta_r2 in recommender.py) {Recommender().min_delta_r2}")
    print("Sensitivity: rows expected to be selected; false selection rate: rows expected not to be")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
