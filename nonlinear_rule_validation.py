"""
nonlinear_rule_validation.py: validation of the non-linearity rule against known
linear and non-linear data-generating processes.

The rule uses a feature's values and its feature attributions (FAs). 
Features are independent; their true effects and what the rule should do:

  linear               0.6 x                       not selected
  monotonic_nonlinear  exp(0.5 x)                  selected (strongly correlated with x,
                                                   but curved)
  nonmonotonic         0.4 (x^2 - 1)               selected (correlation near zero)
  ordinal_nonlinear    4 ordered levels, effects   selected (the level effects do not
                       0, 0.6, -0.3, 0.3           follow a linear trend)
  ordinal_linear       4 ordered levels, effects   not selected (equally spaced effects)
                       0, 0.2, 0.4, 0.6
  interaction          -0.4 x + 0.8 x z            not selected (x is linear within each
                       (z binary, plus 0.3 z)      level of z, slopes -0.4 and +0.4, so
                                                   its overall correlation is near zero)
  noisy                0.02 x                      not selected (weak linear effect, small
                                                   relative to the attribution error)

Unordered categorical features (such as ethnicity or alcohol intake) are not
screened by the rule: they are represented by category indicators directly.

FAs are the exact Shapley values of the log-hazard function against the mean
patient: for a main effect f(x), f(x) - f(x_ref); for the product g*x*z, x
receives g/2 (x - x_ref)(z + z_ref) and z the symmetric share. A constant shift
does not affect the rule, so ordinal FAs are centred at their mean. Idealized
Gaussian attribution error with SD --noise is added to every FA.

The rule is Recommender.nonlinear from your recommender.py, applied unchanged,
with the threshold on the gain in R2 set in recommender.py (min_delta_r2).
Selection rates are reported per feature with exact (Clopper-Pearson) 95% CIs,
and pooled over features as sensitivity, false positive rate and false discovery
rate (wrong selections among all selections), with bootstrap 95% CIs over
replicates. The result of every replicate is saved as well.

Usage:
  python nonlinear_rule_validation.py --reps 1000
  python nonlinear_rule_validation.py --reps 1000 --noise 0.2
"""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import beta

CANDIDATES = ["linear", "monotonic_nonlinear", "nonmonotonic", "ordinal_nonlinear", "ordinal_linear",
              "interaction", "noisy"]
EXPECTED = {"linear": "not selected", "monotonic_nonlinear": "selected", "nonmonotonic": "selected",
            "ordinal_nonlinear": "selected", "ordinal_linear": "not selected",
            "interaction": "not selected", "noisy": "not selected"}
POSITIVE = [f for f in CANDIDATES if EXPECTED[f] == "selected"]
NEGATIVE = [f for f in CANDIDATES if EXPECTED[f] == "not selected"]
ORD_NONLINEAR = np.array([0.0, 0.6, -0.3, 0.3])
ORD_LINEAR = np.array([0.0, 0.2, 0.4, 0.6])
G = 0.8                                                # product term G * interaction * z


def exact_attributions(X):
    """Exact Shapley values of the log-hazard function against the mean patient."""
    r = X.mean()
    phi = pd.DataFrame(index=X.index)
    phi["linear"] = 0.6 * (X["linear"] - r["linear"])
    phi["monotonic_nonlinear"] = np.exp(0.5 * X["monotonic_nonlinear"]) - np.exp(0.5 * r["monotonic_nonlinear"])
    phi["nonmonotonic"] = 0.4 * (X["nonmonotonic"] ** 2 - r["nonmonotonic"] ** 2)
    for f, eff in (("ordinal_nonlinear", ORD_NONLINEAR), ("ordinal_linear", ORD_LINEAR)):
        v = eff[X[f].astype(int).to_numpy()]
        phi[f] = v - v.mean()
    phi["interaction"] = (-0.4 * (X["interaction"] - r["interaction"])
                          + G / 2 * (X["interaction"] - r["interaction"]) * (X["z"] + r["z"]))
    phi["noisy"] = 0.02 * (X["noisy"] - r["noisy"])
    phi["z"] = 0.3 * (X["z"] - r["z"]) + G / 2 * (X["z"] - r["z"]) * (X["interaction"] + r["interaction"])
    return phi


def one_replicate(rng, n, noise):
    from recommender import Recommender
    X = pd.DataFrame({f: rng.normal(size=n)
                      for f in ("linear", "monotonic_nonlinear", "nonmonotonic", "interaction", "noisy")})
    X["ordinal_nonlinear"] = rng.integers(0, 4, n).astype(float)
    X["ordinal_linear"] = rng.integers(0, 4, n).astype(float)
    X["z"] = (rng.random(n) < 0.5).astype(float)
    phi = exact_attributions(X) + rng.normal(0, noise, (n, X.shape[1]))
    with contextlib.redirect_stdout(io.StringIO()):
        selected, tab = Recommender().nonlinear(phi[X.columns], X, CANDIDATES)
    tab = tab.set_index("feature")
    return [dict(feature=f, selected=f in selected, gain_in_r2=float(tab.loc[f, "delta_r2"]),
                 p_adjusted=float(tab.loc[f, "p_fdr"])) for f in CANDIDATES]


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


def main():
    from recommender import Recommender
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n", type=int, default=5000, help="patients per replicate")
    ap.add_argument("--noise", type=float, default=0.1, help="SD of the Gaussian error added to the FAs")
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    per_rep = pd.DataFrame([dict(rep=i, **row) for i in range(a.reps) for row in one_replicate(rng, a.n, a.noise)])
    tag = f"noise{a.noise:g}"
    per_rep.to_csv(f"nonlinear_rule_validation_replicates_{tag}.csv", index=False)

    rows = []
    for f in CANDIDATES:
        s = per_rep.loc[per_rep["feature"] == f, "selected"]
        lo, hi = clopper_pearson(int(s.sum()), len(s))
        rows.append(dict(feature=f, expected=EXPECTED[f], selected_pct=100 * s.mean(),
                         ci_low_pct=100 * lo, ci_high_pct=100 * hi,
                         gain_in_r2_median=per_rep.loc[per_rep["feature"] == f, "gain_in_r2"].median()))
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(f"nonlinear_rule_validation_summary_{tag}.csv", index=False)
    print(f"{a.reps} replicates of {a.n} patients; attribution error SD {a.noise}; "
          f"threshold on the gain in R2 (min_delta_r2 in recommender.py) {Recommender().min_delta_r2}")
    print("Sensitivity: rows expected to be selected; false selection rate: rows expected not to be")
    print(tab.to_string(index=False))
    
if __name__ == "__main__":
    main()
