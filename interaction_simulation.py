"""
interaction_simulation.py: type I error of the interaction screen under the
additive null interaction (type I error), and its power for one real interaction.

The setup: two strata of 2362 and 4566 patients, a binary comorbidity
y with prevalence 5% in one stratum and 25% in the other, and the log hazard

    eta = b y + c s + gamma y s       (b = 0.8, c = 0.5)

which is strictly additive when gamma = 0. Attributions are the exact Shapley
values of eta for the 2 features (y, s) against a single reference row r:

    phi_y = (y - r_y) (b + gamma (s + r_s) / 2)

with idealized Gaussian attribution error (SD 0.15) added. Two designs:

  published reference   each stratum has its own reference, its mean row
  current reference     one reference for the subcohort, its mean row

and two tests of whether y's attributions differ between the strata:

  Wilcoxon rank-sum (published)        phi_y compared across strata
  within-stratum contrast (current)    E[phi_y | y=1] - E[phi_y | y=0] compared across
                                       strata, by Recommender.stratified_screen, unchanged

The published rule is the published reference with the Wilcoxon test; the
current rule is the current reference with the within-stratum contrast. The
other 2 combinations show which part of the change matters. Power is reported
for the current rule at gamma = 0.1 and 0.2 (the published rule rejects
even without an interaction, so its power is not informative).

Continuous partners are screened by product-term regression instead. Its check
uses a continuous focal feature x with a curved (U-shaped) main effect and a
continuous partner z correlated with it (correlation 0.6), the hardest case for
this test, since a partner correlated with x could absorb x's curve:

    eta = 0.4 (x^2 - 1) + 0.5 z + gamma x z

with exact Shapley values against the mean patient, the same attribution error,
and Recommender.stratified_screen, unchanged, which routes a continuous partner
to the product-term regression.

Usage:  python interaction_simulation.py --reps 1000
        python interaction_simulation.py --reps 1000 --noise 0.3   (sensitivity to the error)
The P value and effect size of every replicate are saved as well.
"""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import beta, mannwhitneyu

ALPHA = 0.05
SETTINGS = {"no interaction (type I error)": 0.0,
            "interaction gamma = 0.1 (power)": 0.1,
            "interaction gamma = 0.2 (power)": 0.2}
CURRENT = ("current reference", "within-stratum contrast (current)")


def attributions(rng, gamma, noise=0.15, n=(2362, 4566), prev=(0.05, 0.25), b=0.8, c=0.5):
    s = np.r_[np.zeros(n[0]), np.ones(n[1])]
    y = (rng.random(s.size) < np.where(s == 1, prev[1], prev[0])).astype(float)
    X = pd.DataFrame({"s": s, "y": y})

    def shapley(ry, rs):
        return pd.DataFrame({"y": (y - ry) * (b + gamma * (s + rs) / 2),
                             "s": (s - rs) * (c + gamma * (y + ry) / 2)})

    stratum_mean = np.where(s == 1, y[s == 1].mean(), y[s == 0].mean())
    designs = {"published reference": shapley(stratum_mean, s),
               "current reference": shapley(y.mean(), s.mean())}
    for phi in designs.values():
        phi += rng.normal(0, noise, phi.shape)
    return X, designs


def p_values(X, designs):
    """P value and effect size of each design and test. The effect size of the
    within-stratum contrast is the difference between the 2 contrasts; the
    Wilcoxon test has none (NaN)."""
    from recommender import Recommender
    rg = Recommender(min_cell=20, seed=20)
    s = X["s"].to_numpy()
    out = {}
    for design, phi in designs.items():
        out[(design, "Wilcoxon rank-sum (published)")] = (
            float(mannwhitneyu(phi.loc[s == 0, "y"], phi.loc[s == 1, "y"]).pvalue), np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            t = rg.stratified_screen(s, 0.0, phi, X, "s", ["y"])
        out[(design, "within-stratum contrast (current)")] = (
            float(t["p_screen"].iloc[0]), float(t["effect_size"].iloc[0]))
    return out


def continuous_attributions(rng, gamma, noise=0.15, n=6928, rho=0.6):
    x = rng.normal(size=n)
    z = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    X = pd.DataFrame({"x": x, "z": z})
    rx, rz = x.mean(), z.mean()
    phi = pd.DataFrame({"x": 0.4 * (x ** 2 - rx ** 2) + gamma / 2 * (x - rx) * (z + rz),
                        "z": 0.5 * (z - rz) + gamma / 2 * (z - rz) * (x + rx)})
    return X, phi + rng.normal(0, noise, phi.shape)


def continuous_p_value(X, phi):
    """P value and effect size (standardized product-term coefficient)."""
    from recommender import Recommender
    rg = Recommender(min_cell=20, seed=20)
    with contextlib.redirect_stdout(io.StringIO()):
        t = rg.stratified_screen(X["x"].to_numpy(), None, phi, X, "x", ["z"])
    return float(t["p_screen"].iloc[0]), float(t["effect_size"].iloc[0])


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.15,
                    help="SD of the attribution error; 0 makes the tests degenerate (see below)")
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()
    if a.noise == 0:
        print("Note: with no attribution error the within-stratum contrasts have zero variance, so their "
              "P values are set to 1 under both hypotheses, and the product-term regression can pick up "
              "the spline's approximation error; results with --noise 0 are not interpretable.")
    per_rep = []
    for label, gamma in SETTINGS.items():
        rng = np.random.default_rng([a.seed, int(gamma * 1000)])
        for rep in range(a.reps):
            for (design, test), (p, eff) in p_values(*attributions(rng, gamma, a.noise)).items():
                if gamma > 0 and (design, test) != CURRENT:
                    continue            # power only for the current rule; the published one rejects anyway
                per_rep.append(dict(setting=label, partner="binary", reference=design, test=test,
                                    rep=rep, p=p, effect_size=eff))
        rng = np.random.default_rng([a.seed, int(gamma * 1000), 1])
        for rep in range(a.reps):
            p, eff = continuous_p_value(*continuous_attributions(rng, gamma, a.noise))
            per_rep.append(dict(setting=label, partner="continuous", reference="current reference",
                                test="product-term regression (current)", rep=rep, p=p, effect_size=eff))
    per_rep = pd.DataFrame(per_rep)
    tag = f"noise{a.noise:g}"
    per_rep.to_csv(f"interaction_simulation_replicates_{tag}.csv", index=False)

    rows = []
    for (setting, partner, ref, test), g in per_rep.groupby(["setting", "partner", "reference", "test"],
                                                            sort=False):
        k = int((g["p"] < ALPHA).sum())
        lo, hi = clopper_pearson(k, len(g))
        rows.append(dict(setting=setting, partner=partner, reference=ref, test=test,
                         rejection_pct=100 * k / len(g), ci_low_pct=100 * lo, ci_high_pct=100 * hi))
    tab = pd.DataFrame(rows).round(2)
    tab.to_csv(f"interaction_simulation_summary_{tag}.csv", index=False)
    print(f"{a.reps} replicates per setting; attribution error SD {a.noise}; rejection at P < {ALPHA}, "
          f"exact 95% CIs")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
