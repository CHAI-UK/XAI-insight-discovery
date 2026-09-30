"""
simulations.py: simulations of the interaction screen (C10) and the
non-linearity rule (C11), requested in review (comments 1 and 15a).

recommender.py and utils.py are not modified. Every rule is the shared code
called unchanged; only the data are simulated.

Two levels:

  A. Attributions (the reviewer's setup). Two strata of 2,362 and 4,566
     patients, a binary comorbidity at 5% and 25% prevalence, a strictly
     additive log hazard, and attributions computed exactly (with Gaussian
     noise standing in for estimation error). Compares the published rule
     (a stratum-specific mean-row reference, then a Wilcoxon rank-sum test
     of the partner's attributions across strata) with the current one (one
     reference for the subcohort, then the within-stratum contrast
     E[phi_y | y=1] - E[phi_y | y=0] compared across strata, as
     Recommender.stratified_screen computes it). Type I error, and power
     when the log hazard contains old x comorbidity.

  B. The whole pipeline. Survival times from a Cox model with known main
     effects and interactions; a random survival forest (or the true risk
     function, as an oracle exploratory model); KernelSHAP; the margin
     search, the three rules, the reference Cox model check and the
     parameter budget, exactly as utils.recommend runs them. With few
     features KernelSHAP enumerates every coalition, so the attributions are
     exact for the model explained. Records, per replicate, which features
     are excluded and flagged as non-linear, and which pairs are screened,
     confirmed and entered, against the truth.

Run:  python simulations.py            (full; results/sim_*.csv)
      python simulations.py --quick    (a few replicates, for checking)
"""
import argparse
import contextlib
import inspect
import io
import json
import os
import time

os.environ.setdefault('XAI_N_JOBS', '1')       # one worker per replicate; set before utils

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sksurv.util import Surv

import utils
from recommender import Recommender
from public_extras import _budgeted, _jaccard
from public_analyses import _jsonable

RESULT_DIR = utils.RESULT_DIR
ALPHA = 0.05
RSF_TREES = 300          # get_model uses 1000; lowered for the number of replicates


# =============================================================================
# A. Attribution level: the reviewer's setup
# =============================================================================
def reviewer_attributions(rng, n=(2362, 4566), prev=(0.05, 0.25), b=0.8, c=0.5, gamma=0.0,
                          noise=0.15):
    """Strata s = 0 (n[0] patients) and s = 1 (n[1]), a comorbidity y with
    prevalence prev[s], and log hazard eta = b y + c s + gamma y s.

    Exact Shapley values of eta against a single reference row r, for the
    two features (y, s):  phi_y = (y - r_y) (b + gamma (s + r_s) / 2).
    Published design: each stratum has its own reference, the stratum's mean
    row, r = (p_s, s). Current design: one reference for the subcohort,
    r = (mean y, mean s). Gaussian noise (SD `noise`) is added to both
    attributions, standing in for estimation error.
    Returns (X, phi_published, phi_current) as data frames.
    """
    s = np.r_[np.zeros(n[0]), np.ones(n[1])]
    y = (rng.random(s.size) < np.where(s == 1, prev[1], prev[0])).astype(float)
    X = pd.DataFrame({'s': s, 'y': y})

    def shapley(ry, rs):
        return pd.DataFrame({'y': (y - ry) * (b + gamma * (s + rs) / 2),
                             's': (s - rs) * (c + gamma * (y + ry) / 2)})

    p_s = np.where(s == 1, y[s == 1].mean(), y[s == 0].mean())
    published = shapley(p_s, s)
    current = shapley(y.mean(), s.mean())
    for phi in (published, current):
        phi += rng.normal(0, noise, phi.shape)
    return X, published, current


def attribution_screens(X, published, current, min_cell=20):
    """P values of the four combinations of design and test for the pair
    (s, y): the Wilcoxon rank-sum test of phi_y across the strata of s, and
    the within-stratum contrast of the current rule (stratified_screen,
    unchanged)."""
    rg = Recommender(min_cell=min_cell, seed=20)
    out = {}
    for design, phi in (('published reference', published), ('current reference', current)):
        s = X['s'].to_numpy()
        out[(design, 'Wilcoxon rank-sum (published)')] = float(
            mannwhitneyu(phi.loc[s == 0, 'y'], phi.loc[s == 1, 'y']).pvalue)
        with contextlib.redirect_stdout(io.StringIO()):
            t = rg.stratified_screen(s, 0.0, phi, X, 's', ['y'])
        out[(design, 'within-stratum contrast (current)')] = float(t['p_screen'].iloc[0])
    return out


ATTRIBUTION_GRID = ((0.0, 0.15), (0.05, 0.15), (0.1, 0.15), (0.2, 0.15),    # (gamma, noise SD)
                    (0.0, 0.05), (0.0, 0.3))


def simulate_attributions(reps, grid=ATTRIBUTION_GRID, seed=1):
    rows = []
    for gamma, noise in grid:
        rng = np.random.default_rng([seed, int(round(gamma * 1000)), int(round(noise * 1000))])
        ps = [attribution_screens(*reviewer_attributions(rng, gamma=gamma, noise=noise))
              for _ in range(reps)]
        for key in ps[0]:
            p = np.array([r[key] for r in ps])
            rows.append(dict(level='A: attributions', scenario='old x comorbidity',
                             gamma=gamma, reference=key[0], test=key[1], reps=reps, noise_sd=noise,
                             rejection_rate=float((p < ALPHA).mean()),
                             mc_se=float(np.sqrt((p < ALPHA).mean() * (1 - (p < ALPHA).mean()) / reps)),
                             median_p=float(np.median(p))))
    return pd.DataFrame(rows)


# =============================================================================
# B. Whole pipeline
# =============================================================================
# Each scenario: features, their true log-hazard terms, and the truth.
def _interaction_design(rng, n):
    """C10 design. old: binary (66%); comorb: binary, 5% if not old and 25% if
    old; x1, x2: standard normal with correlation 0.4; b2: binary (40%);
    z: standard normal, no effect. x2 has a quadratic main effect, which the
    product-term screen must not mistake for an interaction with x1."""
    old = (rng.random(n) < 0.66).astype(float)
    comorb = (rng.random(n) < np.where(old == 1, 0.25, 0.05)).astype(float)
    x1 = rng.normal(size=n)
    x2 = 0.4 * x1 + np.sqrt(1 - 0.4 ** 2) * rng.normal(size=n)
    return pd.DataFrame(dict(old=old, comorb=comorb, x1=x1, x2=x2,
                             b2=(rng.random(n) < 0.4).astype(float), z=rng.normal(size=n)))


INTERACTION_MAIN = dict(old=('lin', 0.7), comorb=('lin', 0.8), x1=('lin', 0.5), x2=('quad', 0.25),
                        b2=('lin', 0.4), z=('none', 0.0))


def _nonlinear_design(rng, n):
    """C11 design: one feature per shape. Continuous features are standard
    normal; cat is a nominal feature with four levels coded 0-3, ordlin an
    ordinal feature with five levels; bin a binary feature."""
    X = pd.DataFrame({k: rng.normal(size=n) for k in ('null', 'lin', 'quad', 'hinge', 'sine')})
    X['cat'] = rng.integers(0, 4, n).astype(float)
    X['ordlin'] = rng.integers(0, 5, n).astype(float)
    X['bin'] = (rng.random(n) < 0.4).astype(float)
    return X


NONLINEAR_MAIN = dict(null=('none', 0.0), lin=('lin', 0.5), quad=('quad', 0.4), hinge=('hinge', 0.8),
                      sine=('sine', 0.6), cat=('levels', (0.0, 0.6, -0.3, 0.4)),
                      ordlin=('lin', 0.2), bin=('lin', 0.5))


def _term(kind, coef, x):
    if kind == 'none':
        return np.zeros_like(x)
    if kind == 'lin':
        return coef * x
    if kind == 'quad':
        return coef * (x ** 2 - 1)
    if kind == 'hinge':
        return coef * np.maximum(x, 0)
    if kind == 'sine':
        return coef * np.sin(2 * x)
    if kind == 'levels':
        return np.asarray(coef)[x.astype(int)]
    raise ValueError(kind)


SCENARIOS = {
    # name: (design, {pair: coefficient of the product term})
    'additive': ('interaction', {}),
    'old x comorb': ('interaction', {('old', 'comorb'): 0.8}),
    'x1 x x2': ('interaction', {('x1', 'x2'): 0.4}),
    'old x x1': ('interaction', {('old', 'x1'): 0.5}),
    'nonlinear shapes': ('nonlinear', {}),
}


def scenario_spec(name):
    design, pairs = SCENARIOS[name]
    if design == 'interaction':
        return dict(make=_interaction_design, main=INTERACTION_MAIN, pairs=pairs,
                    continuous=['x1', 'x2', 'z'], ordinal=[])
    return dict(make=_nonlinear_design, main=NONLINEAR_MAIN, pairs=pairs,
                continuous=['null', 'lin', 'quad', 'hinge', 'sine'], ordinal=['cat', 'ordlin'])


def log_hazard(X, spec):
    eta = sum(_term(kind, coef, X[f].to_numpy(float)) for f, (kind, coef) in spec['main'].items())
    for (a, b), g in spec['pairs'].items():
        eta = eta + g * X[a].to_numpy(float) * X[b].to_numpy(float)
    return eta


class TrueRisk:
    """The data-generating risk exp(eta(x)), with the interface utils needs
    from an exploratory model: explaining it gives exact attributions of the
    true log hazard (the oracle exploratory model)."""

    def __init__(self, spec, columns):
        self.spec, self.columns = spec, list(columns)

    def predict(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        return np.exp(log_hazard(X, self.spec))


def simulate_data(spec, n, rng, base_rate=0.1, censor_rate=0.05, admin=10.0):
    """Exponential survival times with hazard base_rate * exp(eta),
    exponential censoring and administrative censoring at `admin`
    (about half the patients have the event)."""
    X = spec['make'](rng, n)
    t = rng.exponential(1 / (base_rate * np.exp(log_hazard(X, spec))))
    c = np.minimum(rng.exponential(1 / censor_rate, n), admin)
    return X, Surv.from_arrays(event=t <= c, time=np.minimum(t, c))


def _recommend_defaults():
    d = inspect.signature(utils.recommend).parameters
    return dict(min_cell=d['min_cell'].default, min_n=d['min_subcohort'].default,
                grid=d['margin_grid'].default, target=d['stability_target'].default,
                stable_steps=d['stable_steps'].default, nsamples=d['nsamples'].default,
                seed=d['seed'].default), d['shrinkage'].default


def run_pipeline(model, X, y, spec):
    """utils.recommend without its plots and files: the margin search, the
    rules at the chosen margin, then the parameter budget."""
    kw, shrinkage = _recommend_defaults()
    cache = {}
    eps, n_sub, _ = utils.choose_margin(model, X, y, {}, list(X.columns), groups={},
                                        continuous=spec['continuous'], ordinal=spec['ordinal'],
                                        result_cache=cache, **kw)
    rec_pre, info = cache['rec'], cache['info']
    rec = _budgeted(X, y, rec_pre, info['tests'], {}, shrinkage)
    return eps, n_sub, rec_pre, info, rec, cache['shap_sel']


def _pairs(spec_or_tests, col=None):
    if isinstance(spec_or_tests, dict):
        return sorted({'||'.join(sorted((a, b))) for a, bs in spec_or_tests.items() for b in bs})
    t = spec_or_tests
    return sorted(t.loc[t[col], 'pair'].unique()) if len(t) and col in t else []


def _old_nonlinear_rule(shap_sel, candidates, thresh=0.1):
    """The published non-linearity rule, for comparison: |corr(x, phi_x)| <
    0.1 in either subcohort."""
    out = set()
    for sh, sel in shap_sel.values():
        for c in candidates:
            x, phi = sel[c].to_numpy(float), sh[c].to_numpy(float)
            if np.std(x) > 0 and np.std(phi) > 0 and abs(np.corrcoef(x, phi)[0, 1]) < thresh:
                out.add(c)
    return sorted(out)


def one_replicate(scenario, explainer, n, rep, seed=2):
    spec = scenario_spec(scenario)
    rng = np.random.default_rng([seed, rep, list(SCENARIOS).index(scenario)])
    X, y = simulate_data(spec, n, rng)
    t0 = time.time()
    if explainer == 'rsf':
        model = utils.get_model('rf', int(rng.integers(2 ** 31))).set_params(
            n_estimators=RSF_TREES, n_jobs=1).fit(X, y)
    else:
        model = TrueRisk(spec, X.columns)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            eps, n_sub, rec_pre, info, rec, shap_sel = run_pipeline(model, X, y, spec)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            return dict(scenario=scenario, explainer=explainer, n=n, rep=rep, error=str(exc)[:200])
    tests = info['tests']
    cand_nl = spec['continuous'] + spec['ordinal']
    return dict(scenario=scenario, explainer=explainer, n=n, rep=rep, events=int(y['event'].sum()),
                margin=eps, subcohort_low=n_sub[0], subcohort_high=n_sub[1],
                n_pairs_tested=int(tests['pair'].nunique()) if len(tests) else 0,
                excluded=rec_pre['exclusion'], nonlinear=rec_pre['nonlinear'],
                nonlinear_old_rule=_old_nonlinear_rule(shap_sel, cand_nl),
                screened=_pairs(tests, 'screen_selected' if 'screen_selected' in tests else 'selected'),
                confirmed=_pairs(rec_pre['interaction']), entered=_pairs(rec['interaction']),
                true_pairs=['||'.join(sorted(p)) for p in spec['pairs']],
                seconds=time.time() - t0)


def summarise_pipeline(reps):
    """Rates over replicates, per scenario and exploratory model: pairs
    (interactions), the non-linearity rule (current and published), and
    exclusion."""
    ok = reps[reps['error'].isna()] if 'error' in reps else reps
    inter, nonl, excl = [], [], []
    for (scenario, explainer), g in ok.groupby(['scenario', 'explainer']):
        spec = scenario_spec(scenario)
        truth = set('||'.join(sorted(p)) for p in spec['pairs'])
        for stage in ('screened', 'confirmed', 'entered'):
            false_any = g[stage].map(lambda s: bool(set(s) - truth))
            row = dict(scenario=scenario, explainer=explainer, stage=stage, reps=len(g),
                       any_false_pair=float(false_any.mean()),
                       mean_false_pairs=float(g[stage].map(lambda s: len(set(s) - truth)).mean()),
                       true_pair_found=float(g[stage].map(lambda s: bool(truth & set(s))).mean())
                       if truth else np.nan,
                       note=('exact attributions: the screen returns P = 1 when the attribution '
                             'covariance is numerically zero, so no pair can be screened')
                       if explainer == 'oracle' else '')
            inter.append(row)
        for f, (kind, coef) in spec['main'].items():
            excl.append(dict(scenario=scenario, explainer=explainer, feature=f, shape=kind,
                             no_effect=kind == 'none', reps=len(g),
                             excluded=float(g['excluded'].map(lambda s: f in s).mean())))
            if f not in spec['continuous'] + spec['ordinal']:
                continue
            truth_nl = kind in ('quad', 'hinge', 'sine', 'levels')
            nonl.append(dict(scenario=scenario, explainer=explainer, feature=f, shape=kind,
                             truly_nonlinear=truth_nl, reps=len(g),
                             flagged_current=float(g['nonlinear'].map(lambda s: f in s).mean()),
                             flagged_old_rule=float(g['nonlinear_old_rule'].map(lambda s: f in s).mean())))
    inter = pd.DataFrame(inter)
    inter['mc_se_false'] = np.sqrt(inter['any_false_pair'] * (1 - inter['any_false_pair']) / inter['reps'])
    return inter, pd.DataFrame(nonl), pd.DataFrame(excl)


NULL_SCENARIOS = ('additive', 'nonlinear shapes')      # no interaction in the truth


def simulate_pipeline(reps_null, reps_alt, n=2000, explainers=('oracle', 'rsf'),
                      scenarios=tuple(SCENARIOS), n_jobs=-1, seed=2, checkpoint=None):
    """Run the replicates in parallel. With checkpoint (a .jsonl path), each
    finished replicate is appended as it completes, and replicates already
    in the file are not run again, so an interrupted run resumes."""
    from joblib import Parallel, delayed
    done = []
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            done = [json.loads(line) for line in fh if line.strip()]
    have = {(d['scenario'], d['explainer'], d['n'], d['rep']) for d in done}
    jobs = [(s, e, n, r, seed) for s in scenarios for e in explainers
            for r in range(reps_null if s in NULL_SCENARIOS else reps_alt)
            if (s, e, n, r) not in have]
    print(f'[simulation] {len(jobs)} replicates to run, {len(done)} already done', flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5, return_as='generator_unordered')(
        delayed(one_replicate)(*j) for j in jobs)
    for r in results:
        done.append(r)
        if checkpoint:
            with open(checkpoint, 'a') as fh:
                fh.write(json.dumps(_jsonable(r)) + '\n')
    return pd.DataFrame(done)


# =============================================================================
# Command line
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--quick', action='store_true', help='a few replicates, for checking')
    ap.add_argument('--reps-attr', type=int, default=1000)
    ap.add_argument('--reps-null', type=int, default=200,
                    help='replicates of the scenarios without interactions')
    ap.add_argument('--reps-alt', type=int, default=100,
                    help='replicates of each scenario with an interaction')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--out', default=RESULT_DIR)
    a = ap.parse_args()
    if a.quick:
        a.reps_attr, a.reps_null, a.reps_alt = 50, 2, 1
    os.makedirs(a.out, exist_ok=True)
    path = lambda name: os.path.join(a.out, f'sim_{name}.csv')

    attr = simulate_attributions(a.reps_attr)
    attr.to_csv(path('attribution_level'), index=False)
    print(attr.round(3).to_string(index=False), flush=True)

    reps = simulate_pipeline(a.reps_null, a.reps_alt, n=a.n,
                             checkpoint=os.path.join(a.out, 'sim_pipeline_replicates.jsonl'))
    inter, nonl, excl = summarise_pipeline(reps)
    inter.to_csv(path('interaction_summary'), index=False)
    nonl.to_csv(path('nonlinear_summary'), index=False)
    excl.to_csv(path('exclusion_summary'), index=False)
    print(inter.round(3).to_string(index=False))
    print(nonl.round(3).to_string(index=False))

    kw, shrinkage = _recommend_defaults()
    settings = dict(alpha=ALPHA, reps_attribution=a.reps_attr, reps_null=a.reps_null,
                    reps_alt=a.reps_alt, n=a.n,
                    rsf=dict(n_estimators=RSF_TREES, note='other settings as utils.get_model'),
                    recommend_defaults=dict(kw, shrinkage=shrinkage), scenarios={
                        s: dict(pairs={'x'.join(p): g for p, g in scenario_spec(s)['pairs'].items()},
                                main=scenario_spec(s)['main']) for s in SCENARIOS},
                    failed=int(reps['error'].notna().sum()) if 'error' in reps else 0)
    with open(os.path.join(a.out, 'sim_settings.json'), 'w') as fh:
        json.dump(_jsonable(settings), fh, indent=2, default=str)


if __name__ == '__main__':
    main()
