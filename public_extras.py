"""
public_extras.py: additional analyses on the open data sets, requested in
review and not run on DataLoch.
"""
import contextlib
import io
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sksurv.linear_model import CoxPHSurvivalAnalysis

import utils
from recommender import Recommender
from public_analyses import _defaults, _jsonable, export_coefficients

RESULT_DIR = utils.RESULT_DIR
PARTS = ('exclusion', 'nonlinear', 'interaction')
COX_ALPHA = _defaults(utils.fit_and_score)['alpha']


# =============================================================================
# Shared helpers
# =============================================================================
def _fit_cox(X, y):
    """The Cox model as fit_and_score fits it."""
    return CoxPHSurvivalAnalysis(alpha=COX_ALPHA, ties='efron').fit(X, y)


def _subset(rec, parts):
    """rec with only the recommendation types in parts."""
    return dict(exclusion=list(rec['exclusion']) if 'exclusion' in parts else [],
                nonlinear=list(rec['nonlinear']) if 'nonlinear' in parts else [],
                interaction=dict(rec['interaction']) if 'interaction' in parts else {})


def _design(rec, X_train, X_test, parts=PARTS, groups=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return Recommender.apply(X_train, X_test, _subset(rec, parts), variant='all',
                                 groups=groups)


def _pair_names(spec):
    return sorted({'||'.join(sorted((a, b))) for a, bs in spec.items() for b in bs})


def confirmed_pairs(tests):
    """Every pair that passed the screen and the reference Cox model check,
    as an interaction spec, before the parameter budget is applied."""
    if not len(tests) or 'screen_selected' not in tests:
        return {}
    sel = (tests.loc[tests['selected']].sort_values('effect_vs_scale', ascending=False)
           .drop_duplicates('pair'))
    spec = {}
    for _, r in sel.iterrows():
        spec.setdefault(r['strat'], []).append(r['partner'])
    return spec


def _budgeted(X, y, rec_pre, tests, groups, shrinkage):
    """The parameter budget step of utils.recommend, applied to one screening
    result: the same calls, in the same order."""
    groups = groups or {}
    base_frame, _ = _design(dict(rec_pre, interaction={}), X, X, groups=groups)
    riley = Recommender.riley_parameter_budget(base_frame, y, shrinkage=shrinkage)
    budget = max(0, riley['p_max'] - base_frame.shape[1])
    spec, _ = Recommender.spec_within_budget(tests, base_frame, budget, groups)
    return dict(exclusion=rec_pre['exclusion'], nonlinear=rec_pre['nonlinear'], interaction=spec)


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def _sets(rec_pre, tests, rec_final):
    """Recommendation sets of one run, keyed by kind."""
    screened = (tests.loc[tests['screen_selected'] if 'screen_selected' in tests
                          else tests['selected'], 'pair'].unique().tolist()
                if len(tests) else [])
    return dict(exclusion=sorted(rec_pre['exclusion']), nonlinear=sorted(rec_pre['nonlinear']),
                interaction_screened=sorted(screened),
                interaction_confirmed=_pair_names(rec_pre['interaction']),
                interaction_entered=_pair_names(rec_final['interaction']))


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _out(out_dir, tag, name, ext='csv'):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{tag}_{name}.{ext}')


# =============================================================================
# 3.0  Interaction sensitivity model (budget ignored)
# =============================================================================
def interaction_sensitivity(rec, tests, X_train, X_test, y_train, y_test, t0, tag,
                            groups=None, plot_dir='plots/', out_dir=RESULT_DIR, evaluator=None):
    """The final model plus every pair confirmed in the reference Cox model,
    whether or not it fitted within the parameter budget.

    This is a sensitivity analysis outside the method: the budget exists to
    stop exactly this model from overfitting, so its test-set gain is reported
    only to show what the budget withheld. Writes
    <tag>_interaction_sensitivity.csv (the final model and this one, with
    paired differences in C) and <tag>_interaction_sensitivity_hr.tsv.
    Returns the augmented rec, or None when no confirmed pair was withheld.
    """
    spec = confirmed_pairs(tests)
    added = sorted(set(_pair_names(spec)) - set(_pair_names(rec['interaction'])))
    path = _out(out_dir, tag, 'interaction_sensitivity')
    if not added:
        pd.DataFrame([dict(model='Cox_all_confirmed', note='no confirmed pair was withheld '
                           'by the budget; the sensitivity model equals Cox_all')]).to_csv(path, index=False)
        print(f'[sensitivity/{tag}] no confirmed pair outside the final model')
        return None
    rec_s = dict(rec, interaction=spec)
    evaluator = evaluator or utils.MetricEval(times=utils.evaluation_times(y_train))
    calib = utils.CalibrationPerform(t0=t0, n_bins=_defaults(utils.evaluate_recommendations)['n_bins'],
                                     kind='survival', save_folder=plot_dir)
    base = _fit_cox(X_train, y_train)
    s_base = utils.cox_risk_score(base, X_test)
    Atr, Ate = _design(rec, X_train, X_test, groups=groups)
    s_all = utils.cox_risk_score(_fit_cox(Atr, y_train), Ate)
    Str, Ste = _design(rec_s, X_train, X_test, groups=groups)
    model, rep = utils.fit_and_score(Str, y_train, Ste, y_test, evaluator, calib, 'Cox_all_confirmed')
    s_sens = utils.cox_risk_score(model, Ste)
    # column names as in <tag>_model_comparison.csv and its commented-out
    # Cox_all_lasso row
    d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, s_base, s_sens)
    rep.update(delta_c_vs_baseline=d, delta_ci_low=lo, delta_ci_high=hi, delta_p=p)
    d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, s_all, s_sens)
    rep.update(delta_c_vs_cox_all=d, delta_vs_cox_all_ci_low=lo, delta_vs_cox_all_ci_high=hi,
               delta_vs_cox_all_p=p)

    # likelihood ratio test of the added terms on the training data
    from lifelines import CoxPHFitter
    def loglik(A):
        d = A.copy()
        d['time'], d['event'] = np.asarray(y_train['time'], float), np.asarray(y_train['event'], int)
        return CoxPHFitter().fit(d, 'time', 'event').log_likelihood_
    k = Str.shape[1] - Atr.shape[1]
    lr = 2 * (loglik(Str) - loglik(Atr))
    rep.update(pairs_added='; '.join(added), n_terms_added=k, train_lr=lr,
               train_lr_df=k, train_lr_p=float(chi2.sf(lr, k)),
               events_per_parameter=int(np.sum(y_train['event'])) / Str.shape[1])
    comparison = pd.read_csv(os.path.join(out_dir, f'{tag}_model_comparison.csv'))
    rows = pd.concat([comparison[comparison['model'] == 'Cox_all'], pd.DataFrame([rep])],
                     ignore_index=True)
    rows.to_csv(path, index=False)
    export_coefficients(model, Str, y_train, _out(out_dir, tag, 'interaction_sensitivity_hr', 'tsv'))
    print(f'[sensitivity/{tag}] added {added}: C {rep["harrell_c"]:.3f}, '
          f'dC vs Cox_all {rep["delta_c_vs_cox_all"]:+.3f} '
          f'({rep["delta_vs_cox_all_ci_low"]:+.3f} to {rep["delta_vs_cox_all_ci_high"]:+.3f})')
    return rec_s


# =============================================================================
# 3.1  Calibration-in-the-large
# =============================================================================
def _survival_at(model, X, times):
    """Predicted S(t_i | x_i) for each row at its own time t_i."""
    fns = model.predict_survival_function(X)
    return np.array([float(fn(min(max(t, fn.domain[0]), fn.domain[1]))) for fn, t in zip(fns, times)])


def calibration_in_the_large(model, X, y, t0):
    """Observed over expected events up to t0.
    """
    time, event = np.asarray(y['time'], float), np.asarray(y['event'], bool)
    tt = np.minimum(time, t0)
    observed = int((event & (time <= t0)).sum())
    expected = float(-np.log(np.clip(_survival_at(model, X, tt), 1e-300, 1)).sum())
    a = np.log(observed / expected) if observed and expected > 0 else np.nan
    se = 1 / np.sqrt(observed) if observed else np.nan
    km = utils.CalibrationPerform(t0=t0).km_s_at(time, event)
    pred = _survival_at(model, X, np.full(len(time), float(t0)))
    return dict(t0=t0, n=len(time), observed_events=observed, expected_events=expected,
                oe_ratio=observed / expected if expected > 0 else np.nan,
                citl=a, citl_ci_low=a - 1.96 * se, citl_ci_high=a + 1.96 * se,
                citl_p=float(2 * norm.sf(abs(a) / se)) if observed else np.nan,
                km_risk_t0=1 - km, mean_predicted_risk_t0=float(1 - pred.mean()))


def calibration_intercept(models, y_test, t0, tag, out_dir=RESULT_DIR):
    """calibration_in_the_large for each (model, X_test) in models, written to
    <tag>_calibration_intercept.csv."""
    tab = pd.DataFrame([dict(model=name, **calibration_in_the_large(m, X, y_test, t0))
                        for name, (m, X) in models.items()])
    tab.to_csv(_out(out_dir, tag, 'calibration_intercept'), index=False)
    return tab


# =============================================================================
# 3.2  Proportional hazards: global test and residual plots
# =============================================================================
def _ph_frame(X, y):
    d = X.copy()
    d['time'] = np.asarray(y['time'], float)
    d['event'] = np.asarray(y['event'], int)
    return d


def schoenfeld_tests(fitter, d, time_transform='km'):
    """Per-term and global Schoenfeld tests of a fitted lifelines CoxPHFitter.
    """
    from lifelines.statistics import TimeTransformers
    events, durations, weights = fitter.event_observed, fitter.durations, fitter.weights
    times = TimeTransformers().get(time_transform)(durations, events, weights)[events.values]
    g = np.asarray(times, float) - float(np.mean(times))
    R = fitter.compute_residuals(d, kind='schoenfeld')[list(fitter.params_.index)].to_numpy(float)
    V = fitter.variance_matrix_.loc[fitter.params_.index, fitter.params_.index].to_numpy(float)
    D, gg = float(events.sum()), float(g @ g)
    u = g @ R
    Vu = V @ u
    per = D * Vu ** 2 / (np.diag(V) * gg)
    stat = float(D * u @ Vu / gg)
    terms = pd.DataFrame(dict(term=list(fitter.params_.index), test_statistic=per,
                              p=chi2.sf(per, 1)))
    return terms, dict(test_statistic=stat, df=len(u), p=float(chi2.sf(stat, len(u))),
                       time_transform=time_transform, n_events=int(D))


def _residual_plot(fitter, d, path, time_transform='km', title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lifelines.statistics import TimeTransformers
    from statsmodels.nonparametric.smoothers_lowess import lowess
    events, durations, weights = fitter.event_observed, fitter.durations, fitter.weights
    t = np.asarray(TimeTransformers().get(time_transform)(durations, events, weights)[events.values], float)
    S = fitter.compute_residuals(d, kind='scaled_schoenfeld')
    terms = list(fitter.params_.index)
    ncol = min(4, len(terms))
    nrow = int(np.ceil(len(terms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for ax, term in zip(axes.ravel(), terms):
        r = S[term].to_numpy(float) + float(fitter.params_[term])     # beta(t) estimate
        ax.scatter(t, r, s=4, alpha=0.3, color='grey')
        fit = lowess(r, t, frac=0.6, return_sorted=True)
        ax.plot(fit[:, 0], fit[:, 1], color='C0')
        ax.axhline(float(fitter.params_[term]), color='k', linestyle='--', linewidth=0.8)
        ax.set_title(term, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(terms):]:
        ax.axis('off')
    fig.supxlabel(f'Time ({time_transform}-transformed)', fontsize=8)
    fig.supylabel('Scaled Schoenfeld residual + coefficient', fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def ph_tests(designs, y_train, tag, plot_dir='plots/', out_dir=RESULT_DIR):
    """Schoenfeld tests for each training design in designs ({name: X}),
    fitted as ph_assumption_report fits the final model (same penalty and
    time transform).
    """
    settings = _defaults(utils.ph_assumption_report, skip=('out',))
    globals_, terms = [], []
    for name, X in designs.items():
        fitter, lifelines_tab = _quiet(utils.ph_assumption_report, X, y_train, **settings)
        d = _ph_frame(X, y_train)
        per, glob = schoenfeld_tests(fitter, d, settings['time_transform'])
        ref = lifelines_tab.set_index('term')['test_statistic']          # sorted by name
        if not np.allclose(per['test_statistic'], ref.loc[per['term']], rtol=1e-6):
            raise RuntimeError('per-term statistics do not reproduce lifelines')
        globals_.append(dict(model=name, n_terms=X.shape[1],
                             n_terms_p_below_05=int((per['p'] < 0.05).sum()),
                             penalizer=settings['penalizer'], **glob))
        terms.append(per.assign(model=name))
        os.makedirs(plot_dir, exist_ok=True)
        _residual_plot(fitter, d, os.path.join(plot_dir, f'ph_residuals_{tag}_{name}.pdf'),
                       settings['time_transform'], title=f'{tag}: {name}')
    g = pd.DataFrame(globals_)
    g.to_csv(_out(out_dir, tag, 'ph_global'), index=False)
    t = pd.concat(terms, ignore_index=True)
    t.insert(0, 'model', t.pop('model'))
    t.to_csv(_out(out_dir, tag, 'ph_terms'), index=False)
    return g, t


# =============================================================================
# 3.3  Events per parameter, coefficient stability
# =============================================================================
def events_per_parameter(designs, y_train, y_test, subcohorts, riley_p_max, tag, out_dir=RESULT_DIR):
    """Patients and events per split and subcohort, and parameters and events
    per parameter (training events / columns) of each model.
    """
    ev_train = int(np.sum(y_train['event']))
    rows = [dict(row='split', name='train', n=len(y_train), events=ev_train),
            dict(row='split', name='test', n=len(y_test), events=int(np.sum(y_test['event'])))]
    ev = pd.Series(np.asarray(y_train['event'], bool), index=next(iter(designs.values())).index)
    rows += [dict(row='subcohort', name=lv, n=len(idx), events=int(ev.loc[idx].sum()))
             for lv, idx in subcohorts.items()]
    rows += [dict(row='model', name=name, n=len(y_train), events=ev_train,
                  n_parameters=X.shape[1], n_interaction_terms=sum('__x__' in c for c in X.columns),
                  events_per_parameter=ev_train / max(X.shape[1], 1), riley_p_max=riley_p_max)
             for name, X in designs.items()]
    tab = pd.DataFrame(rows)
    tab.to_csv(_out(out_dir, tag, 'events_per_parameter'), index=False)
    return tab


def coefficient_stability(X, y, tag, name='Cox_all', n_boot=500, seed=utils.RANDOM_STATE,
                          n_jobs=utils.N_JOBS, out_dir=RESULT_DIR):
    """Refit the Cox model on n_boot bootstrap samples of the training data,
    with the design (columns, centring) held fixed, and summarise each
    coefficient: bootstrap SD against the model SE, percentile interval, and
    the share of samples with the sign of the full-data estimate.
    """
    from joblib import Parallel, delayed
    from lifelines import CoxPHFitter
    full = _fit_cox(X, y).coef_
    se = CoxPHFitter().fit(_ph_frame(X, y), 'time', 'event').standard_errors_[X.columns].to_numpy()
    seeds = np.random.SeedSequence(seed).generate_state(n_boot)

    def one(sd):
        i = np.random.default_rng(sd).integers(0, len(X), len(X))
        try:
            return _fit_cox(X.iloc[i], y[i]).coef_
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return np.full(X.shape[1], np.nan)

    B = np.vstack(Parallel(n_jobs=n_jobs, batch_size=8)(delayed(one)(s) for s in seeds))
    ok = np.isfinite(B).all(axis=1)
    Bk = B[ok]
    tab = pd.DataFrame(dict(model=name, term=X.columns, coef=full, se_model=se,
                            boot_mean=Bk.mean(0), boot_sd=Bk.std(0, ddof=1),
                            boot_ci_low=np.percentile(Bk, 2.5, axis=0),
                            boot_ci_high=np.percentile(Bk, 97.5, axis=0),
                            sign_agreement=(np.sign(Bk) == np.sign(full)).mean(0),
                            n_boot=int(ok.sum()), n_failed=int((~ok).sum())))
    tab['sd_ratio'] = tab['boot_sd'] / tab['se_model']
    tab.to_csv(_out(out_dir, tag, 'coefficient_stability'), index=False)
    return tab


# =============================================================================
# 3.4  Selection frequency
# =============================================================================
def _screen_args(call):
    a = call['args']
    kw = {k: a[k] for k in ('continuous', 'ordinal', 'cutpoints', 'groups', 'min_cell', 'seed')}
    kw['features'] = a.get('features')
    return a['X'], a['y'], a['shap_sel'], kw


def _screen_and_budget(X, y, shap_sel, kw, shrinkage, cache):
    rec_pre, info = utils._screen_recommendations(X, y, shap_sel, score_cache=cache, **kw)
    rec = _budgeted(X, y, rec_pre, info['tests'], kw['groups'], shrinkage)
    return rec_pre, info, rec


def _resample(shap_sel, rng):
    """Bootstrap each subcohort; rows get new unique labels, since the rules
    align attributions and feature values by index."""
    out = {}
    for lv, (sh, sel) in shap_sel.items():
        i = rng.integers(0, len(sh), len(sh))
        idx = pd.RangeIndex(len(i))
        out[lv] = (sh.iloc[i].set_axis(idx), sel.iloc[i].set_axis(idx))
    return out


def selection_frequency(call, rec, tag, shrinkage=0.9, n_boot=500, seed=utils.RANDOM_STATE,
                        n_jobs=utils.N_JOBS, out_dir=RESULT_DIR):
    """How often each recommendation is made when the two subcohorts at the
    chosen margin are resampled with replacement.
    """
    from joblib import Parallel, delayed
    X, y, shap_sel, kw = _screen_args(call)
    rec_pre, info = call['result']
    main = _sets(rec_pre, info['tests'], rec)
    seeds = np.random.SeedSequence([seed, 1]).generate_state(n_boot)
    n_workers = (os.cpu_count() or 1) if n_jobs in (-1, None) else n_jobs
    chunks = [c for c in np.array_split(seeds, max(1, n_workers)) if len(c)]

    def run(chunk):
        cache, out = {}, []
        with contextlib.redirect_stdout(io.StringIO()):
            for sd in chunk:
                try:
                    rp, inf, rf = _screen_and_budget(X, y, _resample(shap_sel, np.random.default_rng(sd)),
                                                     kw, shrinkage, cache)
                    out.append(_sets(rp, inf['tests'], rf))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    out.append(None)
        return out

    reps = [r for part in Parallel(n_jobs=len(chunks))(delayed(run)(c) for c in chunks) for r in part]
    done = [r for r in reps if r is not None]
    rows = []
    for kind in main:
        counts = {}
        for r in done:
            for item in r[kind]:
                counts[item] = counts.get(item, 0) + 1
        for item in sorted(set(counts) | set(main[kind])):
            rows.append(dict(kind=kind, item=item, frequency=counts.get(item, 0) / max(len(done), 1),
                             in_main_run=item in main[kind]))
    freq = pd.DataFrame(rows, columns=['kind', 'item', 'frequency', 'in_main_run'])
    freq.to_csv(_out(out_dir, tag, 'selection_frequency'), index=False)
    summary = pd.DataFrame([dict(kind=kind, n_boot=len(reps), n_failed=len(reps) - len(done),
                                 n_main=len(main[kind]),
                                 mean_jaccard_vs_main=float(np.mean([_jaccard(r[kind], main[kind])
                                                                     for r in done])) if done else np.nan,
                                 share_identical_to_main=float(np.mean([set(r[kind]) == set(main[kind])
                                                                        for r in done])) if done else np.nan,
                                 mean_n=float(np.mean([len(r[kind]) for r in done])) if done else np.nan)
                            for kind in main])
    summary.to_csv(_out(out_dir, tag, 'selection_frequency_summary'), index=False)
    return freq, summary


# =============================================================================
# 3.5  Ablation: every subset and every ordering
# =============================================================================
def ablation(rec, X_train, X_test, y_train, y_test, tag, groups=None, rec_sensitivity=None,
             evaluator=None, out_dir=RESULT_DIR):
    """Cox models with every subset of the three recommendation types, each
    compared with the model without recommendations (paired bootstrap, the
    same resamples as the model comparison), and the gain from each type when
    added in each order. Exclusion is always applied first, so only the order
    of the non-linear and interaction terms varies (two orders). With
    rec_sensitivity (the confirmed
    pairs, budget ignored), the lattice is repeated with its interactions.
    """
    evaluator = evaluator or utils.MetricEval(times=utils.evaluation_times(y_train))
    sources = [('recommended', rec)]
    if rec_sensitivity is not None:
        sources.append(('confirmed, budget ignored', rec_sensitivity))
    subsets, orders = [], []
    for source, r in sources:
        scores, cols = {}, {}
        for k in range(len(PARTS) + 1):
            for parts in itertools.combinations(PARTS, k):
                Xtr, Xte = _design(r, X_train, X_test, parts, groups)
                scores[parts] = utils.cox_risk_score(_fit_cox(Xtr, y_train), Xte)
                cols[parts] = tuple(Xtr.columns)
        for parts, s in scores.items():
            c, (lo, hi) = evaluator.boot_metric(y_test, s)
            row = dict(interactions=source, components=' + '.join(parts) or 'none',
                       n_parameters=len(cols[parts]), harrell_c=c, ci_low=lo, ci_high=hi,
                       same_design_as=next((' + '.join(q) or 'none' for q in scores
                                            if q != parts and cols[q] == cols[parts]
                                            and len(q) < len(parts)), ''))
            if parts:
                d, (dlo, dhi), p = evaluator.boot_cindex_diff(y_test, scores[()], s)
                row.update(delta_c_vs_baseline=d, delta_ci_low=dlo, delta_ci_high=dhi, delta_p=p)
            subsets.append(row)
        # exclusion is always applied first; only the later types are permuted
        for rest in itertools.permutations([p for p in PARTS if p != 'exclusion']):
            order = ('exclusion',) + rest
            for step in range(len(order)):
                before = tuple(p for p in PARTS if p in order[:step])
                after = tuple(p for p in PARTS if p in order[:step + 1])
                d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, scores[before], scores[after])
                orders.append(dict(interactions=source, order=' > '.join(order), step=step + 1,
                                   added=order[step], applied=' + '.join(after),
                                   n_parameters=len(cols[after]), delta_c_step=d,
                                   delta_ci_low=lo, delta_ci_high=hi, delta_p=p))
    subsets, orders = pd.DataFrame(subsets), pd.DataFrame(orders)
    subsets.to_csv(_out(out_dir, tag, 'ablation_subsets'), index=False)
    orders.to_csv(_out(out_dir, tag, 'ablation_orderings'), index=False)
    return subsets, orders


# =============================================================================
# 3.6  Attributions of 1 - S(t0)
# =============================================================================
class SurvivalTarget:
    """A survival model whose predict() returns exp(1 - S(t0 | x)).
    """

    def __init__(self, model, t0):
        self.model, self.t0 = model, float(t0)

    def risk(self, X):
        S = self.model.predict_survival_function(X, return_array=True)
        j = int(np.searchsorted(self.model.unique_times_, self.t0, side='right')) - 1
        return 1.0 - (S[:, j] if j >= 0 else np.ones(len(S)))

    def predict(self, X):
        return np.exp(self.risk(X))

    @property
    def n_jobs(self):
        return getattr(self.model, 'n_jobs', None)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


def survival_target(call, rec, ex_model, X_test, y_test, t0, tag, nsamples='auto',
                    shrinkage=0.9, groups=None, evaluator=None, out_dir=RESULT_DIR):
    """Re-run the rules on attributions of 1 - S(t0) instead of the log risk.
    """
    X, y, shap_sel, kw = _screen_args(call)
    rec_pre_main, info_main = call['result']
    target = SurvivalTarget(ex_model, t0)
    new = {}
    for lv, (sh, sel) in shap_sel.items():
        print(f'[survival target/{lv}] SHAP of 1 - S({t0:g}) for {len(sel)} patients', flush=True)
        s2, sel2 = utils.get_explanations(target, X, y, eps=None, risk_level=lv, nsamples=nsamples,
                                          groups=kw['groups'], selected_index=sel.index, seed=kw['seed'])
        s2.to_csv(_out(out_dir, tag, f'shap_values_surv_{lv}'))
        new[lv] = (s2, sel2)
    rec_pre, info, rec_s = _screen_and_budget(X, y, new, kw, shrinkage, {})
    main, alt = _sets(rec_pre_main, info_main['tests'], rec), _sets(rec_pre, info['tests'], rec_s)
    comp = pd.DataFrame([dict(kind=k, main='; '.join(main[k]), survival_target='; '.join(alt[k]),
                              n_main=len(main[k]), n_survival_target=len(alt[k]),
                              jaccard=_jaccard(main[k], alt[k])) for k in main])
    comp.to_csv(_out(out_dir, tag, 'survival_target_recommendations'), index=False)

    evaluator = evaluator or utils.MetricEval(times=utils.evaluation_times(y))
    rows, scores = [], {}
    for name, r in (('Cox_org', None), ('Cox_all', rec), ('Cox_all_survival_target', rec_s)):
        Xtr, Xte = (X, X_test) if r is None else _design(r, X, X_test, groups=groups)
        scores[name] = utils.cox_risk_score(_fit_cox(Xtr, y), Xte)
        c, (lo, hi) = evaluator.boot_metric(y_test, scores[name])
        rows.append(dict(model=name, n_parameters=Xtr.shape[1], harrell_c=c, ci_low=lo, ci_high=hi))
    for row in rows[1:]:
        for ref in ('Cox_org', 'Cox_all'):
            if ref != row['model']:
                d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, scores[ref], scores[row['model']])
                row.update({f'delta_c_vs_{ref}': d, f'delta_vs_{ref}_ci_low': lo,
                            f'delta_vs_{ref}_ci_high': hi, f'delta_vs_{ref}_p': p})
    models = pd.DataFrame(rows)
    models.to_csv(_out(out_dir, tag, 'survival_target_models'), index=False)
    return comp, models


# =============================================================================
# Everything, as the demos call it
# =============================================================================
def run_extras(rec, info, ex_model, X_train, X_test, y_train, y_test, *, t0, tag, groups=None,
               plot_dir='plots/', out_dir=RESULT_DIR, n_boot_coef=500, n_boot_select=500,
               run_survival_target=True):
    """All additional analyses for one data set. info is the second value
    returned by public_analyses.run_recommend.
    """
    groups = dict(groups or {})
    call, settings = info['screen_call'], info['settings']
    shrinkage = settings['shrinkage']
    evaluator = utils.MetricEval(times=utils.evaluation_times(y_train))

    rec_s = interaction_sensitivity(rec, info['tests'], X_train, X_test, y_train, y_test, t0, tag,
                                    groups, plot_dir, out_dir, evaluator)

    fitted, designs = {}, {}
    for name, r, parts in (('Cox_org', rec, ()), ('Cox_exclu', rec, ('exclusion',)),
                           ('Cox_nonlinear', rec, ('nonlinear',)),
                           ('Cox_inter', rec, ('interaction',)), ('Cox_all', rec, PARTS),
                           ('Cox_all_confirmed', rec_s, PARTS)):
        if r is None:
            continue
        Xtr, Xte = _design(r, X_train, X_test, parts, groups)
        fitted[name] = (_fit_cox(Xtr, y_train), Xte)
        designs[name] = Xtr
    saved = pd.read_csv(os.path.join(out_dir, f'{tag}_final_cox_coefficients.csv'), index_col=0)
    if not np.allclose(saved['coef'], fitted['Cox_all'][0].coef_, rtol=1e-8, atol=1e-10):
        raise RuntimeError('refitted final model does not match the saved coefficients')

    calibration_intercept(dict(rsf=(ex_model, X_test), **fitted), y_test, t0, tag, out_dir)
    ph_tests({k: designs[k] for k in ('Cox_org', 'Cox_all', 'Cox_all_confirmed') if k in designs},
             y_train, tag, plot_dir, out_dir)
    sub_index = {lv: sel.index for lv, (_, sel) in call['args']['shap_sel'].items()}
    events_per_parameter(designs, y_train, y_test, sub_index, info['summary']['riley_p_max'],
                         tag, out_dir)
    coefficient_stability(designs['Cox_all'], y_train, tag, n_boot=n_boot_coef, out_dir=out_dir)
    print(f'[selection frequency/{tag}] {n_boot_select} resamples of the subcohorts', flush=True)
    selection_frequency(call, rec, tag, shrinkage, n_boot=n_boot_select, out_dir=out_dir)
    ablation(rec, X_train, X_test, y_train, y_test, tag, groups, rec_s, evaluator, out_dir)
    if run_survival_target:
        survival_target(call, rec, ex_model, X_test, y_test, t0, tag, settings['nsamples'],
                        shrinkage, groups, evaluator, out_dir)
    record = dict(t0=t0, seed=utils.RANDOM_STATE, n_boot_coefficients=n_boot_coef,
                  n_boot_selection=n_boot_select, shrinkage=shrinkage,
                  survival_target=run_survival_target,
                  interaction_sensitivity_pairs=_pair_names(rec_s['interaction']) if rec_s else [],
                  ph=_defaults(utils.ph_assumption_report, skip=('out',)),
                  calibration_intercept='Crowson et al. 2016: log(O/E) up to t0, SE 1/sqrt(O)')
    with open(_out(out_dir, tag, 'extras_settings', 'json'), 'w') as fh:
        json.dump(_jsonable(record), fh, indent=2, default=str)
