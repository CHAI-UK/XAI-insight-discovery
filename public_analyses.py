"""
public_analyses.py: outputs for the open data sets that the shared code
computes but does not save, and a record of the settings each run used.
"""
import contextlib
import inspect
import json
import os
import subprocess
from importlib import metadata

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

import utils
from recommender import Recommender
from utils import write_tsv

RESULT_DIR = utils.RESULT_DIR
PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'scikit-survival', 'shap', 'scipy',
            'matplotlib', 'seaborn', 'patsy', 'statsmodels', 'lifelines', 'joblib')


# =============================================================================
# Recommendations, with the tables recommend() computes but does not return
# =============================================================================
@contextlib.contextmanager
def capture_screens():
    """Record every call of utils._screen_recommendations, which
    choose_margin makes once per margin evaluated.
    """
    calls, original = [], utils._screen_recommendations
    signature = inspect.signature(original)

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(dict(args=signature.bind(*args, **kwargs).arguments, result=result))
        return result

    utils._screen_recommendations = observed
    try:
        yield calls
    finally:
        utils._screen_recommendations = original


def _chosen_call(calls, summary):
    """The screening call made at the margin recommend() chose, identified by
    its subcohort sizes (these grow with the margin)."""
    sizes = (summary['subcohort_low'], summary['subcohort_high'])
    match = [c for c in calls
             if (len(c['args']['shap_sel']['low'][0]), len(c['args']['shap_sel']['high'][0])) == sizes]
    if not match:
        raise RuntimeError(f'no screening call with subcohort sizes {sizes}')
    return match[-1]


def screen_tables(call):
    """Exclusion and non-linearity tables of one screening call.
    """
    a = call['args']
    rec, _ = call['result']
    rg = Recommender(min_cell=a['min_cell'], groups=a['groups'], seed=a['seed'])
    shap_sel = a['shap_sel']
    excl = {lv: rg.exclusion(shap_sel[lv][0]) for lv in shap_sel}
    cand_nl = list(a['continuous']) + list(a['ordinal'])
    nonl = {lv: rg.nonlinear(shap_sel[lv][0], shap_sel[lv][1], cand_nl) for lv in shap_sel}
    if sorted(set(excl['low'][0]) & set(excl['high'][0])) != rec['exclusion']:
        raise RuntimeError('recomputed exclusion tables do not reproduce the recommendations')
    if sorted(set(nonl['low'][0]) | set(nonl['high'][0])) != rec['nonlinear']:
        raise RuntimeError('recomputed non-linearity tables do not reproduce the recommendations')
    tables = {}
    for name, res in (('exclusion', excl), ('nonlinear', nonl)):
        parts = [t.assign(subcohort=lv) for lv, (_, t) in res.items() if len(t)]
        if not parts:
            tables[name] = pd.DataFrame(columns=['subcohort'])
            continue
        tables[name] = pd.concat(parts, ignore_index=True)
        tables[name].insert(0, 'subcohort', tables[name].pop('subcohort'))
    return tables


def run_recommend(*args, tag, out_dir=RESULT_DIR, **kwargs):
    bound = inspect.signature(utils.recommend).bind(*args, tag=tag, **kwargs)
    bound.apply_defaults()
    with capture_screens() as calls:
        rec, info = utils.recommend(*bound.args, **bound.kwargs)
    call = _chosen_call(calls, info['summary'])

    os.makedirs(out_dir, exist_ok=True)
    path = lambda name: os.path.join(out_dir, f'{tag}_{name}.csv')
    for name, tab in screen_tables(call).items():
        tab.to_csv(path(f'{name}_tests'), index=False)
    call['result'][1]['feature_screen'].to_csv(path('feature_screen'), index=False)
    info['budget_log'].to_csv(path('budget_log'), index=False)
    for lv, (shap_df, sel) in call['args']['shap_sel'].items():
        shap_df.to_csv(path(f'shap_values_{lv}'))
        sel.to_csv(path(f'shap_data_{lv}'))

    data_args = ('ex_model', 'X_train', 'y_train')
    info['settings'] = {k: v for k, v in bound.arguments.items() if k not in data_args}
    info['screen_call'] = call
    return rec, info


# =============================================================================
# Record of the settings a run used
# =============================================================================
def _defaults(fn, skip=()):
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty and k not in skip}


def _git_state():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        run = lambda *cmd: subprocess.run(['git', *cmd], cwd=here, capture_output=True,
                                          text=True, check=True).stdout.strip()
        changed = run('status', '--porcelain', '--', 'recommender.py', 'utils.py')
        return dict(commit=run('rev-parse', 'HEAD'), shared_code_modified=bool(changed))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, shared_code_modified=None)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


def write_run_record(tag, *, split, ex_model, t0, info, eval_times, out_dir=RESULT_DIR):
    """Write <tag>_run_settings.json: the settings the run used.
    """
    rsf = {k: v for k, v in ex_model.get_params().items()
           if k in ('n_estimators', 'min_samples_split', 'min_samples_leaf',
                    'max_features', 'max_depth', 'bootstrap', 'random_state', 'low_memory')}
    settings = dict(info['settings'])
    record = dict(
        dataset=tag,
        code=_git_state(),
        packages={p: metadata.version(p) for p in PACKAGES},
        split=split,
        exploratory_model=dict(model='RandomSurvivalForest', **rsf),
        attributions=dict(explainer='KernelSHAP', target='log of the forest risk score',
                          nsamples=settings.pop('nsamples'), seed=settings['seed'],
                          reference='mean feature vector of training patients without '
                                    '(low) or with (high) the event'),
        recommend=settings,
        recommender_defaults=_defaults(Recommender.__init__,
                                       skip=('res_dir', 'min_cell', 'seed', 'groups')),
        target_model_filter=dict(spline_df=_defaults(Recommender.target_model_filter)['df'],
                                 ties='breslow', alpha=1e-6,
                                 note='ties and alpha fixed in the function body'),
        evaluation=dict(
            t0=t0, times=list(eval_times),
            bootstrap=_defaults(utils.MetricEval.__init__, skip=('times',)),
            calibration=dict(_defaults(utils.CalibrationPerform.__init__,
                                       skip=('t0', 'save_folder', 'model_name')),
                             n_bins=_defaults(utils.evaluate_recommendations)['n_bins']),
            uno_tau='90th percentile of test follow-up times',
            cox=dict(alpha=_defaults(utils.fit_and_score)['alpha'], ties='efron'),
            comparators=dict(spline_df=4, lasso_n_alphas=50, lasso_alpha_min_ratio=0.01,
                             cv_folds=_defaults(utils._coxnet_cv)['n_folds'],
                             note='spline df and penalty grid fixed in the function bodies'),
            ph_test=_defaults(utils.ph_assumption_report, skip=('out',))),
        result=dict(margin=info['summary']['margin'],
                    margin_stable=info['summary']['margin_stable'],
                    subcohort_low=info['summary']['subcohort_low'],
                    subcohort_high=info['summary']['subcohort_high']),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{tag}_run_settings.json')
    with open(path, 'w') as fh:
        json.dump(_jsonable(record), fh, indent=2, default=str)
    return path


# =============================================================================
# Hazard ratios of the final model
# =============================================================================
# Copied unchanged from survival-model-toolkit 0.3.0 (core.export_coefficients,
# MIT licence), the package used to report the DataLoch coefficients.
def export_coefficients(model, X_train, y_train, out, penalizer=0.0, tol=1e-2):
    """
    Coefficient table with hazard ratios and confidence intervals for a
    fitted model.
    """
    from lifelines import CoxPHFitter

    terms = list(X_train.columns)
    hr_direct = pd.Series(np.exp(np.asarray(model.coef_).ravel()), index=terms, name="HR")
    model_ties = getattr(model, "ties", "unknown")

    d = X_train.copy()
    d["time"] = np.asarray(y_train["time"], float)
    d["event"] = np.asarray(y_train["event"], int)
    fitter = CoxPHFitter(penalizer=penalizer).fit(d, "time", "event")   # always Efron
    s = fitter.summary.reset_index().rename(columns={"index": "term", "covariate": "term"})
    keep = [c for c in ["term", "se(coef)", "exp(coef)",
                        "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"]
            if c in s.columns]
    tab = s[keep].rename(columns={"exp(coef)": "HR_lifelines_refit_efron",
                                  "exp(coef) lower 95%": "HR_low",
                                  "exp(coef) upper 95%": "HR_high"})
    tab.insert(1, "HR", tab["term"].map(hr_direct))  # point estimate from the fitted model itself

    max_diff = float(np.nanmax(np.abs(np.log(tab["HR"])
                                      - np.log(tab["HR_lifelines_refit_efron"])))) if len(tab) else 0.0
    if max_diff > tol:
        print(f"[export_coefficients] WARNING: max |log HR| difference between the "
             f"fitted model (ties='{model_ties}') and the lifelines Efron refit is "
             f"{max_diff:.3f} (tol={tol}). If model_ties != 'efron', this is expected; "
             f"refit the scikit-survival model with ties='efron' if the two must match exactly.")
    else:
        print(f"[export_coefficients] refit consistent with the fitted model "
             f"(max |log HR| diff = {max_diff:.4f})")

    write_tsv(tab, out)
    return fitter, tab


def final_cox_table(rec, X_train, y_train, tag, groups=None, out_dir=RESULT_DIR):
    """Hazard ratios with 95% CIs for the Cox model with all recommendations,
    """
    Xtr, _ = Recommender.apply(X_train, X_train, rec, variant='all', groups=groups)
    model = CoxPHSurvivalAnalysis(alpha=_defaults(utils.fit_and_score)['alpha'],
                                  ties='efron').fit(Xtr, y_train)
    saved = pd.read_csv(os.path.join(out_dir, f'{tag}_final_cox_coefficients.csv'), index_col=0)
    if list(saved.index) != list(Xtr.columns) or not np.allclose(saved['coef'], model.coef_,
                                                                  rtol=1e-8, atol=1e-10):
        raise RuntimeError('refitted final model does not match the saved coefficients')
    return export_coefficients(model, Xtr, y_train, os.path.join(out_dir, f'{tag}_final_cox_hr.tsv'))
