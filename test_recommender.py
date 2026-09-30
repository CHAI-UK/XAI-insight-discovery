"""Regression tests. Run with:  pytest test_recommender.py -q
"""
import numpy as np
import pandas as pd
import pytest
from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from recommender import Recommender
from utils import (MetricEval, get_explanations, cox_risk_score, as_surv, _window_similarity,
                   _coxnet_cv, _screen_recommendations)


@pytest.fixture(scope='module')
def gbsg2_fit():
    X, y = load_gbsg2()
    X = OneHotEncoder().fit_transform(X)
    y = as_surv(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=20)
    return CoxPHSurvivalAnalysis().fit(Xtr, ytr), Xte, yte


def test_bootstrap_interval_has_width(gbsg2_fit):
    model, Xte, yte = gbsg2_fit
    obs, (lo, hi) = MetricEval(300).boot_metric(yte, cox_risk_score(model, Xte))
    assert hi > lo, "interval collapsed onto the point estimate"
    assert lo <= obs <= hi


def test_cindex_difference_of_identical_scores(gbsg2_fit):
    model, Xte, yte = gbsg2_fit
    s = cox_risk_score(model, Xte)
    d, (lo, hi), p = MetricEval(200).boot_cindex_diff(yte, s, s)
    assert d == 0 and lo == 0 and hi == 0


class _EmptyReport:
    """Stands in for MetricEval and CalibrationPerform when only the fit matters."""
    def full_report(self, *args, **kwargs):
        return {}

    def report(self, *args, **kwargs):
        return {}


def test_coxnet_penalty_uses_training_data_only(gbsg2_fit):
    _, X, y = gbsg2_fit
    P = np.c_[X[['age', 'pnodes']].to_numpy(float), np.ones(len(X))]
    names = ['age', 'pnodes', 'constant']
    Ptr, Pte = P[:100], P[100:]
    original = Ptr.copy()
    rep = _EmptyReport()
    first = _coxnet_cv(Ptr, Pte, names, y[:100], y[100:], rep, rep, 'test', n_folds=3)
    # changing the test design and outcomes must not change the chosen penalty
    second = _coxnet_cv(Ptr, Pte * 2, names, y[:100], y[100:][::-1], rep, rep, 'test', n_folds=3)
    assert first['alpha'] == second['alpha']
    np.testing.assert_array_equal(Ptr, original)
    assert first['n_parameters'] == 2                    # the constant column is dropped


class _MultiplicativeRisk:
    """Risk score exp(x'b): additive in log risk, multiplicative on the raw scale."""
    def __init__(self, beta):
        self.beta = np.asarray(beta, dtype=float)

    def predict(self, X):
        return np.exp(np.asarray(X, dtype=float) @ self.beta)


def test_attributions_are_additive_on_the_log_scale():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=['a', 'b', 'c'])
    beta = np.array([1.0, -0.5, 0.25])
    event = rng.random(300) < 0.4
    y = np.array(list(zip(event, rng.uniform(1, 10, 300))), dtype=[('event', bool), ('time', float)])
    sh, sel = get_explanations(_MultiplicativeRisk(beta), X, y, eps=0.5, risk_level='high')
    ref = X[event].mean().to_numpy()
    expected = (sel.to_numpy() - ref) * beta
    # an additive log-risk model gives attributions that depend on each feature alone
    assert np.allclose(sh.to_numpy(), expected, atol=1e-6)
    assert (sh.index == sel.index).all()


def test_budget_costs_pairs_by_columns():
    rng = np.random.default_rng(1)
    base = pd.DataFrame({'a': rng.random(50), 'b': (rng.random(50) < .5) * 1.,
                         'c': (rng.random(50) < .5) * 1.,
                         'g_lvl1': 0., 'g_lvl2': 0., 'g_lvl3': 0.})
    tests = pd.DataFrame([dict(strat='b', partner='c', effect_vs_scale=2., selected=True, pair='b||c'),
                          dict(strat='a', partner='g', effect_vs_scale=1., selected=True, pair='a||g')])
    spec, log = Recommender.spec_within_budget(tests, base, budget=2)
    assert spec == {'b': ['c']}                       # a x g needs 3 columns and does not fit
    assert log.set_index('pair').loc['a||g', 'reason'] == 'budget exhausted'


def test_margin_stability_rule():
    w = lambda *sets: [dict(sets=dict(exclusion=set(), nonlinear=s, interaction=set())) for s in sets]
    same = _window_similarity(w({'age', 'bmi'}, {'age', 'bmi'}, {'age', 'bmi'}))
    assert min(same.values()) == 1.0                      # identical recommendations agree
    one_off = _window_similarity(w({'a', 'b', 'c', 'd', 'e'}, {'a', 'b', 'c', 'd', 'e'},
                                   {'a', 'b', 'c', 'd', 'e', 'f'}))
    assert one_off['nonlinear'] < 0.9                     # one change in six breaks stability


def test_apply_uses_training_statistics_only():
    """C05: centring comes from the training data, and each test row is
    transformed on its own, so no test-set statistic enters."""
    rng = np.random.default_rng(2)
    def frame(n):
        return pd.DataFrame({'a': rng.normal(1.0, 2.0, n), 'b': (rng.random(n) < .5) * 1.,
                             'g': rng.integers(0, 4, n).astype(float)})
    X_tr, X_te = frame(200), frame(50)
    rec = dict(exclusion=[], nonlinear=['a', 'g'], interaction={'a': ['b']})
    _, te = Recommender.apply(X_tr, X_te, rec)
    m = X_tr['a'].mean()
    np.testing.assert_allclose(te['a'], X_te['a'] - m)
    np.testing.assert_allclose(te['a_quad'], (X_te['a'] - m) ** 2)
    np.testing.assert_allclose(te['a__x__b'], (X_te['a'] - m) * X_te['b'])
    assert {'g_lvl1', 'g_lvl2', 'g_lvl3'} <= set(te.columns) and 'g' not in te.columns
    _, te_part = Recommender.apply(X_tr, X_te.iloc[:10], rec)
    pd.testing.assert_frame_equal(te_part, te.iloc[:10])


def test_strata_ties_go_to_lower_stratum():
    """C06: patients at the cut point go to the lower stratum (<= / >)."""
    rng = np.random.default_rng(4)
    x = np.repeat([0., 1., 2., 3.], 50)
    b = np.tile([0., 1.], 100)
    X = pd.DataFrame({'x': x, 'b': b})
    shap_df = pd.DataFrame({'x': x - x.mean(),
                            'b': b * (1 + 0.5 * (x > 1)) + rng.normal(0, .1, x.size)})
    t = Recommender(min_cell=20, min_level=5).stratified_screen(x, 1.0, shap_df, X, 'x', ['b'])
    assert t['n_group1'].iloc[0] == (x <= 1).sum() == 100
    assert t['n_group2'].iloc[0] == (x > 1).sum() == 100


def _screen_data(seed=3, n=400):
    """Training data and attributions with one interaction (a x d, b x a) and
    one feature with no effect (z), for the screening tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'a': rng.normal(size=n), 'd': rng.normal(size=n),
                      'b': (rng.random(n) < .5) * 1., 'c': (rng.random(n) < .5) * 1.,
                      'z': (rng.random(n) < .5) * 1.})
    noise = lambda s: rng.normal(0, s, n)
    shap_df = pd.DataFrame({'a': X['a'] * (1 + X['d']) + noise(.05),
                            'd': 0.5 * X['d'] + noise(.05),
                            'b': (X['b'] - .5) * (1 + X['a']) + noise(.05),
                            'c': 0.5 * (X['c'] - X['c'].mean()) + noise(.05),
                            'z': noise(1e-5)}, index=X.index)
    y = np.array(list(zip(rng.random(n) < .5, rng.exponential(5, n))),
                 dtype=[('event', bool), ('time', float)])
    return X, y, {'low': (shap_df, X), 'high': (shap_df, X)}


def test_screen_skips_self_pairs_and_excluded_features():
    """C08: a feature is never its own partner, and excluded features are not
    interaction candidates."""
    X, y, shap_sel = _screen_data()
    rec, info = _screen_recommendations(X, y, shap_sel, continuous=['a', 'd'], ordinal=[],
                                        cutpoints={}, groups={}, min_cell=20, seed=20)
    assert rec['exclusion'] == ['z']
    tests = info['tests']
    assert len(tests)
    assert not (tests['strat'] == tests['partner']).any()
    assert 'z' not in set(tests['strat']) | set(tests['partner'])


def test_exclusion_threshold_is_sd_of_feature_means():
    """C12: the threshold is 5% of the SD across features of the mean |SHAP|,
    and only the upper confidence bound is compared with it."""
    rng = np.random.default_rng(5)
    shap_df = pd.DataFrame({'large': rng.normal(0, 1, 300), 'mid': rng.normal(0, .5, 300),
                            'tiny': rng.normal(0, 1e-4, 300)})
    excl, t = Recommender(n_boot=200).exclusion(shap_df)
    assert np.isclose(t['threshold'].iloc[0], 0.05 * shap_df.abs().mean().std(ddof=1))
    assert (t['excluded'] == (t['ci_high'] < t['threshold'])).all()
    assert excl == ['tiny']


def test_captured_screens_are_unchanged_and_tables_reproduce_them():
    """public_analyses: observing the screening calls changes nothing, the
    original function is restored, and the recomputed exclusion and
    non-linearity tables give the recommended lists."""
    import utils
    from public_analyses import capture_screens, screen_tables
    X, y, shap_sel = _screen_data()
    kw = dict(continuous=['a', 'd'], ordinal=[], cutpoints={}, groups={}, min_cell=20, seed=20)
    original = utils._screen_recommendations
    plain, _ = original(X, y, shap_sel, **kw)
    with capture_screens() as calls:
        observed, _ = utils._screen_recommendations(X, y, shap_sel, **kw)
    assert utils._screen_recommendations is original
    assert observed == plain and len(calls) == 1
    tables = screen_tables(calls[0])
    excluded = tables['exclusion'].groupby('feature')['excluded'].all()
    assert sorted(excluded[excluded].index) == plain['exclusion']
    assert set(tables['exclusion']['subcohort']) == {'low', 'high'}


def test_final_cox_table_matches_the_fitted_model(gbsg2_fit, tmp_path):
    """public_analyses: HRs are exp(coef) of the model evaluate_recommendations
    fits, with finite CIs that contain them."""
    from public_analyses import final_cox_table
    _, X, y = gbsg2_fit
    X = X[['age', 'pnodes', 'horTh=yes', 'tsize']]
    rec = dict(exclusion=[], nonlinear=['age'], interaction={'pnodes': ['horTh=yes']})
    Xtr, _ = Recommender.apply(X, X, rec, variant='all')
    model = CoxPHSurvivalAnalysis(alpha=1e-6, ties='efron').fit(Xtr, y)
    pd.Series(model.coef_, index=Xtr.columns, name='coef').to_csv(
        tmp_path / 'toy_final_cox_coefficients.csv')
    _, tab = final_cox_table(rec, X, y, 'toy', out_dir=str(tmp_path))
    np.testing.assert_allclose(tab['HR'], np.exp(model.coef_))
    assert ((tab['HR_low'] < tab['HR']) & (tab['HR'] < tab['HR_high'])).all()
    assert (tmp_path / 'toy_final_cox_hr.tsv').exists()


# ---- public_extras (open data only) ------------------------------------------
def test_global_schoenfeld_reproduces_lifelines(gbsg2_fit):
    """The global test reproduces lifelines per term, and with one term the
    global and per-term statistics coincide."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from public_extras import schoenfeld_tests, _ph_frame
    _, X, y = gbsg2_fit
    for cols in (['tsize', 'age', 'pnodes'], ['pnodes']):      # not in alphabetical order
        d = _ph_frame(X[cols], y)
        f = CoxPHFitter(penalizer=0.01).fit(d, 'time', 'event')
        per, glob = schoenfeld_tests(f, d, 'km')
        ref = proportional_hazard_test(f, d, time_transform='km').summary['test_statistic']
        assert list(per['term']) == cols
        np.testing.assert_allclose(per['test_statistic'], ref.loc[cols], rtol=1e-8)
        assert glob['df'] == len(cols)
    assert np.isclose(glob['test_statistic'], per['test_statistic'].iloc[0])


def test_calibration_in_the_large_is_zero_on_training_data(gbsg2_fit):
    """With Breslow ties and the Breslow baseline hazard, the cumulative
    hazards of the training patients sum to the number of events, so O/E = 1
    when t0 is the last follow-up time."""
    from public_extras import calibration_in_the_large
    _, X, y = gbsg2_fit
    X = X[['age', 'pnodes', 'tsize', 'progrec']]
    m = CoxPHSurvivalAnalysis(alpha=0, ties='breslow').fit(X, y)
    r = calibration_in_the_large(m, X, y, t0=float(y['time'].max()))
    assert r['observed_events'] == int(y['event'].sum())
    assert abs(r['citl']) < 1e-6
    shorter = calibration_in_the_large(m, X, y, t0=float(np.median(y['time'])))
    assert shorter['observed_events'] < r['observed_events']


def test_survival_target_explains_risk_by_t0(gbsg2_fit):
    """log(predict) of the wrapper is 1 - S(t0), and n_jobs passes through."""
    from sksurv.ensemble import RandomSurvivalForest
    from public_extras import SurvivalTarget
    _, X, y = gbsg2_fit
    rsf = RandomSurvivalForest(n_estimators=10, min_samples_leaf=15, random_state=0, n_jobs=2).fit(X, y)
    t0 = float(np.median(y['time']))
    target = SurvivalTarget(rsf, t0)
    expected = 1 - np.array([fn(t0) for fn in rsf.predict_survival_function(X.iloc[:5])])
    np.testing.assert_allclose(np.log(target.predict(X.iloc[:5])), expected)
    assert target.n_jobs == 2 and target.set_params(n_jobs=1).n_jobs == 1


def test_subcohort_resample_keeps_rows_aligned():
    from public_extras import _resample
    X, _, shap_sel = _screen_data(n=100)
    out = _resample(shap_sel, np.random.default_rng(0))
    for lv, (sh, sel) in out.items():
        assert sh.index.is_unique and (sh.index == sel.index).all()
        # each resampled attribution row belongs to the resampled patient
        orig_sh, orig_x = shap_sel[lv]
        match = orig_x.reset_index(drop=True).merge(sel.reset_index(), on=list(sel.columns))
        assert len(match) >= len(sel)


def test_ablation_steps_add_up_to_the_total_gain(gbsg2_fit, tmp_path):
    """Under every ordering the per-step gains in C sum to the gain of the
    model with all recommendations, and the full subset is Cox_all."""
    from public_extras import ablation
    _, X, y = gbsg2_fit
    X = X[['age', 'pnodes', 'horTh=yes', 'tsize', 'progrec']]
    Xtr, Xte, ytr, yte = X.iloc[:120], X.iloc[120:], y[:120], y[120:]
    rec = dict(exclusion=['progrec'], nonlinear=['age'], interaction={'pnodes': ['horTh=yes']})
    subsets, orders = ablation(rec, Xtr, Xte, ytr, yte, 'toy', evaluator=MetricEval(50),
                               out_dir=str(tmp_path))
    assert len(subsets) == 8 and orders['order'].nunique() == 6
    total = subsets.set_index('components').loc['exclusion + nonlinear + interaction',
                                                 'delta_c_vs_baseline']
    np.testing.assert_allclose(orders.groupby('order')['delta_c_step'].sum(), total)


def test_confirmed_pairs_are_those_that_passed_the_cox_check():
    from public_extras import confirmed_pairs
    tests = pd.DataFrame(dict(pair=['a||b', 'a||b', 'c||d', 'e||f'], strat=['a', 'b', 'c', 'e'],
                              partner=['b', 'a', 'd', 'f'], effect_vs_scale=[.3, .5, .2, .9],
                              screen_selected=[True, True, True, False],
                              selected=[True, True, False, False]))
    assert confirmed_pairs(tests) == {'b': ['a']}
    assert confirmed_pairs(tests.drop(columns='screen_selected')) == {}
