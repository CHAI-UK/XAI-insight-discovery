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


def test_screen_skips_self_pairs_and_excluded_features():
    """C08: a feature is never its own partner, and excluded features are not
    interaction candidates."""
    rng = np.random.default_rng(3)
    n = 400
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
    shap_sel = {'low': (shap_df, X), 'high': (shap_df, X)}
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
