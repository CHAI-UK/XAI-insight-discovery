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
from utils import MetricEval, get_explanations, cox_risk_score, as_surv, _window_similarity


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


def test_coxnet_penalty_uses_training_data_only(gbsg2_fit):
    from utils import fit_coxnet_design

    class EmptyReport:
        def full_report(self, *args, **kwargs):
            return {}

        def report(self, *args, **kwargs):
            return {}

    _, X, y = gbsg2_fit
    train = X.iloc[:100][['age', 'pnodes']].copy()
    test = X.iloc[100:][['age', 'pnodes']].copy()
    train['constant'], test['constant'] = 1.0, 1.0
    original = train.copy()
    reporter = EmptyReport()
    first, score, coef = fit_coxnet_design(
        train, test, y[:100], y[100:], reporter, reporter, 'test', n_folds=3)
    second, _, coef2 = fit_coxnet_design(
        train, test * 2, y[:100], y[100:][::-1], reporter, reporter, 'test', n_folds=3)
    assert first['alpha'] == second['alpha']
    pd.testing.assert_series_equal(coef, coef2)
    pd.testing.assert_frame_equal(train, original)
    assert 'constant' not in coef.index
    # Returned coefficients preserve risk contrasts on the input feature scale.
    expected = test[coef.index].to_numpy() @ coef.to_numpy()
    np.testing.assert_allclose(score - score[0], expected - expected[0], atol=1e-10)


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


def test_interaction_screen_type_one_error():
    res = Recommender().null_sim(R=200)
    original = res.loc[res['rule'].str.startswith('original'), 'type_I_error'].iloc[0]
    contrast = res.loc[res['rule'].str.startswith('within-stratum'), 'type_I_error'].iloc[0]
    assert original > 0.9, "the original rule should fire under the additive null"
    assert contrast < 0.1


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
