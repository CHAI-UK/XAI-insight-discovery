"""Regression tests. Run with:  pytest test_recommender.py -q
"""
import numpy as np
import pandas as pd
import pytest
from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils import (MetricEval, ShapleyAnalysis, strata_generate, nonlinear_analysis,
                   interaction_analysis, check_interaction_spec, load_config, get_model)


@pytest.fixture(scope='module')
def gbsg2_fit():
    X, y = load_gbsg2()
    X = OneHotEncoder().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=20)
    return CoxPHSurvivalAnalysis().fit(Xtr, ytr), Xte, yte


# ---- C5: bootstrap must actually resample -----------------------------------
def test_bootstrap_interval_has_width(gbsg2_fit):
    model, Xte, yte = gbsg2_fit
    obs, (lo, hi), _, boots = MetricEval(300).cal_metric_CI(model, Xte, yte, 'c-index',
                                                            return_values=True)
    assert hi > lo, "interval collapsed onto the point estimate"
    assert len(np.unique(np.round(boots, 6))) > 1
    assert lo <= obs <= hi


# ---- M3: strata partition the input and y is subset ------------------------
def test_strata_partition_and_alignment():
    df = pd.DataFrame({'age': [60, 65, 65, 70, 80]})
    X = df.copy()
    y = np.array([(1, 1.), (0, 2.), (1, 3.), (0, 4.), (1, 5.)],
                 dtype=[('event', '?'), ('time', '<f8')])
    (X1, X2), (y1, y2), n_tie = strata_generate(df, 'age', X, y, 65)
    assert len(X1) + len(X2) == len(X)
    assert len(X1) == len(y1) and len(X2) == len(y2)
    assert n_tie == 2
    assert set(X1['age']) == {60, 65} and set(X2['age']) == {70, 80}


# ---- C6b: test frame centred on the TRAINING mean ---------------------------
def test_nonlinear_uses_training_centre():
    tr = pd.DataFrame({'age': [10., 20., 30.]})
    te = pd.DataFrame({'age': [100., 200.]})
    tr_out, centre = nonlinear_analysis(tr, ['age'])
    te_out, _ = nonlinear_analysis(te, ['age'], centre=centre)
    assert centre == {'age': 20.0}
    assert np.allclose(te_out['age'], [80., 180.])           # not centred on 150
    assert np.allclose(te_out['age_quadratic'], [6400., 32400.])
    with pytest.raises(KeyError):                             # never falls back to test means
        nonlinear_analysis(te.assign(bmi=1.), ['age', 'bmi'], centre=centre)


def test_interaction_uses_training_centre():
    tr = pd.DataFrame({'a': [0., 2.], 'b': [1., 3.]})
    te = pd.DataFrame({'a': [5.], 'b': [10.]})
    tr_out, centre = interaction_analysis(tr, 'a', ['b'], [], verbose=False)
    te_out, _ = interaction_analysis(te, 'a', ['b'], [], centre=centre, verbose=False)
    assert centre == {'a': 1.0, 'b': 2.0}
    assert np.allclose(tr_out['a_x_b'], [1., 1.])
    assert np.allclose(te_out['a_x_b'], [(5 - 1) * (10 - 2)])   # not centred on the test row
    assert np.allclose(te_out[['a', 'b']], te[['a', 'b']])      # main effects untouched
    with pytest.raises(KeyError):
        interaction_analysis(te.assign(c=0.), 'a', ['c'], [], centre=centre, verbose=False)


# ---- M5: exclusion rule has no dead branch and SD matches Methods -----------
def test_exclusion_rule_upper_bound_only():
    class _S(ShapleyAnalysis):
        def bootstrap_analysis(self, x, b=None, stats_type='mean_abs'):
            m = np.mean(np.abs(x)); return m * 0.9, m * 1.1
    shapana = _S(0.05, 0.05, 0.05, random_state=0)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'big': rng.normal(0, 1, 500), 'tiny': rng.normal(0, 1e-4, 500)})
    res = shapana.inclu_exclu_var(df, verbose=False)
    assert res.set_index('feature').loc['tiny', 'recommend_exclude']
    assert not res.set_index('feature').loc['big', 'recommend_exclude']
    # the across-features SD is the SD of two numbers, not of 1,000 numbers
    res_matrix = shapana.inclu_exclu_var(df, sd_definition='whole_matrix', verbose=False)
    assert res['threshold'].iloc[0] != res_matrix['threshold'].iloc[0]


# ---- C4: interaction spec must match the design matrix ---------------------
def test_missing_comma_would_now_be_caught():
    cols = ['ivdrug', 'cd4', 'priorzdv', 'raceth', 'age', 'karnof', 'sex', 'strat2']
    bad = ['ivdrug', 'cd4', 'priorzdv' 'raceth', 'age']      # the published typo
    lists = [['karnof'], ['karnof', 'raceth'], ['ivdrug', 'priorzdv'],
             ['ivdrug', 'priorzdv', 'sex'], ['strat2', 'priorzdv', 'karnof', 'raceth']]
    with pytest.raises(AssertionError):
        check_interaction_spec(bad, lists, cols)
    good = ['ivdrug', 'cd4', 'priorzdv', 'raceth', 'age']
    check_interaction_spec(good, lists, cols)


def test_interaction_analysis_raises_on_absent_column():
    df = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    with pytest.raises(KeyError):
        interaction_analysis(df, 'a', ['b', 'zzz'], [])
    out, _ = interaction_analysis(df, 'a', ['b'], [])
    assert list(out.columns) == ['a', 'b', 'a_x_b']


def test_interaction_analysis_logs_constructed_terms(capsys):
    df = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.], 'b_quadratic': [9., 16.]})
    interaction_analysis(df, 'a', ['b'], ['b'])
    assert "Interaction terms for a: ['a_x_b', 'a_x_b_quad']" in capsys.readouterr().out
    interaction_analysis(df, 'a', ['b'], ['b'], verbose=False)
    assert capsys.readouterr().out == ''


# ---- C07: one config, values as stated in the paper -------------------------
@pytest.mark.parametrize('dataset', ['gbsg2', 'act'])
def test_config_matches_paper(dataset):
    cfg = load_config(dataset)
    assert cfg['test_size'] == 0.20
    assert cfg['n_bootstrap'] == 1000
    assert cfg['rsf']['n_estimators'] == 1000
    assert cfg['exclusion_threshold'] == 0.05
    assert cfg['nonlinear_r_threshold'] == 0.1
    assert cfg['interaction_p_threshold'] == 0.05


def test_unconfirmed_settings_are_refused():
    with pytest.raises(ValueError, match='not yet confirmed'):
        load_config('dataloch')


def test_forest_needs_config_hyperparameters():
    with pytest.raises(ValueError):
        get_model('rf', 20)
    rsf = get_model('rf', 20, **load_config('gbsg2')['rsf'])
    assert (rsf.n_estimators, rsf.min_samples_split, rsf.min_samples_leaf) == (1000, 10, 15)
