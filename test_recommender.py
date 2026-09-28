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
                   interaction_analysis, check_interaction_spec, load_config, get_model,
                   cox_coefficient_table, adjust_p_bh, save_recommendations,
                   recommended_exclusions, screen_skips, recommended_nonlinear,
                   recommended_interactions)


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
    lists = [['karnof'], ['karnof', 'raceth'], ['ivdrug'],
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


# ---- C16: recommendations and coefficients as tables ------------------------
def test_screens_return_full_tables():
    shapana = ShapleyAnalysis(0.05, None, 0.05, random_state=0)
    x = np.linspace(-1, 1, 201)                                  # symmetric: corr(x, x^2) = 0
    shap_df = pd.DataFrame({'lin': 2 * x, 'u': x ** 2 - 1})
    res = shapana.non_linear_test(shap_df, pd.DataFrame({'lin': x, 'u': x}))
    assert list(res['feature']) == ['lin', 'u']                  # every feature, not only flagged
    assert list(res['recommend_nonlinear']) == [False, True]

    df1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [0., 0., 0.]})
    df2 = pd.DataFrame({'a': [4., 5., 6., 7.], 'b': [0., 0., 0., 0.]})
    res = shapana.wilcoxon_rank_sum_test(df1, df2).set_index('feature')
    assert res.loc['a', 'rank_biserial'] == -1.0                   # every df1 value below df2
    assert (res.loc['a', 'n_stratum1'], res.loc['a', 'n_stratum2']) == (3, 4)
    assert len(res) == 2


def test_bh_adjustment_keeps_untested_as_nan():
    adj = adjust_p_bh([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(adj[1])
    assert np.allclose(adj[[0, 2, 3]], [0.03, 0.04, 0.04])


def test_save_recommendations_writes_tables(tmp_path):
    exc = [pd.DataFrame({'feature': ['a'], 'recommend_exclude': [True]}).assign(cohort='low', sd_definition='across_features')]
    nl = [pd.DataFrame({'feature': ['a'], 'r': [0.05]}).assign(cohort='low')]
    inter = [pd.DataFrame({'feature': ['a', 'b'], 'p_value': [np.nan, 0.01], 'recommended': [False, True],
                           'skipped_reason': ['stratifying variable', '']})
             .assign(cohort='high', stratifying_variable='a', split_on='SHAP value', threshold=0)]
    tables = save_recommendations({'dataset': 'demo'}, exc, nl, inter, out_dir=tmp_path)
    assert {f.name for f in tmp_path.iterdir()} == {'demo_exclusion.csv', 'demo_nonlinearity.csv', 'demo_interactions.csv'}
    t = tables['interactions']
    assert list(t.columns[:4]) == ['cohort', 'stratifying_variable', 'split_on', 'threshold']
    assert np.isnan(t['p_adj_bh'][0]) and np.isclose(t['p_adj_bh'][1], 0.01)


def _breslow_loglik(beta, X, time, event):
    eta = X @ beta
    risk = np.array([np.log(np.sum(np.exp(eta[time >= t]))) for t in time])
    return np.sum(event * (eta - risk))


def test_cox_table_matches_numerical_information():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=['a', 'b', 'c'])
    time = np.ceil(rng.exponential(np.exp(-(X.to_numpy() @ [0.6, -0.4, 0.0]))) * 4)   # tied times
    event = rng.random(150) < 0.8
    y = np.array(list(zip(event, time)), dtype=[('event', '?'), ('time', '<f8')])
    model = CoxPHSurvivalAnalysis().fit(X, y)
    tab = cox_coefficient_table(model, X, y)

    b, h, Xa = model.coef_, 1e-4, X.to_numpy()
    H = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (_breslow_loglik(b + ei + ej, Xa, time, event) - _breslow_loglik(b + ei - ej, Xa, time, event)
                       - _breslow_loglik(b - ei + ej, Xa, time, event) + _breslow_loglik(b - ei - ej, Xa, time, event)) / (4 * h * h)
    se_numeric = np.sqrt(np.diag(np.linalg.inv(-H)))
    assert np.allclose(tab['se'], se_numeric, rtol=1e-4)
    assert np.allclose(tab['hr'], np.exp(model.coef_))
    assert (tab['hr_ci_lo'] < tab['hr']).all() and (tab['hr'] < tab['hr_ci_hi']).all()


# ---- C08: no self-pairs, no excluded features in the screen or the model ----
def test_screen_skips_self_and_excluded():
    shapana = ShapleyAnalysis(0.05, None, 0.05, random_state=0)
    df1 = pd.DataFrame({'age': [1., 2., 3.], 'sex': [0., 1., 2.], 'drug': [5., 6., 7.]})
    df2 = pd.DataFrame({'age': [4., 5., 6.], 'sex': [3., 4., 5.], 'drug': [8., 9., 10.]})
    res = shapana.wilcoxon_rank_sum_test(df1, df2, skip=screen_skips('age', ['drug'])).set_index('feature')
    assert res.loc['age', 'skipped_reason'] == 'stratifying variable' and np.isnan(res.loc['age', 'p_value'])
    assert res.loc['drug', 'skipped_reason'] == 'excluded feature' and np.isnan(res.loc['drug', 'p_value'])
    assert res.loc['sex', 'skipped_reason'] == '' and not np.isnan(res.loc['sex', 'p_value'])


def test_recommended_exclusions_needs_every_cohort():
    tab = lambda cohort, flags: pd.DataFrame({'feature': ['a', 'b'], 'recommend_exclude': flags}).assign(
        cohort=cohort, sd_definition='across_features')
    assert recommended_exclusions([tab('low', [True, True]), tab('high', [True, False])]) == ['a']


def test_spec_rejects_self_pairs_and_excluded():
    cols = ['a', 'b', 'c']
    with pytest.raises(AssertionError, match='own interaction partner'):
        check_interaction_spec(['a'], [['a', 'b']], cols)
    with pytest.raises(AssertionError, match='recommended for exclusion'):
        check_interaction_spec(['a'], [['b', 'c']], cols, excluded=['c'])
    check_interaction_spec(['a'], [['b']], cols, excluded=['c'])


def test_quick_mode_is_separate_from_full():
    full, quick = load_config('act'), load_config('act', quick=True)
    assert quick['rsf']['n_estimators'] < full['rsf']['n_estimators']
    assert quick['rsf']['min_samples_leaf'] == full['rsf']['min_samples_leaf']   # nested keys merged
    assert quick['subgroup_margin'] == full['subgroup_margin']                   # dataset values kept
    assert quick['results_dir'] != full['results_dir'] and quick['plots_dir'] != full['plots_dir']
    assert quick['quick'] and not full['quick']


# ---- Model specification generated from the screen output (R1.6(c)) --------
def test_recommended_nonlinear_skips_binary_and_excluded():
    tab = lambda cohort, flags: pd.DataFrame({'feature': ['age', 'sex', 'drug', 'bmi'],
                                              'recommend_nonlinear': flags}).assign(cohort=cohort)
    X = pd.DataFrame({'age': [50., 60., 70.], 'sex': [0., 1., 0.], 'drug': [1., 2., 3.], 'bmi': [20., 25., 30.]})
    tables = [tab('low', [True, True, True, False]), tab('high', [False, False, False, True])]
    assert recommended_nonlinear(tables, X, excluded=['drug']) == ['age', 'bmi']   # flagged in any cohort


def test_recommended_interactions_dedupes_unordered_pairs():
    t = pd.DataFrame({'stratifying_variable': ['age', 'age', 'sex', 'sex', 'bmi'],
                      'feature':              ['sex', 'bmi', 'age', 'bmi', 'age'],
                      'recommended':          [True,  False, True,  True,  True]})
    feats, lists = recommended_interactions(t)
    assert (feats, lists) == (['age', 'sex', 'bmi'], [['sex'], ['bmi'], ['age']])
    pairs = [frozenset((a, b)) for a, l in zip(feats, lists) for b in l]
    assert len(pairs) == len(set(pairs)) == 3
