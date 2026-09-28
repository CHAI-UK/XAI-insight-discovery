import os
import json
import platform
from importlib.metadata import version

import yaml
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample

from scipy.stats import gaussian_kde, binomtest, mannwhitneyu, pearsonr, norm, false_discovery_control

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator

from patsy import dmatrix


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')

def _null_settings(d, prefix=''):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += _null_settings(v, f'{prefix}{k}.')
        elif v is None:
            out.append(prefix + k)
    return out

def load_config(dataset, quick=False, path=CONFIG_PATH):
    """
    Settings for `dataset`: the 'common' block of config.yml overlaid with the
    dataset's own block and, with quick=True, the 'quick' debugging block.
    Raises if any setting is still null (unconfirmed).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if dataset not in raw:
        raise KeyError(f"no '{dataset}' section in {path}")

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw['common'].items()}
    for block in [raw[dataset]] + ([raw['quick']] if quick else []):
        for k, v in block.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    unconfirmed = _null_settings(cfg)
    if unconfirmed:
        raise ValueError(f"'{dataset}' settings not yet confirmed in {path}: {unconfirmed}")
    cfg['dataset'] = dataset
    cfg['quick'] = quick
    if quick:
        print(f"QUICK MODE: debugging settings, results in {cfg['results_dir']}/ - do not report them")
    return cfg

def save_run_config(cfg, out_dir=None):
    # Record the settings and package versions this run used
    out_dir = out_dir or cfg['results_dir']
    record = dict(cfg,
                  python=platform.python_version(),
                  packages={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn',
                                                    'scikit-survival', 'shap', 'scipy']})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg['dataset']}_run_config.json")
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    return path

def get_model(name, seed, **rsf_params):
    """
    rsf_params are the RSF hyperparameters from config.yml, e.g.
    get_model('rf', cfg['seed'], **cfg['rsf']). They are required for 'rf' so
    the forest never falls back silently to library defaults.
    """
    if name == 'rf':
        if not rsf_params:
            raise ValueError("pass the RSF hyperparameters from config.yml: get_model('rf', cfg['seed'], **cfg['rsf'])")
        model = RandomSurvivalForest(**rsf_params, n_jobs=-1, random_state=seed)
    elif name == 'cox':
        model = CoxPHSurvivalAnalysis()
    else:
        raise ValueError(f"Unrecognised model name '{name}'!")

    return model

## Generate shap analysis results and shap plots for the Recommendar
def _extract_time_event(y):
    """
    Return (time, event) as 1D numpy arrays.

    Supports:
        * sksurv structured arrays from load_gbsg2, load_aids, ...
        * pandas DataFrame with usual column names.
    """

    # Case 1: structured array (what sksurv datasets return)
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = list(y.dtype.names)
        if len(names) != 2:
            raise ValueError(f"Structured array y must have exactly 2 fields, got {names}")

        # By sksurv convention: first field = event indicator (bool),
        # second field = time (float). :contentReference[oaicite:1]{index=1}
        event = y[names[0]].astype(bool)
        time = y[names[1]].astype(float)

        return time, event

    # Case 2: pandas DataFrame
    if isinstance(y, pd.DataFrame):
        # possible event column names across datasets
        event_candidates = [
            "event", "Status", "status", "death", "fstat",
            "cens", "censor", "censor_d"
        ]

        time_candidates = [
            "time", "Survival_in_days", "lenfol", "time_d", "futime"
        ]

        event_col = next((c for c in event_candidates if c in y.columns), None)
        time_col = next((c for c in time_candidates if c in y.columns), None)

        if event_col is None or time_col is None:
            raise ValueError(
                "Could not infer event/time columns in DataFrame y; "
                "got columns: " + ", ".join(y.columns)
            )

        event = y[event_col].astype(bool).to_numpy()
        time = y[time_col].astype(float).to_numpy()

        return time, event

    raise TypeError("Unsupported type for y. Use a sksurv structured array or a pandas DataFrame.")

def get_explanations(model, X_test, y_test, eps, risk_level='low', max_rows=0, seed=None):
    # max_rows > 0 explains a random subset of at most max_rows patients (quick mode)
    names = list(y_test.dtype.names)
    if risk_level == 'high':
        risk_data = X_test[y_test[names[0]] == 1]
    elif risk_level == 'low':
        risk_data = X_test[y_test[names[0]] == 0]
    else:
        risk_data = X_test

    # compute the reference point (average patient)
    X_mean = risk_data.mean().to_frame().T
    # find average risk
    y_mean = model.predict(X_mean)

    if (risk_level == 'high') | (risk_level == 'low'):
        y_std = np.std(model.predict(risk_data))
        # predict risk for all
        y_pred = model.predict(X_test)

        eps = y_std * eps

        # select only those individuals whose change in risk with respect to the reference point is smaller than epsilon (i.e. close to 0)
        sel_mask = np.abs(y_pred - y_mean) < eps

        # use the reference point
        sel_data = X_test[sel_mask]

    else:
        sel_data = X_test

    if max_rows and len(sel_data) > max_rows:
        sel_data = sel_data.sample(n=max_rows, random_state=seed)

    ex = shap.KernelExplainer(model.predict, X_mean)
    explanation = ex(sel_data)
    df_shap = pd.DataFrame(explanation.values, columns=X_test.columns)

    # get SHAP values of the selected individuals (delta_R close to 0)
    return df_shap, sel_data

def make_plot(df_shap, org_df, filename, plot_type='scatter', xlabel='SHAP value', figsize=(4, 3), folder='plots'):
    features = df_shap.columns.to_list()
    n_feats = len(features)

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'violin':
        if org_df is not None:
            shap_long = df_shap.melt(var_name='Feature', value_name='SHAP Value')
            feat_long = org_df.melt(var_name='Feature', value_name='Feature Value')
            data = pd.concat([shap_long, feat_long['Feature Value']], axis=1)
        else:
            raise ValueError('No original faeture values provided')

        for i, feature in enumerate(features):
            df_feat = data[data['Feature'] == feature]

            try:
                kde = gaussian_kde(df_feat['SHAP Value'].values)
                x_vals = df_feat['SHAP Value'].values
                density_vals = kde(x_vals)

                density_vals = density_vals/density_vals.max()*0.4

                y_center = i
                y_vals = np.random.uniform(low=i-density_vals, high=i+density_vals)

            except np.linalg.LinAlgError:
                print(f'KDE failed for feature {feature}, using scatter fallback')
                x_vals = df_feat['SHAP Value'].values
                y_vals = np.full_like(df_feat['SHAP Value'], fill_value=i, dtype=float)

            val = df_feat['Feature Value'].values

            if (val.max() - val.min()) == 0.0:
                val_norm = np.full_like(val, fill_value=val.max(), dtype=float)
            else:
                val_norm = (val - val.min()) / (val.max() - val.min())

            sc = ax.scatter(
                x_vals, y_vals,
                c=val_norm,
                cmap='coolwarm', edgecolors='None', s=20, alpha=0.8
            )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Feature Value')
        cbar.set_ticks([])
        cbar.ax.text(1.05, 0.05, 'Low', transform=cbar.ax.transAxes, va='center')
        cbar.ax.text(1.05, 0.95, 'High', transform=cbar.ax.transAxes, va='center')
    elif plot_type == 'bar':
        for i, feature in enumerate(features):
            x = df_shap[feature].values
            y = np.full_like(x, fill_value=i, dtype=float)
            plt.barh(y, x, align='center')

    ax.axvline(0.0, c='k', linewidth=0.5)
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(features)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Feature')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f'{filename}.png'), dpi=1000)
    # save only, so the demo scripts never block on a plot window
    plt.close(fig)

class ShapleyAnalysis:
  # Tools used for generating recommendations
  def __init__(self, var_threshold, prop_thresh, pval_thresh, random_state, r_thresh=0.1, n_boot=1000):
    self.var_thresh = var_threshold
    self.prop_thresh = prop_thresh
    self.pval_thresh = pval_thresh
    self.random_state = random_state
    self.r_thresh = r_thresh
    self.n_boot = n_boot

  def inclu_exclu_var(self, df, sd_definition='across_features', verbose=True):
    """
      Generate recommendations for feature exclusion

      A feature is recommended for exclusion when the upper bound of the
      bootstrap CI of its mean absolute attribution is <= var_thresh * SD.

      Args:
        df: shap analysis values
        sd_definition: 'across_features' (default, as described in Methods) takes
          the SD across the per-feature mean absolute attributions;
          'whole_matrix' reproduces the originally published code (SD over
          every entry of |df|) for use as a sensitivity analysis.
        verbose: print the features recommended for exclusion

      Returns:
        DataFrame with one row per feature (mean_abs_attr, ci_lo, ci_hi,
        threshold, recommend_exclude)
    """
    abs_df = df.abs()
    if sd_definition == 'across_features':
      eps = np.std(abs_df.mean(axis=0).to_numpy(dtype=float), ddof=1)
    elif sd_definition == 'whole_matrix':
      eps = np.std(abs_df.to_numpy(dtype=float), ddof=1)
    else:
      raise ValueError(f"Unknown sd_definition '{sd_definition}'")
    threshold = self.var_thresh * eps

    rows = []
    for col in df.columns:
      vals = df[col].values
      ci_lo, ci_hi = self.bootstrap_analysis(vals, b=self.n_boot, stats_type='mean_abs')
      # the mean of absolute values is non-negative, so only the upper bound matters
      rows.append(dict(feature=col,
                       mean_abs_attr=float(np.mean(np.abs(vals))),
                       ci_lo=ci_lo, ci_hi=ci_hi,
                       threshold=threshold,
                       recommend_exclude=bool(ci_hi <= threshold)))
      if verbose and rows[-1]['recommend_exclude']:
        print(col)
    return pd.DataFrame(rows)

  def non_linear_test(self, shap_file, sel_data):
    """
      Generate recommendations for non-linearity

      Args:
        shap_file: shap analysis values
        sel_data: feature values

      Returns:
        DataFrame with one row per feature (r, p_value, recommend_nonlinear);
        a feature is flagged when |r| between its values and its SHAP values
        is below r_thresh. Flagged features are also printed.
    """
    rows = []
    for c in shap_file.columns:
      corr = pearsonr(shap_file[c].to_numpy(), sel_data[c].to_numpy())
      flag = bool(np.abs(corr[0]) < self.r_thresh)
      rows.append(dict(feature=c, n=len(shap_file), r=float(corr[0]), p_value=float(corr[1]),
                       recommend_nonlinear=flag))
      if flag:
        print(f'{c}: {corr[0]:.2f} ({corr[1]:.2f})')
    return pd.DataFrame(rows)

  def wilcoxon_rank_sum_test(self, df1, df2, skip=None):
    """
      Genrate recommendations for feature interactions

      Args:
        df1, df2: shap analysis files for two populations after stratification.
        skip: {feature: reason} for features that must not be tested, e.g.
          from screen_skips(); they are kept in the output, untested, with
          the reason in skipped_reason.

      Returns:
        DataFrame with one row per feature: stratum sizes, mean SHAP per
        stratum, Mann-Whitney U (for df1), rank-biserial correlation
        (P(df1 > df2) - P(df1 < df2)), raw two-sided P and whether it passes
        pval_thresh. Features passing are also printed.
    """
    skip = skip or {}
    rows = []
    for col in df1.columns:
        val1 = df1[col].to_numpy()
        val2 = df2[col].to_numpy()
        n1, n2 = len(val1), len(val2)

        row = dict(feature=col, n_stratum1=n1, n_stratum2=n2,
                   mean_shap_stratum1=float(np.mean(val1)) if n1 else np.nan,
                   mean_shap_stratum2=float(np.mean(val2)) if n2 else np.nan,
                   u_statistic=np.nan, rank_biserial=np.nan, p_value=np.nan, recommended=False,
                   skipped_reason=skip.get(col, ''))
        if col in skip:
            print(f'  not tested: {col} ({skip[col]})')
        elif n1 and n2:
            stat, p = mannwhitneyu(val1, val2, method='asymptotic', alternative='two-sided')
            row.update(u_statistic=float(stat), rank_biserial=float(2 * stat / (n1 * n2) - 1),
                       p_value=float(p), recommended=bool(p <= self.pval_thresh))
            if row['recommended']:
                print(col)
        rows.append(row)
    return pd.DataFrame(rows)

  def bootstrap_analysis(self, x, b=1000, alpha=0.05, stats_type='median_abs'):
    """
      Bootstrap analysis for performance

      Args:
        x: dataset that will be evaluated using bootstrap analysis.
        b: bootstrap iterations
        alpha: confidence level

      Returns:
        low and high confidence intervals
    """
    rng = np.random.default_rng(self.random_state)
    # x = x.to_numpy()
    n = x.size
    boot = []
    for _ in range(b):
        xb = resample(
            x,
            replace=True,
            n_samples=n,
            random_state=int(rng.integers(1_000_000_000)),
        )
        boot.append(self.get_data_stats(xb, stats_type))
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return lo, hi


  @staticmethod
  def get_data_stats(x, stats_type='mean_abs'):
      if stats_type == 'mean_abs':
          data = np.mean(np.abs(x), axis=0)
      elif stats_type == 'median_abs':
          data = np.median(np.abs(x), axis=0)
      elif stats_type == 'mean':
          data = np.mean(x, axis=0)
      elif stats_type == 'median':
          data = np.median(x, axis=0)
      else:
          raise ValueError('Unknown stats')
      return data

def sign_balance_test(df):
  # Generate strata
  result = []
  for col in df.columns:
    vals = df[col].values

    n = len(vals[vals != 0])
    if n < 0.1 * len(vals):
      continue
    n_pos = (vals > 0).sum()
    test = binomtest(n_pos, n, 0.5)
    balanced = (test.pvalue > 0.05)
    if balanced:
      print(col)

def strata_generate(data, variable, X_test, y_test, thresh):
  """
    Split X_test (and y_test) into two strata on data[variable], using
    (x <= thresh) / (x > thresh) so no observation is dropped.

    Returns:
      X_test_list, y_test_list, n_at_threshold (number of observations
      exactly on the threshold)
  """
  vals = np.asarray(data[variable], dtype=float)
  mask1 = vals <= thresh
  mask2 = vals > thresh
  assert mask1.sum() + mask2.sum() == len(vals), "strata do not partition the input"

  X_test_list = [X_test[mask1], X_test[mask2]]
  y_test_list = [y_test[mask1], y_test[mask2]]
  assert all(len(X_g) == len(y_g) for X_g, y_g in zip(X_test_list, y_test_list)), \
    "X/y misaligned within a stratum"

  n_at_threshold = int(np.sum(vals == thresh))
  return X_test_list, y_test_list, n_at_threshold

def stratify_shap_analysis(model, X_test_list, y_test_list, variable, risk_level, margin=None, max_rows=0, seed=None):
    # margin (config subgroup_margin) is only used when risk_level is 'low' or 'high'
    num_group = len(X_test_list)
    strata_shap = []
    for i in range(num_group):
        X_test = X_test_list[i]
        y_test = y_test_list[i]
        shap_file, sel_data = get_explanations(model, X_test, y_test, eps=margin, risk_level=risk_level,
                                              max_rows=max_rows, seed=seed)
        # make_plot(shap_file, sel_data, f'aids_shap_{i}_low', plot_type='violin', xlabel='SHAP value', figsize=(8, 6))
        strata_shap.append(shap_file)
    return strata_shap

## Integrate feature exclusion recommendations on the dataset
def exclusion_analysis(data, exclu_feature):
  data = data.drop(columns=exclu_feature, errors='ignore')
  return data

## Integrate non-linearity recommendations on the dataset
def nonlinear_analysis(data, nonlinear_feature, nonlinear_type='quadratic', centre=None):
    """
    Add non-linear terms.

    Call on the training frame with centre=None to learn the centring values,
    then pass the returned dict as `centre` when transforming the test frame,
    so the test data are centred on the training means.

    Returns:
        (transformed_data, centre_dict)
    """
    data = data.copy()
    learned = {} if centre is None else dict(centre)
    nonlinear_cols = []
    for feature in nonlinear_feature:
        if feature not in data.columns:
            continue
        name = f'{feature}_{nonlinear_type}'
        if nonlinear_type == 'quadratic':
            if feature not in learned:
                if centre is not None:
                    raise KeyError(f"no training centre supplied for '{feature}'")
                learned[feature] = float(data[feature].mean())
            data[feature] = data[feature] - learned[feature]
            nonlinear_cols.append((data[feature] ** 2).rename(name).to_frame())
        else:
            # TODO: spline knots should also come from the training data (M4/M7)
            basis = dmatrix(f"0+cr({feature}, df=3)", data,
                            return_type="dataframe", NA_action='raise')
            nonlinear_cols.append(basis.add_prefix(name).reindex(data.index))
    if nonlinear_cols:
        data = pd.concat([data] + nonlinear_cols, axis=1)
    return data, learned

## Integrate interaction recommendations on the dataset
def interaction_analysis(data, inter_feat, interact_list, nonlinear_list, centre=None, strict=True, verbose=True):
    """
    Add product terms between inter_feat and each feature in interact_list.

    Each factor of a product is centred before multiplying. Call on the
    training frame with centre=None to learn the centring values, then pass
    the returned dict as `centre` when transforming the test frame, so the
    test products are centred on the training means. The main-effect columns
    themselves are left unchanged.

    With strict=True, a requested column that does not exist raises a KeyError
    instead of being skipped silently. With verbose=True, the constructed
    terms are printed.

    Returns:
        (transformed_data, centre_dict)
    """
    missing = [c for c in [inter_feat] + list(interact_list) if c not in data.columns]
    if missing and strict:
        raise KeyError(f"interaction requested on absent column(s): {missing}")

    learned = {} if centre is None else dict(centre)
    def centred(col):
        if col not in learned:
            if centre is not None:
                raise KeyError(f"no training centre supplied for '{col}'")
            learned[col] = float(data[col].mean())
        return data[col] - learned[col]

    inter_cols = {}
    a = inter_feat
    for b in interact_list:
        if a in data.columns and b in data.columns:
            inter_cols[f'{a}_x_{b}'] = centred(a) * centred(b)
            if a in nonlinear_list and b in nonlinear_list:
                inter_cols[f'{a}_quad_x_{b}_quad'] = centred(a + '_quadratic') * centred(b + '_quadratic')
            elif a in nonlinear_list:
                inter_cols[f'{a}_quad_x_{b}'] = centred(a + '_quadratic') * centred(b)
            elif b in nonlinear_list:
                inter_cols[f'{a}_x_{b}_quad'] = centred(a) * centred(b + '_quadratic')

    if verbose:
        print(f'Interaction terms for {inter_feat}: {list(inter_cols)}')

    inter = pd.DataFrame(inter_cols, index=data.index)
    return pd.concat([data, inter], axis=1), learned

def recommended_exclusions(exclusion_tables, sd_definition='across_features'):
    """
    Features recommended for exclusion: flagged in every cohort (low and high
    risk) under the given SD definition, from the inclu_exclu_var tables
    (with cohort and sd_definition columns added).
    """
    exc = pd.concat(exclusion_tables, ignore_index=True)
    exc = exc[exc['sd_definition'] == sd_definition]
    flagged = exc.groupby('feature')['recommend_exclude'].all()
    return sorted(flagged[flagged].index)

def recommended_nonlinear(nonlinear_tables, X, excluded=()):
    """
    Features flagged as non-linear in any cohort, for nonlinear_analysis.
    Excluded features get no squared term, and neither do binary features:
    the centred square of a two-valued feature is a linear function of it and
    would duplicate the main effect.
    """
    nl = pd.concat(nonlinear_tables, ignore_index=True)
    out = []
    for f in dict.fromkeys(nl.loc[nl['recommend_nonlinear'], 'feature']):
        if f in excluded:
            print(f'  no squared term for {f} (excluded feature)')
        elif X[f].nunique() <= 2:
            print(f'  no squared term for {f} (binary feature)')
        else:
            out.append(f)
    return out

def recommended_interactions(interaction_table):
    """
    Pairs recommended by the interaction screen, as (strat_features,
    interaction_lists) for check_interaction_spec / interaction_analysis.
    Each unordered pair appears once, under the first stratifying variable
    (in screen order) that recommended it.
    """
    rec = interaction_table[interaction_table['recommended']]
    seen, spec = set(), {}
    for a, b in zip(rec['stratifying_variable'], rec['feature']):
        pair = frozenset((a, b))
        if a != b and pair not in seen:
            seen.add(pair)
            spec.setdefault(a, []).append(b)
    return list(spec), list(spec.values())

def save_model_spec(cfg, excluded, nonlinear, strat_features, interaction_lists, out_dir=None):
    # Record the model specification generated from the screen output
    out_dir = out_dir or cfg['results_dir']
    os.makedirs(out_dir, exist_ok=True)
    spec = dict(excluded=list(excluded), squared_terms=list(nonlinear),
                interactions={a: list(l) for a, l in zip(strat_features, interaction_lists)},
                n_interaction_pairs=sum(len(l) for l in interaction_lists))
    path = os.path.join(out_dir, f"{cfg['dataset']}_model_spec.json")
    with open(path, 'w') as f:
        json.dump(spec, f, indent=2)
    print('Model specification:', spec)
    return spec

def screen_skips(variable, excluded):
    # Features not tested when stratifying on `variable`
    skip = {f: 'excluded feature' for f in excluded}
    skip[variable] = 'stratifying variable'
    return skip

def check_interaction_spec(strat_features, interaction_lists, columns, excluded=()):
    # Call before fitting any interaction model
    assert len(strat_features) == len(interaction_lists), \
        f"{len(strat_features)} stratifying features vs {len(interaction_lists)} interaction lists"
    for a, lst in zip(strat_features, interaction_lists):
        assert a not in lst, f"'{a}' is listed as its own interaction partner"
        for b in [a] + list(lst):
            assert b in columns, f"'{b}' is not a column in the design matrix"
            assert b not in excluded, f"'{b}' was recommended for exclusion but appears in an interaction"

class CalibrationPerform:
  def __init__(self, t0, n_bins=5, kind='survival', n_boot=0,
                random_state=0, save_folder=None, model_name=['RandomForest']):
      self.t0 = t0
      self.n_bins = n_bins
      self.kind = kind
      self.n_boot = n_boot
      self.random_state = random_state
      self.save_folder = save_folder
      self.model_name = model_name

  def _extract_time_event(self, y):
    """
    Return (time, event) as 1D numpy arrays.

    Supports:
        * sksurv structured arrays from load_gbsg2, load_aids, ...
        * pandas DataFrame with usual column names.
    """

    # Case 1: structured array (what sksurv datasets return)
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = list(y.dtype.names)
        if len(names) != 2:
            raise ValueError(f"Structured array y must have exactly 2 fields, got {names}")

        # By sksurv convention: first field = event indicator (bool),
        # second field = time (float). :contentReference[oaicite:1]{index=1}
        event = y[names[0]].astype(bool)
        time = y[names[1]].astype(float)

        return time, event

    # Case 2: pandas DataFrame
    if isinstance(y, pd.DataFrame):
        # possible event column names across datasets
        event_candidates = [
            "event", "Status", "status", "death", "fstat",
            "cens", "censor", "censor_d"
        ]

        time_candidates = [
            "time", "Survival_in_days", "lenfol", "time_d", "futime"
        ]

        event_col = next((c for c in event_candidates if c in y.columns), None)
        time_col = next((c for c in time_candidates if c in y.columns), None)

        if event_col is None or time_col is None:
            raise ValueError(
                "Could not infer event/time columns in DataFrame y; "
                "got columns: " + ", ".join(y.columns)
            )

        event = y[event_col].astype(bool).to_numpy()
        time = y[time_col].astype(float).to_numpy()

        return time, event

    raise TypeError("Unsupported type for y. Use a sksurv structured array or a pandas DataFrame.")

  def plot_survival_calibration(self, X, y, model):
    rng = np.random.default_rng(self.random_state)
    est = self.extract_survival_estimator(model)

    surv_funcs = est.predict_survival_function(X, return_array=False)
    pred_surv_t0 = np.array([float(sf(self.t0)) for sf in surv_funcs])

    if self.kind == 'survival':
        pred_cal = pred_surv_t0
    elif self.kind == 'risk':
        pred_cal = 1.0 - pred_surv_t0
    else:
        raise ValueError("kind must be 'survival' or 'risk'")

    edges = self.quantile_bins(pred_cal)
    bin_idx = np.digitize(pred_cal, edges[1:-1], right=True)
    n_bins_eff = edges.size - 1

    time, event = self._extract_time_event(y)

    bin_pred, bin_obs, bin_counts = [], [], []
    obs_ci_lo, obs_ci_hi = [], []
    MIN_N = 10

    for b in range(n_bins_eff):
        mask = bin_idx == b
        if mask.sum() < MIN_N:
            continue

        # compute KM *first*
        s_hat = self.km_s_at(time[mask], event[mask])
        if np.isnan(s_hat):
            # if KM at t0 is undefined in this bin, skip the bin entirely
            continue

        # only now append pred & obs so lengths always match
        bin_pred.append(pred_cal[mask].mean())

        obs_val = s_hat if self.kind == 'survival' else (1.0 - s_hat)
        bin_obs.append(obs_val)
        bin_counts.append(mask.sum())

        if self.n_boot > 0:
            boot_vals = []
            for _ in range(self.n_boot):
                idx = rng.integers(0, mask.sum(), mask.sum())
                t_b = time[mask][idx]
                e_b = event[mask][idx]
                s_b = self.km_s_at(t_b, e_b)
                if np.isnan(s_b):
                    # extremely unlikely if original s_hat wasn't NaN,
                    # but be safe and skip this resample
                    continue
                boot_vals.append(
                    s_b if self.kind == 'survival' else (1.0 - s_b)
                )
            if len(boot_vals) > 0:
                lo, hi = np.percentile(boot_vals, [2.5, 97.5])
                obs_ci_lo.append(lo)
                obs_ci_hi.append(hi)

    bin_pred = np.asarray(bin_pred)
    bin_obs = np.asarray(bin_obs)
    bin_counts = np.asarray(bin_counts)
    bin_obs_ci = (
        (np.asarray(obs_ci_lo), np.asarray(obs_ci_hi))
        if self.n_boot > 0 and len(obs_ci_lo) > 0
        else None
    )

    return {
        "bin_pred": bin_pred,
        "bin_obs": bin_obs,
        "bin_counts": bin_counts,
        "bin_obs_ci": bin_obs_ci,
        "edges": edges,
        "t0": self.t0,
        "kind": self.kind,
    }

  def calib_estimate(self, model, X, y):
      est_res = self.plot_survival_calibration(X=X, y=y, model=model)
      bin_pred, bin_obs, bin_n = est_res['bin_pred'], est_res['bin_obs'], est_res['bin_counts']

      # coef = np.polyfit(bin_pred, bin_obs, 1)
      # slope_ols, intercept_ols = coef[0], coef[1]
      w = np.asarray(bin_n) if bin_n is not None else np.ones_like(bin_pred)
      X = np.c_[np.ones_like(bin_pred), bin_pred]

      # W = np.diag(w)
      beta = np.linalg.inv(X.T @ (w[:, None] * X)) @ (X.T @ (w * bin_obs))
      intercept, slope = beta[0], beta[1]

      print(f'The intercept is {intercept}')
      print(f'The slope is {slope}')

  def calib_plot(self, model_lst, data_lst, ax=None, title=None):
      ### Generate calibration plots
      if ax is None:
          fig, ax = plt.subplots(figsize=(5, 5))
      for i, mdl in enumerate(model_lst):
          X_i, y_i = data_lst[i][0], data_lst[i][1]
          est_res = self.plot_survival_calibration(X=X_i, y=y_i, model=mdl)
          bin_pred, bin_obs, bin_obs_ci = est_res['bin_pred'], est_res['bin_obs'], est_res['bin_obs_ci']
          ax.plot(bin_pred, bin_obs, marker='o', linestyle='-',
                  label=f'Observed (KM)_{self.model_name[i]}')

          if self.kind == 'survival':
              y_label = 'Observed survival'
              x_label = 'Predicted survival'
          elif self.kind == 'risk':
              y_label = 'Observed risk'
              x_label = 'Predicted risk'
          else:
              raise ValueError('Kind must be "survival" or "risk".')

          minv = min(0.0, bin_pred.min(), bin_obs.min())
          maxv = max(1.0, bin_pred.max(), bin_obs.max())
          ax.plot([minv, maxv], [minv, maxv],
                  color='black', linestyle='--', linewidth=1.2,
                  label='Perfect calibration')
          ax.set_xlim(minv, maxv)
          ax.set_ylim(minv, maxv)
          # ax.set_xlim(0.5, 1.0)
          # ax.set_ylim(0.5, 1.0)
          ax.set_xlabel(x_label)
          ax.set_ylabel(y_label)
          if title is None:
              title = f'Calibration plot ({self.kind})'
          ax.set_title(title)
          ax.grid(True, linestyle='--', alpha=0.4)
          ax.legend()

          if bin_obs_ci is not None:
              lo, hi = bin_obs_ci
              ax.vlines(bin_pred, lo, hi, alpha=0.6)
              # ax.set_xlim(0.5, 1.0)
              # ax.set_ylim(0.5, 1.0)
          if self.save_folder is not None:
              os.makedirs(self.save_folder, exist_ok=True)
              plt.savefig(
                  os.path.join(self.save_folder,
                              f'Calibration_plot_{self.model_name}.pdf'),
                  dpi=1000,
                  bbox_inches='tight'
              )

  def quantile_bins(self, x):
      qs = np.linspace(0, 1, self.n_bins + 1)
      edges = np.quantile(x, qs)
      edges = np.unique(edges)

      if edges.size < 2:
          edges = np.array([x.min(), x.max()])
      return edges

  def km_s_at(self, time, event):
    # Generate Kaplan–Meier estimator
    if time.size == 0:
        return np.nan
    t, s = kaplan_meier_estimator(event, time)
    if self.t0 <= t.min():
        return 1.0
    if self.t0 > t.max():
        return np.nan
    eps = 1e-12
    return float(np.interp(self.t0 - eps, t, s))

  @staticmethod
  def extract_survival_estimator(model):
      if hasattr(model, "predict_survival_function"):
          return model
      if hasattr(model, "named_steps"):
          for _, step in model.named_steps.items():
              if hasattr(step, "predict_survival_function"):
                  return step
      raise AttributeError(
          "Could not find an estimator with predict_survival_function in `model`."
      )

def cox_risk_score(model, x):
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    if hasattr(model, 'predict'):
        return model.predict(x)
    if hasattr(model, 'predict_partial_hazard'):
        return model.predict_partial_hazard(x).ravel()
    raise AttributeError('Cox model has no usable scoring method')

class MetricEval:
    def __init__(self, num_of_b, seed=1):
        self.num_of_b = num_of_b
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def boot_cindex_diff(self, y_test, s1, s2):
        ### Generate bootstrap performance difference between two models
        event = y_test['event'].astype(bool)
        time = y_test['time'].astype(float)
        n = len(time)

        d_obs = self.cindex(event, time, s2) - self.cindex(event, time, s1)

        diffs = np.empty(self.num_of_b)
        for b in range(self.num_of_b):
            idx = self.rng.integers(0, n, size=n)
            diffs[b] = self.cindex(event[idx], time[idx], s2[idx]) - \
                       self.cindex(event[idx], time[idx], s1[idx])

        lo, hi = np.percentile(diffs, [2.5, 97.5])
        upper_tail = np.mean(diffs >= 0)
        lower_tail = np.mean(diffs <= 0)
        p_two = 2 * min(upper_tail, lower_tail)
        p_two = min(1.0, p_two)

        return d_obs, (lo, hi), p_two

    def boot_matric(self, y_test, s, metric_type):
        ### Generate bootstrap performance
        boot_vals = np.empty(self.num_of_b)
        if metric_type == 'c-index':
            event = y_test['event'].astype(bool)
            time = y_test['time'].astype(float)
            n = len(time)
            print(n)
            val_obs = self.cindex(event, time, s)

            for b in range(self.num_of_b):
                idx = self.rng.integers(0, n, size=n)
                if metric_type == 'c-index':
                    boot_vals[b] = self.cindex(event[idx], time[idx], s[idx])
            lo, hi = np.percentile(boot_vals, [2.5, 97.5])
            upper_tail = np.mean(boot_vals >= val_obs)
            lower_tail = np.mean(boot_vals <= val_obs)
            p_two = 2 * min(upper_tail, lower_tail)
            p_two = min(1.0, p_two)

        return val_obs, (lo, hi), p_two

    def cal_metric_CI(self, model, data, data_y, metric, return_values=False):
        """
        Percentile bootstrap CI for a model's C-index on a held-out set.

        Risk scores are predicted once; each replicate resamples the scores and
        the outcome with the same indices.
        """
        if metric != 'c-index':
            raise ValueError(f"metric '{metric}' not supported")

        scores = np.asarray(model.predict(data), dtype=float)
        time, event = _extract_time_event(data_y)
        n = len(time)
        if not (len(scores) == len(event) == n):
            raise ValueError("X, y and predictions have inconsistent lengths")

        val_obs = self.cindex(event, time, scores)

        boot_vals = np.empty(self.num_of_b)
        for b in range(self.num_of_b):
            idx = self.rng.integers(0, n, size=n)
            # a resample with no events cannot be scored
            if event[idx].sum() == 0:
                boot_vals[b] = np.nan
                continue
            boot_vals[b] = self.cindex(event[idx], time[idx], scores[idx])

        boot_vals = boot_vals[~np.isnan(boot_vals)]
        lo, hi = np.percentile(boot_vals, [2.5, 97.5])
        upper_tail = np.mean(boot_vals >= val_obs)
        lower_tail = np.mean(boot_vals <= val_obs)
        p_two = min(1.0, 2 * min(upper_tail, lower_tail))

        print(f'{metric} = {val_obs:.3f} (95% CI {lo:.3f}-{hi:.3f}), p = {p_two:.4f}')
        if return_values:
            return val_obs, (lo, hi), p_two, boot_vals

    @staticmethod
    def cindex(event, time, s):
        c, *_ = concordance_index_censored(event, time, s)
        return c

def cox_coefficient_table(model, X, y, alpha=0.05):
    """
    Coefficient table for a fitted, unpenalised sksurv CoxPHSurvivalAnalysis:
    coefficient, SE, hazard ratio with Wald (1 - alpha) CI, z and P.

    SEs come from the observed information of the Breslow partial likelihood
    (the likelihood sksurv maximises), evaluated at the fitted coefficients
    and on the data the model was fitted on.
    """
    if getattr(model, 'alpha', 0) != 0 or getattr(model, 'ties', 'breslow') != 'breslow':
        raise NotImplementedError('SEs are only valid for an unpenalised model with Breslow ties')

    time, event = _extract_time_event(y)
    Xa = np.asarray(X, dtype=float)
    beta = np.asarray(model.coef_, dtype=float)

    order = np.argsort(-time, kind='stable')          # latest time first
    t, d, Xs = time[order], event[order], Xa[order]
    eta = Xs @ beta
    w = np.exp(eta - eta.max())                       # the scale cancels below

    # risk-set sums for t_j >= t_i, taken at the end of each tie group
    last = np.searchsorted(-t, -t, side='right') - 1
    S0 = np.cumsum(w)[last]
    S1 = np.cumsum(w[:, None] * Xs, axis=0)[last]

    # I = sum_j w_j c_j x_j x_j' - sum_i d_i S1_i S1_i' / S0_i^2,
    # with c_j = sum over events i at or before t_j of 1 / S0_i
    inv_S0 = np.where(d, 1.0 / S0, 0.0)
    c = np.cumsum(inv_S0[::-1])[::-1]
    c = c[np.searchsorted(-t, -t, side='left')]       # include every event tied with t_j
    info = (Xs * (w * c)[:, None]).T @ Xs - (S1 * (d / S0 ** 2)[:, None]).T @ S1

    se = np.sqrt(np.diag(np.linalg.inv(info)))
    z = beta / se
    q = norm.ppf(1 - alpha / 2)
    return pd.DataFrame({'term': list(X.columns), 'coef': beta, 'se': se,
                         'hr': np.exp(beta),
                         'hr_ci_lo': np.exp(beta - q * se), 'hr_ci_hi': np.exp(beta + q * se),
                         'z': z, 'p_value': 2 * norm.sf(np.abs(z))})

def adjust_p_bh(p):
    # Benjamini-Hochberg adjusted P values; NaN (untested) entries stay NaN
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = false_discovery_control(p[ok], method='bh')
    return out

def save_recommendations(cfg, exclusion, nonlinearity, interactions, out_dir=None):
    """
    Write the recommender output as machine-readable tables:
      <dataset>_exclusion.csv     every feature, per cohort and SD definition
      <dataset>_nonlinearity.csv  every feature, per cohort
      <dataset>_interactions.csv  every candidate in the interaction screen, with its
                                  stratifying variable and cut point, BH-adjusted P
                                  across all tests run in this dataset's screen, and
                                  the reason for any candidate not tested
    Each argument is a list of DataFrames, as returned by the ShapleyAnalysis methods
    with identifying columns (cohort, ...) already added.
    """
    out_dir = out_dir or cfg['results_dir']
    os.makedirs(out_dir, exist_ok=True)
    inter = pd.concat(interactions, ignore_index=True)
    inter.insert(inter.columns.get_loc('p_value') + 1, 'p_adj_bh', adjust_p_bh(inter['p_value']))

    tested = inter[inter['p_value'].notna()]
    pairs = {frozenset(p) for p in zip(tested['stratifying_variable'], tested['feature'])}
    print(f"Interaction screen: {len(tested)} tests on {len(pairs)} unique pairs; "
          f"{(inter['skipped_reason'] != '').sum()} candidates not tested "
          f"({inter.loc[inter['skipped_reason'] != '', 'skipped_reason'].value_counts().to_dict()})")

    tables = {'exclusion': pd.concat(exclusion, ignore_index=True),
              'nonlinearity': pd.concat(nonlinearity, ignore_index=True),
              'interactions': inter}
    id_cols = ['cohort', 'sd_definition', 'stratifying_variable', 'split_on', 'threshold']
    for name, df in tables.items():
        df = df[[c for c in id_cols if c in df] + [c for c in df if c not in id_cols]]
        df.to_csv(os.path.join(out_dir, f"{cfg['dataset']}_{name}.csv"), index=False)
        tables[name] = df
    return tables
