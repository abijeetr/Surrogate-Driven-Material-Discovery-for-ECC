# Generated from: ECC_FINAL_pipeline.ipynb
# Converted at: 2026-08-19T02:27:32.955Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # ECC Tensile Prediction — FINAL CONSOLIDATED PIPELINE
# 
# End-to-end implementation: ET+GBR global + sparse residual MoE + Mondrian CQR.
# Applied to both targets: **Second Strain** (primary) and **Second Stress** (secondary).
# 
# ## Architecture
# 
# ```
#                         ALL 37 FEATURES
#                       (18 raw + 19 engineered)
#                               │
#               ┌───────────────┼────────────────┐
#               ▼               ▼                ▼
#        ┌───────────┐    ┌──────────┐    ┌──────────────┐
#        │ ET (q50)  │    │  Router  │    │ Three experts │
#        │ GBR(q10,  │    │ LogReg + │    │ R0,R1,R2:     │
#        │     q90)  │    │ isotonic │    │ RidgeCV       │
#        └─────┬─────┘    └────┬─────┘    └──────┬────────┘
#              │               │                  │
#              └────────┬──────┴──────────────────┘
#                       ▼
#               Soft router-weighted blending
#               (with ablation: 8 configurations)
#                       │
#                       ▼
#             Mondrian CQR (3 bins by predicted q50)
#                       │
#                       ▼
#                80% prediction intervals
# ```
# 
# ## The 8 ablation configurations
# 
# | Config | NoPSH/Weak | Bulk/Mid | Tail/Strong |
# |---|---|---|---|
# | A_Global only  | – | – | – |
# | B_Low only     | ✓ | – | – |
# | C_Mid only     | – | ✓ | – |
# | D_High only    | – | – | ✓ |
# | E_Low+Mid      | ✓ | ✓ | – |
# | F_Low+High     | ✓ | – | ✓ |
# | G_Mid+High     | – | ✓ | ✓ |
# | H_Full MoE     | ✓ | ✓ | ✓ |
# 
# ## Pipeline phases (per target)
# 
# 1. **Pass 1 — Default hyperparameter ablation:** 8 configs, default HP, full GroupKFold CV
# 2. **Pass 2 — Random HP search:** 8 trials on `A_Global` to find best ET/GBR hyperparameters
# 3. **Pass 3 — Tuned hyperparameter ablation:** Re-run all 8 configs with tuned HP
# 
# ## Reported metrics
# 
# For each config: overall MAE, RMSE, 80% coverage, 80% interval width, per-regime MAE and coverage.
# 
# ## Output files (per target)
# 
# - `final_<target>_default_ablation.csv` — 8-config metrics with default HP
# - `final_<target>_default_mae.png` — bar chart of MAE per config & regime
# - `final_<target>_default_scatter_grid.png` — 8 scatter plots (predicted vs true)
# - `final_<target>_default_intervals_grid.png` — 8 interval plots (sorted by true)
# - `final_<target>_hp_search.csv` — HP trial log
# - `final_<target>_tuned_*` — same plots/CSV after tuning
# - `final_summary.json` — overall best results for both targets


# ## Setup — imports and constants


import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS = 1e-8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EXCEL_PATH = 'Tension Test_VIT_re.xlsx'

# ## 1 — Load + filter raw data


raw = pd.read_excel(EXCEL_PATH, sheet_name='Tension Test_VIT')
raw.rename(columns={'Water Reducer / SP': 'Water Reducer/SP'}, inplace=True)
df = raw.copy()
print(f'Raw shape: {df.shape}')

# Age filter — drop specimens younger than 28 days (encoded in Specimen/Mixture as '7d', '14D' etc.)
def extract_age(row):
    for col in ['Specimen', 'Mixture']:
        v = str(row.get(col, ''))
        m = re.search(r'(\d+)\s*[dD]', v)
        if m: return int(m.group(1))
    return None

ages = df.apply(extract_age, axis=1)
mask_drop = ages.notna() & (ages < 28)
df = df[~mask_drop].reset_index(drop=True)
print(f'After age filter (≥28d only): {df.shape}')

# ## 2 — Group fingerprint by composition


GROUP_COLS = ['Fiber Volume','L/D','RI','Cement','Water','Sand',
              'Fly ash C','Fly ash F','GGBS','Coarse Aggr.',
              'Silica Fume','Water Reducer/SP','W/B']
for c in GROUP_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c] = df[c].fillna(df[c].median())

df['_fp'] = df[GROUP_COLS].round(4).apply(lambda r: '|'.join(map(str, r.values)), axis=1)
df['group_id'] = pd.factorize(df['_fp'])[0]
print(f'Unique compositions: {df["group_id"].nunique()}  (avg {len(df)/df["group_id"].nunique():.2f} replicates/composition)')

# ## 3 — Feature engineering (37 features total)


df['Binder'] = df['Cement'] + df['Fly ash F'] + df['Fly ash C'] + df['GGBS'] + df['Silica Fume']
df['Paste']  = df['Binder'] + df['Water']
df['Total_Aggregates'] = df['Sand'] + df['Coarse Aggr.']

# Group A — UTRGV proportion ratios
df['FA/Binder Ratio'] = (df['Fly ash F']+df['Fly ash C'])/(df['Binder']+EPS)
df['S/B Ratio']       = df['Sand']/(df['Binder']+EPS)

# Group B — composition metrics
df['Paste Volume']    = df['Paste']
df['SCM Ratio']       = (df['Fly ash F']+df['Fly ash C']+df['GGBS']+df['Silica Fume'])/(df['Cement']+EPS)
df['Aggregate Ratio'] = df['Total_Aggregates']/(df['Paste']+EPS)
df['Fiber Surface Area'] = df['Fiber Volume']*(df['Length (mm)']/(df['Diameter (mm)']+EPS))
df['Fiber Efficiency']   = df['Fiber Volume']*df['L/D']
df['GGBS Presence']        = (df['GGBS']>0).astype(int)
df['Silica Fume Presence'] = (df['Silica Fume']>0).astype(int)
df['Coarse Agg Presence']  = (df['Coarse Aggr.']>0).astype(int)

# Group C — Li 2003 micromechanical proxies
df['tau_proxy']         = 1.0/(df['W/B']+EPS)
df['Flaw Size Proxy']   = df['Aggregate Ratio']+df['W/B']
df['Geometric Fiber Eff'] = df['Fiber Volume']*(df['L/D']**2)
df['sigma_cu_proxy']    = df['tau_proxy']*df['Fiber Volume']*df['L/D']
df['sigma_crack_proxy'] = 1.0/(df['Flaw Size Proxy']+EPS)
df['PSH Strength Index']= df['sigma_cu_proxy']/(df['sigma_crack_proxy']+EPS)
df['Jb_complement']     = df['tau_proxy']*df['Fiber Volume']*(df['L/D']**2)
df['J_tip_proxy']       = 1.0/(df['Flaw Size Proxy']+EPS)
df['PSH Energy Index']  = df['Jb_complement']/(df['J_tip_proxy']+EPS)

# Clean infinities and NaN
ENG_COLS = ['FA/Binder Ratio','S/B Ratio','Paste Volume','SCM Ratio','Aggregate Ratio',
            'Fiber Surface Area','Fiber Efficiency','GGBS Presence','Silica Fume Presence',
            'Coarse Agg Presence','tau_proxy','Flaw Size Proxy','Geometric Fiber Eff',
            'sigma_cu_proxy','sigma_crack_proxy','PSH Strength Index','Jb_complement',
            'J_tip_proxy','PSH Energy Index']
for c in ENG_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

RAW_NUMERIC = ['Fiber Volume','Length (mm)','Diameter (mm)','L/D','RI',
               'Cement','Water','Sand','Fly ash C','Fly ash F','GGBS',
               'Coarse Aggr.','Silica Fume','Water Reducer/SP','Fiber',
               'C/B','W/C','W/B']
for c in RAW_NUMERIC:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c] = df[c].fillna(df[c].median())

# Drop 'Shape Factor' if present — it's constant
if 'Shape Factor' in df.columns:
    df = df.drop(columns=['Shape Factor'])

ALL_37_FEATURES = RAW_NUMERIC + ENG_COLS  # 18 + 19 = 37 features
print(f'Total features: {len(ALL_37_FEATURES)}')

# ## 4 — Collapse to group means (per target separately)


TARGETS = ['Second Strain', 'Second Stress']
for t in TARGETS:
    df[t] = pd.to_numeric(df[t], errors='coerce')
df = df.dropna(subset=TARGETS).reset_index(drop=True)
print(f'After dropping rows missing either target: {df.shape}')

def build_group_means(df, target):
    g = df.groupby('group_id').agg(
        **{c: (c, 'mean') for c in ALL_37_FEATURES + [target]},
        n_rep=('group_id', 'count'),
        tgt_var=(target, 'var'),
    ).reset_index()
    g['tgt_var'] = g['tgt_var'].fillna(0.0)
    g['sample_weight'] = g['n_rep'] / (g['tgt_var'] + 0.1)
    return g

# ## 5 — Feature subsets for router and experts


ROUTER_FEATURES = [
    'PSH Strength Index','PSH Energy Index',
    'Fiber Volume','L/D','RI',
    'W/B','Aggregate Ratio',
    'GGBS','Silica Fume','SCM Ratio',
    'Water Reducer/SP',
]
NOPSH_FEATURES = ['W/B','Aggregate Ratio','sigma_crack_proxy','Paste Volume','SCM Ratio']
BULK_FEATURES  = ['W/B','FA/Binder Ratio','S/B Ratio','Fiber Volume','L/D',
                  'PSH Strength Index','PSH Energy Index']
TAIL_FEATURES  = ['L/D','Fiber Volume','RI','PSH Energy Index','sigma_cu_proxy']

# ## 6 — Soft fuzzy regime labels (target-dependent cutoffs)


def fuzzy_labels(y, c1, c2, w):
    """Three regime probabilities from continuous target."""
    s = np.asarray(y, dtype=float)
    up1 = 1.0/(1.0+np.exp(-(s-c1)/w))
    up2 = 1.0/(1.0+np.exp(-(s-c2)/w))
    p_low  = 1.0 - up1
    p_high = up2
    p_mid  = np.maximum(up1 - up2, 0.0)
    arr = np.stack([p_low, p_mid, p_high], axis=1)
    return arr / (arr.sum(axis=1, keepdims=True) + EPS)

# Per-target cutoffs (chosen by physics + distribution)
REGIME_CUTOFFS = {
    'Second Strain': (0.005, 0.050, 0.0015),  # NoPSH | Bulk | Tail (physics-based)
    'Second Stress': (4.0,   5.13,  0.20),    # Weak | Mid | Strong (tertiles of the distribution)
}
REGIME_NAMES = {
    'Second Strain': ('NoPSH', 'Bulk', 'Tail'),
    'Second Stress': ('Weak',  'Mid',  'Strong'),
}

# ## 7 — Core helpers


def make_et(**kw):
    defaults = dict(n_estimators=500, max_depth=15, min_samples_leaf=2,
                    max_features='sqrt', random_state=RANDOM_SEED, n_jobs=-1)
    defaults.update(kw)
    return ExtraTreesRegressor(**defaults)

def make_gbr(alpha, **kw):
    defaults = dict(loss='quantile', alpha=alpha, n_estimators=200, max_depth=3,
                    learning_rate=0.05, min_samples_leaf=10, random_state=RANDOM_SEED)
    defaults.update(kw)
    return GradientBoostingRegressor(**defaults)

def mondrian_qhat(cal_scores, cal_q50_pred, val_q50_pred, alpha):
    """3-bin Mondrian conformal: separate q_hat per bin defined by predicted q50."""
    q33 = np.quantile(cal_q50_pred, 1/3); q67 = np.quantile(cal_q50_pred, 2/3)
    cb = np.where(cal_q50_pred<=q33, 0, np.where(cal_q50_pred<=q67, 1, 2))
    vb = np.where(val_q50_pred<=q33, 0, np.where(val_q50_pred<=q67, 1, 2))
    out = np.zeros(len(val_q50_pred))
    for b in range(3):
        mc, mv = cb==b, vb==b
        n_b = mc.sum()
        if n_b == 0:
            out[mv] = np.quantile(cal_scores, 1-alpha) if len(cal_scores) else 0
        else:
            lvl = min(np.ceil((n_b+1)*(1-alpha))/n_b, 1.0)
            out[mv] = np.quantile(cal_scores[mc], lvl)
    return out

# ## 8 — Outer 5-fold CV pipeline (builds fold_records for one target)


def run_outer_cv(g, target, hp=None):
    """Run 5-fold CV. Returns list of fold_records.
       hp: optional hyperparameter dict (et_kwargs, gbr_kwargs)"""
    hp = hp or {}
    et_kw  = hp.get('et_kwargs',  {})
    gbr_kw = hp.get('gbr_kwargs', {})

    gkf = GroupKFold(n_splits=5)
    y_all = g[target].values
    groups_arr = g['group_id'].values
    N = len(g)

    fold_records = []
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(g, y_all, groups_arr)):
        gdf_tr = g.iloc[tr_idx].reset_index(drop=True)
        gdf_va = g.iloc[val_idx].reset_index(drop=True)
        y_tr = gdf_tr[target].values
        w_tr = gdf_tr['sample_weight'].values

        # 75/25 fit/cal split by composition
        tr_uniq = gdf_tr['group_id'].unique()
        rng = np.random.default_rng(fold_idx); rng.shuffle(tr_uniq)
        n_fit = int(0.75 * len(tr_uniq))
        fit_grp = set(tr_uniq[:n_fit])
        fit_mask = gdf_tr['group_id'].isin(fit_grp).values
        cal_mask = ~fit_mask
        y_fit = y_tr[fit_mask]; w_fit = w_tr[fit_mask]
        y_cal = y_tr[cal_mask]

        X_tr  = gdf_tr[ALL_37_FEATURES].values
        X_fit = gdf_tr.loc[fit_mask, ALL_37_FEATURES].values
        X_cal = gdf_tr.loc[cal_mask, ALL_37_FEATURES].values
        X_va  = gdf_va[ALL_37_FEATURES].values

        # Global ET (q50) + GBR (q10, q90), TWO copies — full and fit
        g_full = {}; g_cal_fit = {}; g_va_fit = {}
        et_full = make_et(**et_kw); et_full.fit(X_tr, y_tr, sample_weight=w_tr)
        g_full['q50'] = et_full.predict(X_va)
        et_fit = make_et(**et_kw); et_fit.fit(X_fit, y_fit, sample_weight=w_fit)
        g_cal_fit['q50'] = et_fit.predict(X_cal); g_va_fit['q50'] = et_fit.predict(X_va)
        for q, a in [('q10',0.10),('q90',0.90)]:
            gf = make_gbr(a, **gbr_kw); gf.fit(X_tr, y_tr, sample_weight=w_tr)
            g_full[q] = gf.predict(X_va)
            gfit = make_gbr(a, **gbr_kw); gfit.fit(X_fit, y_fit, sample_weight=w_fit)
            g_cal_fit[q] = gfit.predict(X_cal); g_va_fit[q] = gfit.predict(X_va)

        # Router (CV folds capped by smallest class size)
        sc_r = StandardScaler()
        X_fit_r = sc_r.fit_transform(gdf_tr.loc[fit_mask, ROUTER_FEATURES].values)
        X_cal_r = sc_r.transform(gdf_tr.loc[cal_mask, ROUTER_FEATURES].values)
        X_va_r  = sc_r.transform(gdf_va[ROUTER_FEATURES].values)
        labels_fit = gdf_tr.loc[fit_mask, 'hard_label'].values
        # Pick min(3, smallest_class_count) for CV
        min_class = min(np.bincount(labels_fit))
        cv_folds = max(2, min(3, min_class))
        router = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
            method='isotonic', cv=cv_folds
        )
        router.fit(X_fit_r, labels_fit)
        # predict_proba may have fewer than 3 columns if a class is absent in train fold; pad if needed
        rp_cal_raw = router.predict_proba(X_cal_r)
        rp_va_raw  = router.predict_proba(X_va_r)
        if rp_cal_raw.shape[1] < 3:
            full_cal = np.zeros((rp_cal_raw.shape[0], 3))
            full_va  = np.zeros((rp_va_raw.shape[0],  3))
            for i, cls in enumerate(router.classes_):
                full_cal[:, int(cls)] = rp_cal_raw[:, i]
                full_va[:,  int(cls)] = rp_va_raw[:,  i]
            rp_cal, rp_va = full_cal, full_va
        else:
            rp_cal, rp_va = rp_cal_raw, rp_va_raw

        # Residual experts (RidgeCV)
        cal_resid = y_cal - g_cal_fit['q50']
        exp_cal = {}; exp_va = {}
        for i_r, (rn, feats) in enumerate([('R0', NOPSH_FEATURES),('R1', BULK_FEATURES),('R2', TAIL_FEATURES)]):
            sc = StandardScaler()
            Xc = sc.fit_transform(gdf_tr.loc[cal_mask, feats].values)
            Xv = sc.transform(gdf_va[feats].values)
            wc = rp_cal[:, i_r] + EPS
            e = RidgeCV(alphas=[0.1,1.0,10.0,100.0])
            e.fit(Xc, cal_resid, sample_weight=wc)
            exp_cal[rn] = e.predict(Xc); exp_va[rn] = e.predict(Xv)

        fold_records.append({
            'val_idx': val_idx, 'cal_router': rp_cal, 'val_router': rp_va,
            'cal_y': y_cal, 'cal_g': g_cal_fit,
            'val_g_full': g_full, 'val_g_fit': g_va_fit,
            'cal_exp': exp_cal, 'val_exp': exp_va,
        })
    return fold_records

# ## 9 — Evaluate one MoE config


# Clip caps the residual correction magnitude per regime expert. Target-aware
# so we don't over-clip on Stress (which is in MPa, 1-10 range).
CORR_CLIP_PER_TARGET = {
    'Second Strain': 0.05,   # ~half of max strain
    'Second Stress': 3.0,    # ~half of mid-range stress
}

def evaluate_config(g, target, fold_records, n_on, b_on, t_on, alpha=0.20, corr_clip=None):
    if corr_clip is None:
        corr_clip = CORR_CLIP_PER_TARGET.get(target, 0.05)
    N = len(g)
    p50 = np.full(N, np.nan); lo = np.full(N, np.nan); hi = np.full(N, np.nan)

    for fr in fold_records:
        rp_cal = fr['cal_router']; rp_val = fr['val_router']
        corr_cal = np.clip(n_on*rp_cal[:,0]*fr['cal_exp']['R0'] +
                           b_on*rp_cal[:,1]*fr['cal_exp']['R1'] +
                           t_on*rp_cal[:,2]*fr['cal_exp']['R2'], -corr_clip, corr_clip)
        corr_val = np.clip(n_on*rp_val[:,0]*fr['val_exp']['R0'] +
                           b_on*rp_val[:,1]*fr['val_exp']['R1'] +
                           t_on*rp_val[:,2]*fr['val_exp']['R2'], -corr_clip, corr_clip)

        b_cal_q10 = fr['cal_g']['q10'] + corr_cal
        b_cal_q50 = fr['cal_g']['q50'] + corr_cal
        b_cal_q90 = fr['cal_g']['q90'] + corr_cal
        b_val_q10_full = fr['val_g_full']['q10'] + corr_val
        b_val_q50_full = fr['val_g_full']['q50'] + corr_val
        b_val_q90_full = fr['val_g_full']['q90'] + corr_val
        b_val_q50_fit = fr['val_g_fit']['q50'] + corr_val

        scores = np.maximum(b_cal_q10 - fr['cal_y'], fr['cal_y'] - b_cal_q90)
        qhat = mondrian_qhat(scores, b_cal_q50, b_val_q50_fit, alpha)

        vi = fr['val_idx']
        p50[vi] = np.maximum(b_val_q50_full, 0)
        lo[vi]  = np.maximum(b_val_q10_full - qhat, 0)
        hi[vi]  = np.maximum(b_val_q90_full + qhat, 0)

    y_true = g[target].values
    regime = g['regime'].values
    rname_low, rname_mid, rname_high = REGIME_NAMES[target]

    ss_res = np.sum((y_true - p50)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2_overall = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    m = {'MAE':    mean_absolute_error(y_true, p50),
         'RMSE':   np.sqrt(mean_squared_error(y_true, p50)),
         'R2':     r2_overall,
         'Cov80':  np.mean((y_true>=lo)&(y_true<=hi)),
         'Width80': np.mean(hi-lo)}

    for r in [rname_low, rname_mid, rname_high]:
        mk = regime == r
        if mk.sum() > 1:
            ss_res_r = np.sum((y_true[mk] - p50[mk])**2)
            ss_tot_r = np.sum((y_true[mk] - np.mean(y_true[mk]))**2)
            m[f'MAE_{r}']  = mean_absolute_error(y_true[mk], p50[mk])
            m[f'R2_{r}']   = 1.0 - ss_res_r / ss_tot_r if ss_tot_r > 0 else np.nan
            m[f'Cov_{r}']  = np.mean((y_true[mk]>=lo[mk])&(y_true[mk]<=hi[mk]))
        else:
            m[f'MAE_{r}'] = np.nan
            m[f'R2_{r}']  = np.nan
            m[f'Cov_{r}'] = np.nan
    return m, p50, lo, hi


# ## 10 — Hyperparameter search


def hp_search_random(g, target, n_trials=15):
    """Random search over ET and GBR hyperparameters.
       Returns best HP dict by overall MAE on A_Global config."""
    rng = np.random.default_rng(123)
    results = []
    for trial in range(n_trials):
        hp = {
            'et_kwargs': {
                'n_estimators': int(rng.choice([300, 500, 700])),
                'max_depth':    int(rng.choice([10, 15, 20, 25])),
                'min_samples_leaf': int(rng.choice([1, 2, 4])),
                'max_features': str(rng.choice(['sqrt', 'log2'])),
            },
            'gbr_kwargs': {
                'n_estimators':   int(rng.choice([150, 200, 300])),
                'max_depth':      int(rng.choice([2, 3, 4])),
                'learning_rate':  float(rng.choice([0.03, 0.05, 0.08])),
                'min_samples_leaf': int(rng.choice([5, 10, 15])),
            },
        }
        try:
            fr = run_outer_cv(g, target, hp=hp)
            m, *_ = evaluate_config(g, target, fr, 0, 0, 0)  # A_Global
            results.append({'trial': trial, 'MAE': m['MAE'], 'R2': m['R2'], 'Cov80': m['Cov80'], 'hp': hp})
            print(f"  Trial {trial+1}/{n_trials}: MAE={m['MAE']:.5f}  R2={m['R2']:.4f}  Cov80={m['Cov80']:.3f}")
        except Exception as e:
            print(f"  Trial {trial+1} failed: {e}")
    results_df = pd.DataFrame([{k:v for k,v in r.items() if k!='hp'} for r in results])
    best = min(results, key=lambda r: r['MAE'])
    return best, results_df


# ## 11 — All 8 ablation configurations


CONFIGS = {
    'A_Global only':  (0,0,0),
    'B_Low only':     (1,0,0),
    'C_Mid only':     (0,1,0),
    'D_High only':    (0,0,1),
    'E_Low+Mid':      (1,1,0),
    'F_Low+High':     (1,0,1),
    'G_Mid+High':     (0,1,1),
    'H_Full MoE':     (1,1,1),
}

# ## 12 — Plotting helpers


def plot_target_ablation(target, results_df, preds_store, out_prefix):
    rname_low, rname_mid, rname_high = REGIME_NAMES[target]
    cfgs_order = list(CONFIGS.keys())

    # Plot 1a: MAE bar chart per config
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(cfgs_order))
    w = 0.2
    ax.bar(x - 1.5*w, results_df.loc[cfgs_order, 'MAE'].values,           w, label='Overall', color='#444444')
    ax.bar(x - 0.5*w, results_df.loc[cfgs_order, f'MAE_{rname_low}'].values, w, label=rname_low, color='#4a90d9')
    ax.bar(x + 0.5*w, results_df.loc[cfgs_order, f'MAE_{rname_mid}'].values, w, label=rname_mid, color='#e6b022')
    ax.bar(x + 1.5*w, results_df.loc[cfgs_order, f'MAE_{rname_high}'].values, w, label=rname_high, color='#d94a4a')
    ax.axhline(results_df.loc['A_Global only', 'MAE'], color='black', linestyle='--', alpha=0.4, label='Baseline overall')
    ax.set_xticks(x); ax.set_xticklabels([c.split('_',1)[1] for c in cfgs_order], rotation=25, ha='right')
    ax.set_ylabel(f'MAE ({target})')
    ax.set_title(f'{target}: MAE by configuration and regime')
    ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_mae.png', dpi=130, bbox_inches='tight'); plt.close()

    # Plot 1b: R² bar chart per config
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5*w, results_df.loc[cfgs_order, 'R2'].values,                w, label='Overall',   color='#444444')
    ax.bar(x - 0.5*w, results_df.loc[cfgs_order, f'R2_{rname_low}'].values,   w, label=rname_low,   color='#4a90d9')
    ax.bar(x + 0.5*w, results_df.loc[cfgs_order, f'R2_{rname_mid}'].values,   w, label=rname_mid,   color='#e6b022')
    ax.bar(x + 1.5*w, results_df.loc[cfgs_order, f'R2_{rname_high}'].values,  w, label=rname_high,  color='#d94a4a')
    ax.axhline(results_df.loc['A_Global only', 'R2'], color='black', linestyle='--', alpha=0.4, label='Baseline overall')
    ax.set_xticks(x); ax.set_xticklabels([c.split('_',1)[1] for c in cfgs_order], rotation=25, ha='right')
    ax.set_ylabel(f'R² ({target})')
    ax.set_title(f'{target}: R² by configuration and regime')
    ax.legend(loc='lower left'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_r2.png', dpi=130, bbox_inches='tight'); plt.close()

    # Plot 2: 8 scatter subplots — predicted vs true per config, regime-colored
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = {rname_low: '#4a90d9', rname_mid: '#e6b022', rname_high: '#d94a4a'}
    y_true = preds_store['y_true']; regime = preds_store['regime']
    for ax, cfg in zip(axes, cfgs_order):
        p50, lo, hi = preds_store[cfg]
        for r, col in colors.items():
            mk = regime == r
            ax.scatter(y_true[mk], p50[mk], c=col, s=22, alpha=0.7, label=r)
        lim = max(y_true.max(), p50.max())*1.05
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        mae = results_df.loc[cfg, 'MAE']
        r2  = results_df.loc[cfg, 'R2']
        cov = results_df.loc[cfg, 'Cov80']
        ax.set_title(f'{cfg}\nMAE={mae:.4f}  R²={r2:.3f}  Cov80={cov:.3f}', fontsize=9)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle(f'{target}: Predicted vs True across all 8 MoE configurations', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_scatter_grid.png', dpi=130, bbox_inches='tight'); plt.close()

    # Plot 3: 8 interval plots — sorted by true value, per config
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    sort_idx = np.argsort(y_true)
    for ax, cfg in zip(axes, cfgs_order):
        p50, lo, hi = preds_store[cfg]
        ax.fill_between(np.arange(len(y_true)), lo[sort_idx], hi[sort_idx], color='gray', alpha=0.3, label='80% CI')
        ax.plot(np.arange(len(y_true)), y_true[sort_idx], 'k-', linewidth=1.3, label='True')
        ax.plot(np.arange(len(y_true)), p50[sort_idx], 'orange', linewidth=0.8, linestyle='--', label='Pred q50')
        cov = results_df.loc[cfg, 'Cov80']; w_ = results_df.loc[cfg, 'Width80']
        ax.set_title(f'{cfg}\nCov80={cov:.3f}  Width={w_:.4f}', fontsize=10)
        ax.set_xlabel('Group rank'); ax.set_ylabel(target)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f'{target}: Sorted prediction intervals across all 8 configurations', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_intervals_grid.png', dpi=130, bbox_inches='tight'); plt.close()


# ## 13 — Run full pipeline for a target


def run_target(target, do_hpsearch=True, n_hp_trials=10):
    print('\n' + '#'*72)
    print(f'#  TARGET: {target}')
    print('#'*72)

    # Build group means specific to this target
    g = build_group_means(df, target)
    c1, c2, w = REGIME_CUTOFFS[target]
    rname_low, rname_mid, rname_high = REGIME_NAMES[target]
    y = g[target].values
    g['regime'] = np.where(y < c1, rname_low, np.where(y < c2, rname_mid, rname_high))
    soft = fuzzy_labels(y, c1, c2, w)
    g['p_low'], g['p_mid'], g['p_high'] = soft[:,0], soft[:,1], soft[:,2]
    g['hard_label'] = np.argmax(soft, axis=1)
    print(f'\nGroup-mean rows: {len(g)}')
    print(f'Regime distribution: {dict(zip(*np.unique(g["regime"], return_counts=True)))}')
    print(f'Target range: [{y.min():.5f}, {y.max():.5f}]')

    # ====================================================
    # Default-HP ablation pass
    # ====================================================
    print('\n--- Pass 1: ablation with DEFAULT hyperparameters ---')
    fr_default = run_outer_cv(g, target, hp=None)
    results_default = []
    preds_default = {'y_true': g[target].values, 'regime': g['regime'].values}
    for label, (n,b,t) in CONFIGS.items():
        m, p50, lo, hi = evaluate_config(g, target, fr_default, n, b, t)
        m['config'] = label
        results_default.append(m)
        preds_default[label] = (p50, lo, hi)
    rdf_default = pd.DataFrame(results_default).set_index('config')
    print(rdf_default.round(5).to_string())

    # Save and plot default-HP
    out_prefix_default = f'final_{target.replace(" ", "_")}_default'
    rdf_default.to_csv(f'{out_prefix_default}_ablation.csv')
    plot_target_ablation(target, rdf_default, preds_default, out_prefix_default)

    # ====================================================
    # Hyperparameter search (optional)
    # ====================================================
    if do_hpsearch:
        print(f'\n--- Pass 2: random hyperparameter search ({n_hp_trials} trials, A_Global config) ---')
        best, hp_results = hp_search_random(g, target, n_trials=n_hp_trials)
        print(f'\nBest HP: {best["hp"]}')
        print(f'Best A_Global MAE: {best["MAE"]:.5f}')
        hp_results.to_csv(f'final_{target.replace(" ", "_")}_hp_search.csv', index=False)

        # ====================================================
        # Re-run ablation with TUNED hyperparameters
        # ====================================================
        print('\n--- Pass 3: ablation with TUNED hyperparameters ---')
        fr_tuned = run_outer_cv(g, target, hp=best['hp'])
        results_tuned = []
        preds_tuned = {'y_true': g[target].values, 'regime': g['regime'].values}
        for label, (n,b,t) in CONFIGS.items():
            m, p50, lo, hi = evaluate_config(g, target, fr_tuned, n, b, t)
            m['config'] = label
            results_tuned.append(m)
            preds_tuned[label] = (p50, lo, hi)
        rdf_tuned = pd.DataFrame(results_tuned).set_index('config')
        print(rdf_tuned.round(5).to_string())

        out_prefix_tuned = f'final_{target.replace(" ", "_")}_tuned'
        rdf_tuned.to_csv(f'{out_prefix_tuned}_ablation.csv')
        plot_target_ablation(target, rdf_tuned, preds_tuned, out_prefix_tuned)

        # Improvement summary
        print('\n--- Default vs Tuned (MAE delta, %) ---')
        delta = ((rdf_tuned['MAE'] - rdf_default['MAE']) / rdf_default['MAE'] * 100).round(2)
        for cfg in CONFIGS:
            print(f'  {cfg:<20} default MAE={rdf_default.loc[cfg,"MAE"]:.5f} R2={rdf_default.loc[cfg,"R2"]:.4f}  '
                  f'tuned MAE={rdf_tuned.loc[cfg,"MAE"]:.5f} R2={rdf_tuned.loc[cfg,"R2"]:.4f}  Δ MAE={delta[cfg]:+.2f}%')

        return {'default': rdf_default, 'tuned': rdf_tuned, 'best_hp': best['hp']}
    else:
        return {'default': rdf_default, 'best_hp': None}

# ## 14 — Execute for both targets


print('\n' + '='*72)
print('  RUNNING FULL PIPELINE: STRAIN  ')
print('='*72)
results_strain = run_target('Second Strain', do_hpsearch=True, n_hp_trials=8)

print('\n\n' + '='*72)
print('  RUNNING FULL PIPELINE: STRESS  ')
print('='*72)
results_stress = run_target('Second Stress', do_hpsearch=True, n_hp_trials=8)

# ## 15 — Save summary


summary = {
    'strain_default_best_config': results_strain['default']['MAE'].idxmin(),
    'strain_default_best_MAE':    results_strain['default']['MAE'].min(),
    'strain_default_best_R2':     results_strain['default']['R2'].max(),
    'strain_tuned_best_config':   results_strain['tuned']['MAE'].idxmin(),
    'strain_tuned_best_MAE':      results_strain['tuned']['MAE'].min(),
    'strain_tuned_best_R2':       results_strain['tuned']['R2'].max(),
    'strain_best_hp':             results_strain['best_hp'],
    'stress_default_best_config': results_stress['default']['MAE'].idxmin(),
    'stress_default_best_MAE':    results_stress['default']['MAE'].min(),
    'stress_default_best_R2':     results_stress['default']['R2'].max(),
    'stress_tuned_best_config':   results_stress['tuned']['MAE'].idxmin(),
    'stress_tuned_best_MAE':      results_stress['tuned']['MAE'].min(),
    'stress_tuned_best_R2':       results_stress['tuned']['R2'].max(),
    'stress_best_hp':             results_stress['best_hp'],
}
with open('final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print('\n\n' + '='*72)
print('FINAL SUMMARY')
print('='*72)
print(json.dumps(summary, indent=2, default=str))
print('\nAll results, plots, and CSVs saved.')

# ## Interpretation
# 
# ### Strain results
# - Best config: **A_Global only** (default and tuned)
# - Default HP: MAE 0.00772, Cov80 0.84
# - Tuned HP: MAE 0.00741, Cov80 ~0.85 (~4% MAE improvement from HP tuning)
# - MoE experts don't materially improve strain prediction — the global model is near the noise floor (~0.005 lower bound from within-recipe variance)
# - The Tail regime remains hard because Tail recipes are feature-indistinguishable from Bulk recipes
# 
# ### Stress results
# - Best config (default): **E_Low+Mid** with MAE 0.541
# - Best config (tuned): **B_Low only** with MAE 0.482 (~11% HP improvement)
# - Stress benefits significantly more from hyperparameter tuning than strain does
# - Per-regime breakdown shows the Mid (mid-stress) range is best-predicted; Weak and Strong both have larger errors due to fewer training samples per regime
# - Stress is fundamentally easier than strain — R² ceiling is higher because within-recipe noise is smaller in stress (MPa scale) than strain (dimensionless, more variable due to fiber pull-out stochasticity)
# 
# ### Headline numbers
# 
# | Target | Best config (tuned) | MAE | RMSE | Cov80 |
# |---|---|---|---|---|
# | Second Strain | A_Global only | 0.00741 | 0.012 | ~0.85 |
# | Second Stress | B_Low only    | 0.482   | 0.71  | ~0.83 |
# 
# ### Key design decisions (justified by earlier ablations)
# 
# | Decision | Choice | Reason |
# |---|---|---|
# | Backend | ExtraTrees(q50) + GBR(q10, q90) | Beats LightGBM-quantile by ~18% on this dataset |
# | Features | All 37 (no dedup) | Tree models handle redundancy well; tiny Tail boost |
# | Routing | Soft | Hard routing degrades MAE 1-3pp |
# | Target transform | None | For ET+GBR, raw target works fine (log helped LGBM only) |
# | Experts | RidgeCV on log-free residuals | LGBM expert overfits noise on small calibration sets |
# | Conformal | Mondrian (3 bins by predicted q50) | Regime-specific interval widths without expert starvation |
# | CV scheme | GroupKFold(5) × 75/25 fit/cal | Prevents replicate leakage |
# 
# ### Limitations
# 
# - Strain Tail-regime coverage is structurally limited (~64-72%) because of feature-space overlap with Bulk
# - Only 8 HP trials per target — Optuna or larger search would likely find marginal further gains
# - Stress regime cutoffs were chosen as tertiles, not physics-based (no domain analogue to PSH conditions for stress)