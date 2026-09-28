# --- 0. Imports ---
import os
import numpy as np
import pandas as pd
import io
import gc
import warnings
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- 1. Azure Configuration (from environment variables) ---
AZURE_TENANT_ID      = os.environ['AZURE_TENANT_ID']
AZURE_CLIENT_ID      = os.environ['AZURE_CLIENT_ID']
AZURE_CLIENT_SECRET  = os.environ['AZURE_CLIENT_SECRET']
AZURE_STORAGE_ACCOUNT   = os.environ['AZURE_STORAGE_ACCOUNT']
AZURE_STORAGE_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']

def get_container_client():
    credential = ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
    )
    service = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=credential,
    )
    return service.get_container_client(AZURE_STORAGE_CONTAINER)

container = get_container_client()

def blob_exists(blob_path):
    try:
        container.get_blob_client(blob_path).get_blob_properties()
        return True
    except Exception:
        return False

def download_blob(blob_path):
    buf = io.BytesIO()
    container.get_blob_client(blob_path).download_blob().readinto(buf)
    buf.seek(0)
    return buf

def upload_blob(blob_path, data_bytes):
    container.get_blob_client(blob_path).upload_blob(data_bytes, overwrite=True)

def list_blobs_prefix(prefix):
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]

# --- 2. K sweep parameters ---
K_VALUES      = [1] + list(range(5, 51, 5))   # [1, 5, 10, 15, ..., 50]
P300_CHANNELS = None
P300_FEATURES = None

# --- 3. Subject metadata per dataset ---
korean_meta = {
    'gender': {
        0: ['01','02','03','04','05','07','08','09','10','11','12','13','14',
            '16','18','19','20','22','23','24','25','26','27','28','30','31',
            '32','34','36','39','41','42','44','45','46','47','51','52','53','54','55'],
        1: ['06','15','17','21','29','33','35','37','38','40','43','48','49','50']
    },
    'corrected_vision': {
        0: ['01','03','05','13','15','21','22','23','25','31','36','37','54'],
        1: ['02','04','06','07','08','09','10','11','12','14','16','17','18',
            '19','20','24','26','27','28','29','30','32','33','34','35','38',
            '39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','55']
    },
    'biofeedback_experience': {
        0: ['01','02','03','04','05','06','07','08','10','11','12','13','14',
            '16','17','18','19','20','21','22','23','24','25','26','28','29',
            '30','31','32','33','34','35','36','37','39','40','42','43','44',
            '45','46','47','48','49','50','52','53','54','55'],
        1: ['09','15','27','38','41','51']
    },
    'subjective_state': {
        0: ['03','05','08','10','17','20','22','24','30','33','34','35','36',
            '37','39','45','46','47','54','55'],
        1: ['01','02','04','06','11','12','15','16','18','19','21','25','26',
            '28','29','32','41','48','49','51','52','53'],
        2: ['07','09','13','14','23','27','31','38','40','42','43','44','50']
    },
    'task_difficulty': {
        0: ['01','02','03','04','05','07','11','12','13','14','16','17','20',
            '22','23','24','27','28','32','43','50','51','52','53','54','55'],
        1: ['06','08','09','10','15','18','19','21','25','26','29','30','31',
            '33','34','35','36','37','38','39','40','41','42','44','45','46','47','48','49']
    }
}

giga_meta = {
    'gender': {
        0: [f"{i:02d}" for i in [5,7,9,13,14,15,16,20,23,24,25,26,27,28,29,
                                  32,33,34,35,37,38,40,41,42,45,48,50,51,52,53,54]],
        1: [f"{i:02d}" for i in [1,2,3,4,6,8,10,11,12,17,18,19,21,22,30,31,
                                  36,39,43,44,46,47,49]]
    },
    'bci_experience': {
        0: [f"{i:02d}" for i in [4,7,8,9,11,12,15,16,17,18,19,22,23,24,25,26,
                                  28,29,31,32,34,35,36,37,38,39,41,43,44,45,48,50,51,54]],
        1: [f"{i:02d}" for i in [1,2,3,5,6,10,13,14,20,21,27,30,33,40,42,46,47,49,52,53]]
    },
    'subjective_state': {
        0: [f"{i:02d}" for i in [1,2,4,7,14,15,21,24,28,34,35,41,45,51]],
        1: [f"{i:02d}" for i in [3,6,9,10,11,12,13,19,20,25,27,29,31,32,33,36,43,44,48,49,50,53]],
        2: [f"{i:02d}" for i in [5,8,16,17,18,22,23,26,30,37,38,39,40,42,46,47,52,54]]
    },
    'task_performance': {
        0: [f"{i:02d}" for i in [8,18,40,42,54]],
        1: [f"{i:02d}" for i in [2,3,4,5,9,10,11,12,14,17,19,22,27,28,29,33,37,38,41,45,46,47,48,49]],
        2: [f"{i:02d}" for i in [1,6,7,13,15,16,20,21,23,24,25,26,30,31,32,34,35,36,39,43,44,50,51,52,53]]
    }
}

adaptive_meta = {
    'gender': {
        0: [s.replace('s','').zfill(2) for s in
            ['s01','s03','s04','s06','s07','s10','s13','s15','s18','s19',
             's22','s23','s26','s28','s30','s40','s43','s44','s45']],
        1: [s.replace('s','').zfill(2) for s in
            ['s02','s05','s08','s09','s11','s12','s14','s16','s17','s20',
             's21','s24','s25','s27','s29','s31','s32','s33','s34','s35',
             's36','s37','s38','s39','s41','s42','s46','s47']]
    },
    'subjective_state': {
        0: [s.replace('s','').zfill(2) for s in
            ['s02','s05','s07','s16','s17','s19','s23','s25','s26','s31','s36','s38','s40']],
        1: [s.replace('s','').zfill(2) for s in
            ['s08','s10','s11','s12','s13','s14','s15','s22','s32','s34','s39','s45']],
        2: [s.replace('s','').zfill(2) for s in
            ['s01','s03','s04','s06','s09','s18','s20','s21','s24','s27','s28',
             's29','s30','s33','s35','s37','s41','s42','s43','s44','s46','s47']]
    },
    'task_performance': {
        0: [s.replace('s','').zfill(2) for s in ['s42','s44']],
        1: [s.replace('s','').zfill(2) for s in
            ['s03','s04','s05','s06','s10','s11','s15','s16','s17','s19','s20',
             's22','s23','s26','s28','s29','s30','s36','s38','s41','s45']],
        2: [s.replace('s','').zfill(2) for s in
            ['s01','s02','s07','s08','s09','s12','s13','s14','s18','s21','s24',
             's25','s27','s31','s32','s33','s34','s35','s37','s39','s40','s43','s46','s47']]
    }
}

# --- 4. Dataset configs ---
dataset_configs = {
    'korean': {
        'enabled':         True,
        'ft_prefix':      'korean/parquet_ft',
        'binned_prefix':  'korean/parquet_binned_both',
        'epoch_type':     'target',
        'channels':       [
            'fp1', 'f7', 'f3', 'fc1', 'fc5', 'c4', 'cp2', 'fc6',
            'f8', 'f4', 'fc2', 'fp2', 'fz', 'cz', 'cp1', 'pz'
        ],
        'features':       None,
        'group_cols':     ['subject_id', 'run_type', 'run_id'],
        'run_covariate':  'run_group',
        'run_grouping':   {'train': 'train', 'test': 'test'},
        'metadata':       korean_meta,
    },
    'giga': {
        'enabled':         True,
        'ft_prefix':      'giga_preprocessed/parquet_ft',
        'binned_prefix':  'giga_preprocessed/parquet_binned_both',
        'epoch_type':     'non-target',
        'channels':       [
            'fc1', 'fc2', 'cz', 'c3', 'cp5', 'cp1', 'cp2', 'pz',
            'o1', 'oz', 'o2', 'c1', 'c2', 'cp3', 'cpz', 'cp4',
            'p1', 'p2', 'poz', 'c4',
        ],
        'features':       None,
        'group_cols':     ['subject_id', 'session_id', 'split'],
        'run_covariate':  'run_group',
        'run_grouping':   {'train': 'train', 'test': 'test'},
        'metadata':       giga_meta,
    },
    'adaptive': {
        'enabled':         True,
        'ft_prefix':      'adaptiveP300/parquet_ft',
        'binned_prefix':  'adaptiveP300/parquet_binned_both',
        'epoch_type':     'target',
        'channels':       [
            'cp1', 'cp2', 'cz', 'fp1', 'fz', 'f3', 'f4', 'fc1',
            'fc2', 'c4', 'c3',
        ],
        'features':       None,
        'group_cols':     ['subject_id', 'file_type'],
        'run_covariate':  'run_group',
        'run_grouping': {
            'calibration-signals': 'calibration',
            'eval-raw':            'eval',
            'training-run-1-raw':  'training',
            'training-run-2-raw':  'training',
            'training-run-3-raw':  'training',
            'training-run-4-raw':  'training',
            'training-run-5-raw':  'training',
            'post-training-raw':   'post_training',
        },
        'metadata':       adaptive_meta,
    },
}
# --- 5. Helpers ---

def build_meta_map(meta_dict, col_name):
    rows = [(subj, label) for label, subjs in meta_dict.items() for subj in subjs]
    return pd.DataFrame(rows, columns=['subject_id', col_name])

def bin_features(df, group_cols, feature_cols, k, epoch_type):
    """Average features over bins of k selected epoch types per group."""
    df = df[df['epoch_type'] == epoch_type].copy()
    results = []
    for keys, group in df.groupby(group_cols):
        group = group.sort_values('epoch_num')
        num_bins = len(group) // k
        for b in range(num_bins):
            block    = group.iloc[b*k:(b+1)*k]
            avg_feat = block[feature_cols].mean()
            entry    = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
            entry['bin_epoch_num'] = b + 1
            entry.update(avg_feat.to_dict())
            results.append(entry)
    return pd.DataFrame(results)

def _bin_one_azure_blob(blob_path, group_cols, feature_cols, k, epoch_type):
    """
    Worker: construct own Azure client, download one parquet, bin it.
    Returns binned DataFrame or None on error.
    Uses loky backend so must construct its own client (not picklable across processes).
    """
    try:
        credential = ClientSecretCredential(
            tenant_id=AZURE_TENANT_ID,
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET,
        )
        service = BlobServiceClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential,
        )
        container_local = service.get_container_client(AZURE_STORAGE_CONTAINER)

        buf = io.BytesIO()
        container_local.get_blob_client(blob_path).download_blob().readinto(buf)
        buf.seek(0)
        df_file = pd.read_parquet(buf)
        df_file['subject_id'] = df_file['subject_id'].astype(str).str.zfill(2)
        full_group_cols = group_cols + ['channel_name']
        return bin_features(df_file, full_group_cols, feature_cols, k, epoch_type)
    except Exception as e:
        print(f"  [error] {blob_path}: {e}")
        return None

def get_or_create_binned_ft(ft_prefix, binned_prefix, group_cols, k,
                            dataset_name, epoch_type, feature_names):
    epoch_tag = epoch_type.replace('-', '_')
    out_path = f"{binned_prefix}/k_{k}/{dataset_name}_binned_{epoch_tag}.parquet"

    # Cache hit
    if blob_exists(out_path):
        print(f"  [cache hit] {out_path}")
        buf = download_blob(out_path)
        return pd.read_parquet(buf)

    print(f"  [cache miss] computing binned parquet...")

    # List all input blobs
    blob_paths = [b for b in list_blobs_prefix(ft_prefix) if b.endswith('.parquet')]

    if not blob_paths:
        print(f"  No ft parquets found under {ft_prefix}")
        return None

    # Infer feature cols from first blob
    first_buf = download_blob(blob_paths[0])
    df_first  = pd.read_parquet(first_buf)

    non_feat = {'subject_id','session_id','run_type','run_id',
                'file_type','split','channel_name','epoch_type',
                'epoch_num','run_group'}
    feature_cols = (
        [f for f in feature_names if f in df_first.columns]
        if feature_names is not None
        else [c for c in df_first.columns if c not in non_feat]
    )
    del df_first

    # Parallel binning — loky backend, own Azure client per worker
    binned_chunks = Parallel(n_jobs=-1, backend='loky', verbose=5)(
        delayed(_bin_one_azure_blob)(
            blob_path, group_cols, feature_cols, k, epoch_type
        )
        for blob_path in blob_paths
    )
    binned_chunks = [c for c in binned_chunks if c is not None and not c.empty]

    if not binned_chunks:
        print(f"  No data after binning under {ft_prefix}")
        return None

    df_binned = pd.concat(binned_chunks, ignore_index=True)
    df_binned['subject_id'] = df_binned['subject_id'].astype(str).str.zfill(2)
    del binned_chunks
    gc.collect()

    # Upload
    out_buf = io.BytesIO()
    df_binned.to_parquet(out_buf, index=False, engine='pyarrow', compression='snappy')
    upload_blob(out_path, out_buf.getvalue())
    print(f"  [uploaded] {out_path}  ({len(df_binned)} rows)")

    return df_binned

# ==============================================================================
# 5. Singularity Criterion Definition (Scale-Aware / Numerical Precision)
# ==============================================================================
# In mixed-effects models (LME), singularity indicates that the random effects
# variance-covariance matrix lies on the boundary of the parameter space.
#
# WHY SCALE-INDEPENDENT CRITERIA ARE NECESSARY:
# EEG features vary across physical units:
#   - Raw voltages (V): feature variances are ~ 10^-12 to 10^-11 V^2.
#     An absolute threshold (e.g. 1e-5) would erroneously drop 100% of valid fits.
#   - Microvolts squared (uV^2) or power: variances can exceed 10^2 to 10^4.
#     An absolute threshold would fail to detect true boundary collapse.
#
# DEFINITION BASED ON NUMERICAL PRECISION AND MODEL SCALE:
# We evaluate boundary singularity using scale-invariant metrics:
#   - sigma_u^2: between-subject random intercept variance (fit.cov_re)
#   - sigma_e^2: residual within-subject variance (fit.scale)
#   - Intraclass Correlation Coefficient: ICC = sigma_u^2 / (sigma_u^2 + sigma_e^2)
#   - Relative variance: rel_var = sigma_u^2 / sigma_e^2
#
# A fit is classified as NUMERICALLY BOUNDARY-SINGULAR if:
#   1. Non-positive variance: sigma_u^2 <= 0, or is NaN / None.
#   2. Scale-relative boundary: ICC <= ICC_BOUNDARY_TOL (1e-5) or rel_var <= REL_VAR_BOUNDARY_TOL (1e-5).
#      At ICC <= 1e-5, < 0.001% of variance is between-subject, placing the parameter
#      on the numerical boundary floor of the optimizer. (Note: in lme4, isSingular()
#      uses a Cholesky factor tolerance theta <= 1e-4, which corresponds to rel_var <= 1e-8).
#   3. Degenerate parameter covariance: fit.bse contains NaNs or Infs (singular Hessian).
# ==============================================================================
ICC_BOUNDARY_TOL     = 1e-5
REL_VAR_BOUNDARY_TOL = 1e-5

def fit_one(chan, feat, df, group_var, run_covariate):
    warnings.simplefilter('ignore')
    df_chan = df[df['channel_name'] == chan]
    cols    = ['subject_id', group_var, run_covariate, 'bin_epoch_num', feat]
    d       = df_chan[cols].dropna(subset=[feat]).copy()
    d       = d.rename(columns={
        'subject_id': 'Subject', 'bin_epoch_num': 'Bin', feat: 'Value'
    })
    d['Bin'] = d['Bin'].astype('category')

    res_record = {
        'chan': chan,
        'feature': feat,
        'group_var': group_var,
        'converged': False,
        're_var': np.nan,
        'resid_var': np.nan,
        'icc': np.nan,
        'rel_var': np.nan,
        'is_singular': True,
        'singular_reason': None,
        'p_interaction': None,
        'p_main': None,
        'p_bin': None,
        'error': None
    }

    try:
        formula = f"Value ~ {group_var} * Bin + {run_covariate}"
        fit     = smf.mixedlm(formula, d, groups=d['Subject']).fit(
                      reml=True, method='lbfgs')

        # 1. Optimizer convergence status
        converged = bool(getattr(fit, 'converged', False))
        res_record['converged'] = converged

        # 2. Record random-intercept variance and residual variance for EVERY fit
        try:
            re_var = float(np.asarray(fit.cov_re).item())
        except Exception:
            re_var = np.nan
        res_record['re_var'] = re_var

        try:
            resid_var = float(fit.scale)
        except Exception:
            resid_var = np.nan
        res_record['resid_var'] = resid_var

        # 3. Compute scale-free ICC and relative variance ratio
        if not np.isnan(re_var) and not np.isnan(resid_var) and resid_var > 0:
            total_var = re_var + resid_var
            res_record['icc'] = re_var / total_var if total_var > 0 else 0.0
            res_record['rel_var'] = re_var / resid_var
        else:
            res_record['icc'] = np.nan
            res_record['rel_var'] = np.nan

        # 4. Check numerical boundary singularity
        is_singular = False
        singular_reason = None

        if np.isnan(re_var) or re_var <= 0:
            is_singular = True
            singular_reason = 'non_positive_re_var'
        elif res_record['icc'] <= ICC_BOUNDARY_TOL or res_record['rel_var'] <= REL_VAR_BOUNDARY_TOL:
            is_singular = True
            singular_reason = f'boundary_icc_le_{ICC_BOUNDARY_TOL}'
        elif hasattr(fit, 'bse') and (fit.bse.isna().any() or np.isinf(fit.bse).any()):
            is_singular = True
            singular_reason = 'singular_hessian_bse_nan'

        res_record['is_singular'] = is_singular
        res_record['singular_reason'] = singular_reason

        # Gate out non-converged or singular models from hypothesis tests
        if not converged:
            res_record['error'] = 'not_converged'
            return res_record

        if is_singular:
            res_record['error'] = singular_reason
            return res_record

        # 5. Extract Wald chi2 ANOVA terms for valid, non-singular models
        anova = fit.wald_test_terms().summary_frame()
        res_record['p_interaction'] = anova.loc[f'{group_var}:Bin', 'P>chi2']
        res_record['p_main']        = anova.loc[group_var,          'P>chi2']
        res_record['p_bin']         = anova.loc['Bin',              'P>chi2']
        return res_record

    except Exception as e:
        res_record['error'] = str(e)
        return res_record

def run_lme(df, group_var, run_covariate, features, channels):
    tasks = [(chan, feat) for chan in channels for feat in features]
    results = Parallel(n_jobs=-1, backend='loky', verbose=5)(
        delayed(fit_one)(chan, feat, df, group_var, run_covariate)
        for chan, feat in tasks
    )
    raw_df = pd.DataFrame(results)

    n_total = len(raw_df)
    n_conv = int(raw_df['converged'].sum())
    n_not_conv = n_total - n_conv
    pct_not_conv = (n_not_conv / n_total * 100) if n_total > 0 else 0.0

    n_singular = int(raw_df['is_singular'].sum())
    pct_singular = (n_singular / n_total * 100) if n_total > 0 else 0.0

    valid_mask = (raw_df['converged']) & (~raw_df['is_singular']) & raw_df['p_main'].notna() & raw_df['p_interaction'].notna()
    gate_df = raw_df[valid_mask].copy()
    n_survived = len(gate_df)
    pct_survived = (n_survived / n_total * 100) if n_total > 0 else 0.0

    print(f"    LME Fits Audit [{group_var}]: {n_survived}/{n_total} survived ({pct_survived:.1f}%) | "
          f"Singular: {n_singular} ({pct_singular:.1f}%) | "
          f"Non-converged: {n_not_conv} ({pct_not_conv:.1f}%)")

    if n_singular > 0:
        reasons = raw_df[raw_df['is_singular']]['singular_reason'].value_counts().to_dict()
        print(f"      Singularity breakdown: {reasons}")

    if gate_df.empty:
        gate_df['p_interaction_fdr'] = []
        gate_df['p_main_fdr']        = []
        return gate_df, raw_df

    gate_df['p_interaction_fdr'] = multipletests(gate_df['p_interaction'], method='fdr_bh')[1]
    gate_df['p_main_fdr']        = multipletests(gate_df['p_main'],        method='fdr_bh')[1]
    return gate_df, raw_df

def binwise_localization(df, chan, feat, group_col, bin_col='bin_epoch_num'):
    collapsed = (
        df[df['channel_name'] == chan]
        .dropna(subset=[feat])
        .groupby(['subject_id', group_col, bin_col])[feat]
        .mean()
        .reset_index()
    )
    n_groups = collapsed[group_col].nunique()
    results  = []
    for b in sorted(collapsed[bin_col].unique()):
        db     = collapsed[collapsed[bin_col] == b]
        groups = [db[db[group_col] == g][feat].values
                  for g in sorted(db[group_col].unique())]
        ns = [len(g) for g in groups]
        if min(ns) < 2 or any(g.var() == 0 for g in groups):
            p = 1.0
        elif n_groups == 2:
            _, p = ttest_ind(groups[0], groups[1], equal_var=False)
        else:
            _, p = f_oneway(*groups)
        results.append({'bin': b, 'p_raw': p, 'n_per_group': ns})

    pvals            = [r['p_raw'] for r in results]
    rej, p_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    for i in range(len(results)):
        results[i]['p_fdr']       = p_fdr[i]
        results[i]['significant'] = bool(rej[i])
    return results

def summarize_localization(passed_df, all_locs, p_col):
    summary = []
    for _, row in passed_df.iterrows():
        chan, feat = row['chan'], row['feature']
        loc      = all_locs.get((chan, feat), [])
        sig_bins = [r['bin'] for r in loc if r['significant']]
        summary.append({
            'channel':          chan,
            'feature':          feat,
            p_col:              row[p_col],
            'effect_type':      'localized' if sig_bins else 'distributed',
            'significant_bins': ','.join(map(str, sig_bins)) if sig_bins else 'none'
        })
    return pd.DataFrame(summary)

def cohens_d(x, y):
    x, y   = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    if vx == 0 or vy == 0 or nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled

# --- 6. Main pipeline — sweep over K values ---
# Adopted optimal k* per dataset (matches Table 2 / Table 5 in main_long.tex)
K_STAR_MAP = {
    'korean':   15,
    'giga':     22 if 22 in K_VALUES else 20,
    'adaptive': 20,
}

all_sweep_records    = []
all_scenario_records = []

# results_by_dataset[dataset_name] = {'k': [], 'n_main': [], 'n_inter': []}
results_by_dataset = {
    name: {'k': [], 'n_main': [], 'n_inter': []}
    for name in dataset_configs
    if dataset_configs[name].get('enabled', True)
}

for k_val in K_VALUES:
    print(f"\n{'#'*60}")
    print(f"K = {k_val}")
    print(f"{'#'*60}")

    for dataset_name, cfg in dataset_configs.items():
        if not cfg.get('enabled', True):
            continue

        # Skip K=1 for giga (large dataset: memory limit on OS)
        if dataset_name == 'giga' and k_val == 1:
            print(f"  [skip] K=1 skipped for giga (exceeds single-node memory limit)")
            continue

        print(f"\n  {'='*50}")
        print(f"  DATASET: {dataset_name.upper()}  |  K={k_val}")
        print(f"  {'='*50}")

        df_binned = get_or_create_binned_ft(
            ft_prefix     = cfg['ft_prefix'],
            binned_prefix = cfg['binned_prefix'],
            group_cols    = cfg['group_cols'],
            k             = k_val,
            dataset_name  = dataset_name,
            epoch_type    = cfg['epoch_type'],
            feature_names = cfg.get('features', P300_FEATURES),
        )
        if df_binned is None:
            continue

        # Derive run_group covariate
        if 'file_type' in df_binned.columns:
            df_binned['run_group'] = df_binned['file_type'].map(cfg['run_grouping'])
        elif 'run_type' in df_binned.columns:
            df_binned['run_group'] = df_binned['run_type'].map(cfg['run_grouping'])
        elif 'split' in df_binned.columns:
            df_binned['run_group'] = df_binned['split'].map(cfg['run_grouping'])

        # Normalize channel names
        df_binned['channel_name'] = df_binned['channel_name'].astype(str).str.casefold()

        dataset_channels = cfg.get('channels', P300_CHANNELS)
        if dataset_channels is not None:
            allowed_channels = {str(c).casefold() for c in dataset_channels}
            df_binned = df_binned[df_binned['channel_name'].isin(allowed_channels)]

        available_chans = sorted(df_binned['channel_name'].unique())

        # Derive available features
        non_feat_cols = {'subject_id','session_id','run_type','run_id','file_type',
                         'split','channel_name','epoch_type','epoch_num',
                         'bin_epoch_num','run_group'}
        dataset_features = cfg.get('features', P300_FEATURES)
        available_feats = (
            [f for f in dataset_features if f in df_binned.columns]
            if dataset_features is not None
            else [c for c in df_binned.columns if c not in non_feat_cols]
        )

        if not available_chans or not available_feats:
            print(f"  No channels/features for {dataset_name} at K={k_val}; skipping.")
            del df_binned; gc.collect()
            continue

        # Cast categoricals
        for col in ['subject_id', 'channel_name', 'run_group']:
            if col in df_binned.columns:
                df_binned[col] = df_binned[col].astype('category')
        df_binned['bin_epoch_num'] = df_binned['bin_epoch_num'].astype('category')

        # Accumulate counts across all group variables
        total_main     = 0
        total_inter    = 0
        scen_records_k = []

        for group_var, meta_dict in cfg['metadata'].items():
            meta_map = build_meta_map(meta_dict, group_var)
            df = df_binned.merge(meta_map, on='subject_id', how='left')
            df = df.dropna(subset=[group_var])
            df[group_var] = df[group_var].astype('category')

            check = (df[['subject_id', group_var]]
                     .drop_duplicates()
                     .groupby('subject_id')[group_var]
                     .nunique().max())
            if check != 1:
                print(f"    WARNING: multiple {group_var} labels per subject — skipping.")
                continue

            gate_df, raw_df = run_lme(df, group_var, 'run_group', available_feats, available_chans)

            n_main  = int((gate_df['p_main_fdr']        < 0.05).sum()) if not gate_df.empty else 0
            n_inter = int((gate_df['p_interaction_fdr'] < 0.05).sum()) if not gate_df.empty else 0
            total_main  += n_main
            total_inter += n_inter

            n_tot_scen  = len(raw_df)
            n_conv_scen = int(raw_df['converged'].sum())
            n_sing_scen = int(raw_df['is_singular'].sum())
            n_surv_scen = len(gate_df)
            pct_conv    = (n_conv_scen / n_tot_scen * 100.0) if n_tot_scen > 0 else 0.0
            pct_sing    = (n_sing_scen / n_tot_scen * 100.0) if n_tot_scen > 0 else 0.0
            pct_surv    = (n_surv_scen / n_tot_scen * 100.0) if n_tot_scen > 0 else 0.0

            valid_re  = raw_df.loc[raw_df['converged'] & ~raw_df['is_singular'], 're_var'].dropna()
            med_re    = float(valid_re.median()) if not valid_re.empty else 0.0

            valid_icc = raw_df.loc[raw_df['converged'] & ~raw_df['is_singular'], 'icc'].dropna()
            med_icc   = float(valid_icc.median()) if not valid_icc.empty else 0.0

            valid_res = raw_df.loc[raw_df['converged'] & ~raw_df['is_singular'], 'resid_var'].dropna()
            med_res   = float(valid_res.median()) if not valid_res.empty else 0.0

            scen_rec = {
                'dataset':             dataset_name,
                'k':                   k_val,
                'group_var':           group_var,
                'total_models':        n_tot_scen,
                'converged_models':    n_conv_scen,
                'converged_pct':       pct_conv,
                'singular_models':     n_sing_scen,
                'singular_pct':        pct_sing,
                'survived_models':     n_surv_scen,
                'survived_pct':        pct_surv,
                'median_re_var':       med_re,
                'median_resid_var':    med_res,
                'median_icc':          med_icc,
                'main_effects':        n_main,
                'interaction_effects': n_inter,
            }
            all_scenario_records.append(scen_rec)
            scen_records_k.append(scen_rec)

            print(f"    [{group_var}]  main={n_main}  inter={n_inter} | Conv: {pct_conv:.1f}% | Sing: {pct_sing:.1f}% | Surv: {pct_surv:.1f}%")

        if scen_records_k:
            df_k_scen       = pd.DataFrame(scen_records_k)
            pooled_conv_pct = float(df_k_scen['converged_pct'].mean())
            pooled_sing_pct = float(df_k_scen['singular_pct'].mean())
            pooled_surv_pct = float(df_k_scen['survived_pct'].mean())
            m_per_scen      = int(df_k_scen['total_models'].iloc[0])

            # Median sigma_u^2 for sex/gender scenario (as reported in Table 5)
            sex_row = df_k_scen[df_k_scen['group_var'] == 'gender']
            if not sex_row.empty:
                sex_re_var = float(sex_row['median_re_var'].iloc[0])
                sex_icc    = float(sex_row['median_icc'].iloc[0])
            else:
                sex_re_var = float(df_k_scen['median_re_var'].median())
                sex_icc    = float(df_k_scen['median_icc'].median())

            all_sweep_records.append({
                'dataset':             dataset_name,
                'k':                   k_val,
                'models_per_scenario': m_per_scen,
                'conv_pct':            pooled_conv_pct,
                'sing_pct':            pooled_sing_pct,
                'surv_pct':            pooled_surv_pct,
                're_var_sex':          sex_re_var,
                'icc_sex':             sex_icc,
                'main_effects':        total_main,
                'interaction_effects': total_inter,
            })

        results_by_dataset[dataset_name]['k'].append(k_val)
        results_by_dataset[dataset_name]['n_main'].append(total_main)
        results_by_dataset[dataset_name]['n_inter'].append(total_inter)
        print(f"  >> K={k_val} {dataset_name}: total main={total_main}  total inter={total_inter}")

        del df_binned; gc.collect()

# --- 7. Plot results ---
enabled_datasets = [
    name for name, cfg in dataset_configs.items() if cfg.get('enabled', True)
]

# Color palette: one hue per dataset, solid for main, dashed for interaction
dataset_colors = {
    'korean':   '#1f77b4',
    'giga':     '#ff7f0e',
    'adaptive': '#2ca02c',
}
# Fallback for any other dataset names
fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i, name in enumerate(enabled_datasets):
    if name not in dataset_colors:
        dataset_colors[name] = fallback_colors[i % len(fallback_colors)]

fig, ax = plt.subplots(figsize=(11, 6))

for dataset_name in enabled_datasets:
    res   = results_by_dataset[dataset_name]
    ks    = res['k']
    color = dataset_colors.get(dataset_name, '#333333')

    ax.plot(ks, res['n_main'],  color=color, linestyle='-',  marker='o',
            linewidth=2, markersize=6,
            label=f'{dataset_name} — main effects')
    ax.plot(ks, res['n_inter'], color=color, linestyle='--', marker='s',
            linewidth=2, markersize=6,
            label=f'{dataset_name} — interaction effects')

ax.set_xlabel('K (epochs per bin)', fontsize=13)
ax.set_ylabel('Number of significant effects (FDR < 0.05)', fontsize=13)
ax.set_title('Main vs Interaction Effects Across K Values', fontsize=14, fontweight='bold')
ax.set_xticks(K_VALUES)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_fig = 'k_sweep_effects_comparison_adaptive.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_fig}")
plt.close()

# ==============================================================================
# 8. Summary Statistics Reports & Publication Tables (Console, CSV, LaTeX)
# ==============================================================================

def format_sci_latex(val):
    """Format numerical variance for LaTeX tables (e.g. $1.9\\!\\times\\!10^{-12}$ or $0$)."""
    if val is None or np.isnan(val) or val <= 0:
        return "$0$"
    if val < 1e-4:
        exp = int(np.floor(np.log10(val)))
        coeff = val / (10.0 ** exp)
        return f"${coeff:.1f}\\!\\times\\!10^{{{exp}}}$"
    return f"${val:.4f}$"

def format_sci_text(val):
    """Format numerical variance for clean terminal display."""
    if val is None or np.isnan(val) or val <= 0:
        return "0"
    if val < 1e-4:
        exp = int(np.floor(np.log10(val)))
        coeff = val / (10.0 ** exp)
        return f"{coeff:.1f}e{exp}"
    return f"{val:.4f}"

if all_sweep_records:
    df_sweep = pd.DataFrame(all_sweep_records)
    df_scen  = pd.DataFrame(all_scenario_records)

    # 1. Export CSV summaries
    out_sweep_csv = "k_sweep_stability_and_effects_summary.csv"
    out_scen_csv  = "k_sweep_scenario_audit.csv"
    df_sweep.to_csv(out_sweep_csv, index=False)
    df_scen.to_csv(out_scen_csv, index=False)
    print(f"\n[Exported] Summary CSV: {out_sweep_csv}")
    print(f"[Exported] Scenario Audit CSV: {out_scen_csv}")

    # 2. Formatted Console Table: Full K-Sweep Continuum (Table 5 style)
    print("\n" + "=" * 98)
    print("TABLE 5: LABEL-BLIND K-SWEEP STABILITY AND EFFECT COUNTS (ALL 3 DATASETS)")
    print("Pooled across scenarios per dataset (* indicates adopted k* operating point)")
    print("=" * 98)
    header_t5 = f"{'Dataset':<10} {'k':<6} {'M':<6} {'Conv.%':<9} {'Sing.%':<9} {'Surv.%':<9} {'sigma2_u (sex)':<18} {'main':<7} {'int.':<6}"
    print(header_t5)
    print("-" * 98)

    for rec in all_sweep_records:
        dname     = rec['dataset'].capitalize()
        kval      = rec['k']
        is_k_star = (kval == K_STAR_MAP.get(rec['dataset']))
        k_str     = f"{kval}*" if is_k_star else str(kval)
        m_str     = str(rec['models_per_scenario'])
        conv_str  = f"{rec['conv_pct']:.1f}%"
        sing_str  = f"{rec['sing_pct']:.1f}%"
        surv_str  = f"{rec['surv_pct']:.1f}%"
        re_str    = format_sci_text(rec['re_var_sex'])
        main_str  = str(rec['main_effects'])
        int_str   = str(rec['interaction_effects'])
        print(f"{dname:<10} {k_str:<6} {m_str:<6} {conv_str:<9} {sing_str:<9} {surv_str:<9} {re_str:<18} {main_str:<7} {int_str:<6}")
    print("=" * 98)

    # 3. Formatted Console Table: Three Operating Points (Table 2 style)
    k_max = max(K_VALUES)
    op_records = []
    print("\n" + "=" * 82)
    print(f"TABLE 2: THREE OPERATING POINTS (k=1, k=k*, k=k_max={k_max})")
    print("Surviving BH-FDR (q < 0.05) effects pooled within each dataset across scenarios")
    print("=" * 82)
    print(f"{'Dataset':<12} {'k*':<6} {'k = 1':^20} {'k = k*':^20} {f'k = {k_max}':^20}")
    print(f"{'':<12} {'':<6} {'main':^10} {'int.':^10} {'main':^10} {'int.':^10} {'main':^10} {'int.':^10}")
    print("-" * 82)

    for dname, cfg in dataset_configs.items():
        if not cfg.get('enabled', True):
            continue
        k_star = K_STAR_MAP.get(dname, 15)

        # Lookup k=1
        r_k1 = df_sweep[(df_sweep['dataset'] == dname) & (df_sweep['k'] == 1)]
        if not r_k1.empty:
            k1_main = int(r_k1['main_effects'].iloc[0])
            k1_int  = int(r_k1['interaction_effects'].iloc[0])
            k1_main_str = str(k1_main)
            k1_int_str  = str(k1_int)
        else:
            k1_main = np.nan
            k1_int  = np.nan
            k1_main_str = "--"
            k1_int_str  = "--"

        # Lookup k=k*
        r_kstar = df_sweep[(df_sweep['dataset'] == dname) & (df_sweep['k'] == k_star)]
        if not r_kstar.empty:
            ks_main = int(r_kstar['main_effects'].iloc[0])
            ks_int  = int(r_kstar['interaction_effects'].iloc[0])
            ks_main_str = str(ks_main)
            ks_int_str  = str(ks_int)
        else:
            ks_main = np.nan
            ks_int  = np.nan
            ks_main_str = "--"
            ks_int_str  = "--"

        # Lookup k=k_max
        r_kmax = df_sweep[(df_sweep['dataset'] == dname) & (df_sweep['k'] == k_max)]
        if not r_kmax.empty:
            km_main = int(r_kmax['main_effects'].iloc[0])
            km_int  = int(r_kmax['interaction_effects'].iloc[0])
            km_main_str = str(km_main)
            km_int_str  = str(km_int)
        else:
            km_main = np.nan
            km_int  = np.nan
            km_main_str = "--"
            km_int_str  = "--"

        op_records.append({
            'dataset':    dname.capitalize(),
            'k_star':     k_star,
            'k1_main':    k1_main_str,
            'k1_int':     k1_int_str,
            'kstar_main': ks_main_str,
            'kstar_int':  ks_int_str,
            'kmax_main':  km_main_str,
            'kmax_int':   km_int_str,
        })

        print(f"{dname.capitalize():<12} {k_star:<6} {k1_main_str:^10} {k1_int_str:^10} {ks_main_str:^10} {ks_int_str:^10} {km_main_str:^10} {km_int_str:^10}")
    print("=" * 82)

    df_op = pd.DataFrame(op_records)
    out_op_csv = "k_sweep_operating_points_summary.csv"
    df_op.to_csv(out_op_csv, index=False)
    print(f"[Exported] Operating Points CSV: {out_op_csv}")

print("\nAll datasets complete.")
