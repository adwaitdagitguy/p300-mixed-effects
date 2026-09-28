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
from patsy import Sum
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
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

# --- 2. User-defined parameters ---
K             = 22
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
        'k':              15,
        'ft_prefix':      'korean/parquet_ft',
        'binned_prefix':  'korean/parquet_binned_both',
        'epoch_type':     'target',
        'channels':       [
            'fp1', 'f7', 'f3', 'fc1', 'fc5', 'c4', 'cp2', 'fc6',
            'f8', 'f4', 'fc2', 'fp2', 'fz', 'cz', 'cp1', 'pz', 'oz', 'po4', 'p4'
        ],
        'features':       None,
        'group_cols':     ['subject_id', 'run_type', 'run_id'],
        'run_covariate':  'run_group',
        'run_grouping':   {'train': 'train', 'test': 'test'},
        'metadata':       korean_meta,
    },
    'giga': {
        'enabled':         True,
        'k':              22,
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
        'enabled':        True,
        'k':              20,
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
        formula = f"Value ~ C({group_var}, Sum) * C(Bin, Sum) + {run_covariate}"
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
        res_record['p_interaction'] = anova.loc[f'C({group_var}, Sum):C(Bin, Sum)', 'P>chi2']
        res_record['p_main']        = anova.loc[f'C({group_var}, Sum)',             'P>chi2']
        res_record['p_bin']         = anova.loc['C(Bin, Sum)',                      'P>chi2']
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

    # Recalculate FDR on valid, non-singular surviving fits
    if not gate_df.empty:
        gate_df['p_interaction_fdr'] = multipletests(gate_df['p_interaction'], method='fdr_bh')[1]
        gate_df['p_main_fdr']        = multipletests(gate_df['p_main'],        method='fdr_bh')[1]
    else:
        gate_df['p_interaction_fdr'] = []
        gate_df['p_main_fdr']        = []

    return gate_df, raw_df

def plot_re_var_distribution(dataset_fits_df, dataset_name, out_dir="."):
    """
    Plots the empirical distribution of random-intercept variance (re_var),
    residual variance, and Intraclass Correlation Coefficient (ICC)
    per dataset across all fitted models.
    Highlights numerical boundary / singularity thresholds.
    """
    if dataset_fits_df.empty:
        print(f"  [plot] No fit records to plot for {dataset_name}.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Log10(re_var) distribution
    valid_re = dataset_fits_df['re_var'].dropna()
    valid_re_pos = valid_re[valid_re > 0]

    ax0 = axes[0]
    if not valid_re_pos.empty:
        log_re = np.log10(valid_re_pos)
        ax0.hist(log_re, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
        med_val = np.median(log_re)
        ax0.axvline(med_val, color='red', linestyle='--', linewidth=1.5,
                    label=f'Median: 10^{med_val:.2f}')
        ax0.set_xlabel(r'$\log_{10}(\sigma_u^2)$ (Random Intercept Variance)', fontsize=11)
        ax0.set_ylabel('Count of Models', fontsize=11)
        ax0.set_title(f'{dataset_name.upper()}: Random Intercept Variance', fontsize=12, fontweight='bold')
        ax0.legend(loc='upper right')
        ax0.grid(True, linestyle=':', alpha=0.6)
    else:
        ax0.text(0.5, 0.5, 'No positive re_var', ha='center', va='center')
        ax0.set_title(f'{dataset_name.upper()}: Random Intercept Variance')

    # 2. Scale-free ICC distribution
    ax1 = axes[1]
    valid_icc = dataset_fits_df['icc'].dropna()
    if not valid_icc.empty:
        singular_mask = dataset_fits_df['is_singular']
        icc_valid = dataset_fits_df.loc[~singular_mask, 'icc'].dropna()
        icc_sing = dataset_fits_df.loc[singular_mask, 'icc'].dropna()

        ax1.hist([icc_valid, icc_sing], bins=30, stacked=True,
                 color=['#2ca02c', '#d62728'], label=['Non-singular', 'Singular (Boundary)'],
                 edgecolor='black', alpha=0.75)
        ax1.axvline(ICC_BOUNDARY_TOL, color='black', linestyle=':', linewidth=2,
                    label=f'Boundary Tolerance ({ICC_BOUNDARY_TOL:.1e})')
        ax1.set_xlabel(r'ICC $= \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\varepsilon^2}$ (Scale-Free)', fontsize=11)
        ax1.set_ylabel('Count of Models', fontsize=11)
        ax1.set_title(f'{dataset_name.upper()}: Scale-Free ICC Distribution', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, 'No valid ICC', ha='center', va='center')
        ax1.set_title(f'{dataset_name.upper()}: ICC Distribution')

    # 3. Model Scale vs Random Intercept: log10(re_var) vs log10(resid_var)
    ax2 = axes[2]
    scatter_df = dataset_fits_df.dropna(subset=['re_var', 'resid_var'])
    scatter_df = scatter_df[(scatter_df['re_var'] > 0) & (scatter_df['resid_var'] > 0)]
    if not scatter_df.empty:
        non_sing = scatter_df[~scatter_df['is_singular']]
        sing = scatter_df[scatter_df['is_singular']]

        ax2.scatter(np.log10(non_sing['resid_var']), np.log10(non_sing['re_var']),
                    color='#2ca02c', alpha=0.6, edgecolors='none', s=25, label='Non-singular')
        if not sing.empty:
            ax2.scatter(np.log10(sing['resid_var']), np.log10(sing['re_var']),
                        color='#d62728', alpha=0.8, edgecolors='black', s=35, marker='x', label='Singular (Boundary)')

        min_res = np.log10(scatter_df['resid_var'].min())
        max_res = np.log10(scatter_df['resid_var'].max())
        res_line = np.linspace(min_res, max_res, 100)
        boundary_line = res_line + np.log10(REL_VAR_BOUNDARY_TOL)
        ax2.plot(res_line, boundary_line, color='black', linestyle=':', linewidth=1.5,
                 label=r'Boundary Line ($\sigma_u^2 = 10^{-5}\sigma_\varepsilon^2$)')

        ax2.set_xlabel(r'$\log_{10}(\sigma_\varepsilon^2)$ (Residual Variance / Scale)', fontsize=11)
        ax2.set_ylabel(r'$\log_{10}(\sigma_u^2)$ (Subject Variance)', fontsize=11)
        ax2.set_title(f'{dataset_name.upper()}: Subject vs Residual Variance', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle=':', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'No positive variance pairs', ha='center', va='center')
        ax2.set_title(f'{dataset_name.upper()}: Random vs Residual Variance')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset_name}_re_var_distribution.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [plot saved] Saved re_var distribution plot to: {out_path}")

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

# --- 6. Main pipeline ---
for dataset_name, cfg in dataset_configs.items():
    if not cfg.get('enabled', True):
        print(f"\nSkipping disabled dataset: {dataset_name.upper()}")
        continue

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    curr_k = cfg.get('k', K)
    df_binned = get_or_create_binned_ft(
        ft_prefix     = cfg['ft_prefix'],
        binned_prefix = cfg['binned_prefix'],
        group_cols    = cfg['group_cols'],
        k             = curr_k,
        dataset_name  = dataset_name,
        epoch_type    = cfg['epoch_type'],
        feature_names = cfg.get('features', P300_FEATURES),
    )
    if df_binned is None:
        continue

    # --- Log bins per subject per run at this k to bin_corrected.log ---
    bin_cols = [col for col in cfg['group_cols'] if col in df_binned.columns]
    bins_per_subj_run = (
        df_binned.groupby(bin_cols)['bin_epoch_num']
        .nunique()
        .reset_index(name='n_bins')
    )

    with open("bin_corrected.log", "a") as f_log:
        f_log.write(f"\n{'='*70}\n")
        f_log.write(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_log.write(f"Dataset     : {dataset_name.upper()}\n")
        f_log.write(f"K (bin size): {curr_k}\n")
        f_log.write(f"Group cols  : {', '.join(bin_cols)}\n")
        f_log.write(f"{'-'*70}\n")
        f_log.write("Summary statistics of bins per subject/run:\n")
        f_log.write(f"  Min bins : {bins_per_subj_run['n_bins'].min()}\n")
        f_log.write(f"  Max bins : {bins_per_subj_run['n_bins'].max()}\n")
        f_log.write(f"  Mean bins: {bins_per_subj_run['n_bins'].mean():.2f}\n")
        f_log.write(f"  Total subject-run units: {len(bins_per_subj_run)}\n")
        f_log.write(f"{'-'*70}\n")
        f_log.write("Detailed breakdown (bins per subject per run):\n")
        f_log.write(bins_per_subj_run.to_string(index=False))
        f_log.write(f"\n{'='*70}\n")
    print(f"  [bin.log] Logged bins per subject per run for {dataset_name} (K={curr_k}) to bin.log")

    # Derive run_group covariate
    if 'file_type' in df_binned.columns:
        df_binned['run_group'] = df_binned['file_type'].map(cfg['run_grouping'])
    elif 'run_type' in df_binned.columns:
        df_binned['run_group'] = df_binned['run_type'].map(cfg['run_grouping'])
    elif 'split' in df_binned.columns:
        df_binned['run_group'] = df_binned['split'].map(cfg['run_grouping'])

    # Normalize channel names to a common case for case-insensitive matching.
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
        print(f"  No channels/features selected for {dataset_name}; skipping dataset.")
        del df_binned; gc.collect()
        continue

    # Cast categoricals (bin_epoch_num left as-is per caller's request)
    for col in ['subject_id', 'channel_name', 'run_group']:
        if col in df_binned.columns:
            df_binned[col] = df_binned[col].astype('category')
    df_binned['bin_epoch_num'] = df_binned['bin_epoch_num'].astype('category')

    dataset_fits_list = []

    # --- LME for each grouping variable ---
    for group_var, meta_dict in cfg['metadata'].items():
        print(f"\n  --- Group variable: {group_var} ---")

        meta_map = build_meta_map(meta_dict, group_var)
        df = df_binned.merge(meta_map, on='subject_id', how='left')

        n_missing = df[group_var].isna().sum()
        if n_missing > 0:
            print(f"    WARNING: {n_missing} rows with missing {group_var}, dropping.")
        df = df.dropna(subset=[group_var])
        df[group_var] = df[group_var].astype('category')

        check = (df[['subject_id', group_var]]
                 .drop_duplicates()
                 .groupby('subject_id')[group_var]
                 .nunique().max())
        assert check == 1, f"Multiple {group_var} labels per subject!"

        print(f"    Group distribution: "
              f"{df[['subject_id', group_var]].drop_duplicates()[group_var].value_counts().to_dict()}")

        gate_df, raw_df = run_lme(df, group_var, 'run_group', available_feats, available_chans)
        raw_df['dataset'] = dataset_name
        dataset_fits_list.append(raw_df)

        passed_main  = gate_df[gate_df['p_main_fdr']        < 0.05]
        passed_inter = gate_df[gate_df['p_interaction_fdr'] < 0.05]

        print(f"\n    Significant main effects (FDR < 0.05): {len(passed_main)}")
        if not passed_main.empty:
            print(passed_main[['chan','feature','p_main_fdr']].to_string(index=False))

        print(f"\n    Significant interactions ({group_var} × Bin, FDR < 0.05): {len(passed_inter)}")
        if not passed_inter.empty:
            print(passed_inter[['chan','feature','p_interaction_fdr']].to_string(index=False))

        # Cohen's d for 2-group main effects
        if df[group_var].nunique() == 2 and not passed_main.empty:
            print(f"\n    Cohen's d for significant main effects:")
            subject_group = df[['subject_id', group_var]].drop_duplicates()
            for _, row in passed_main.iterrows():
                chan, feat = row['chan'], row['feature']
                subj_level = (df[df['channel_name'] == chan][['subject_id', feat]]
                              .dropna()
                              .groupby('subject_id')[feat]
                              .mean()
                              .reset_index()
                              .merge(subject_group, on='subject_id'))
                cats = sorted(subj_level[group_var].unique())
                g0   = subj_level[subj_level[group_var] == cats[0]][feat]
                g1   = subj_level[subj_level[group_var] == cats[1]][feat]
                d    = cohens_d(g1.values, g0.values)
                print(f"      {chan} | {feat} | d={d:.3f} | p_fdr={row['p_main_fdr']:.4f}")

        # Post-hoc bin localization for significant interactions
        if not passed_inter.empty:
            all_locs = {}
            for _, row in passed_inter.iterrows():
                chan, feat = row['chan'], row['feature']
                all_locs[(chan, feat)] = binwise_localization(df, chan, feat, group_var)

            summary_df = summarize_localization(passed_inter, all_locs, 'p_interaction_fdr')
            print(f"\n    Bin localization summary:")
            print(summary_df.sort_values('p_interaction_fdr').to_string(index=False))

    # --- Dataset-level Fit & Singularity Audit and Distribution Plotting ---
    if dataset_fits_list:
        dataset_fits_df = pd.concat(dataset_fits_list, ignore_index=True)

        audit_csv_path = f"{dataset_name}_lme_fits_audit.csv"
        dataset_fits_df.to_csv(audit_csv_path, index=False)
        print(f"\n  [audit saved] Saved complete fits audit table ({len(dataset_fits_df)} rows) to: {audit_csv_path}")

        total_dset_fits = len(dataset_fits_df)
        total_dset_conv = int(dataset_fits_df['converged'].sum())
        total_dset_sing = int(dataset_fits_df['is_singular'].sum())
        total_dset_surv = int(((dataset_fits_df['converged']) & (~dataset_fits_df['is_singular']) & dataset_fits_df['p_main'].notna()).sum())

        print(f"\n  {'='*65}")
        print(f"  DATASET FIT & SINGULARITY AUDIT: {dataset_name.upper()}")
        print(f"  {'='*65}")
        print(f"  Total models fitted      : {total_dset_fits}")
        print(f"  Converged models         : {total_dset_conv} ({total_dset_conv/total_dset_fits*100:.2f}%)")
        print(f"  Singular (boundary) fits : {total_dset_sing} ({total_dset_sing/total_dset_fits*100:.2f}%)")
        print(f"  Valid retained models    : {total_dset_surv} ({total_dset_surv/total_dset_fits*100:.2f}%)")
        print(f"  {'='*65}\n")

        plot_re_var_distribution(dataset_fits_df, dataset_name)

    del df_binned; gc.collect()

print("\nAll datasets complete.")
