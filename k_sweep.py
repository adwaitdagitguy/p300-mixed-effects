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

# --- 2. User-defined parameters & K sweep ---
K_SWEEP       = [1] + list(range(5, 51, 5))  # [1, 5, 10, 15, ..., 50]
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
        'k':              5,
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
        'enabled':         False,
        'k':              5,
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
        'k':              5,
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

def fit_one(chan, feat, df, group_var, run_covariate):
    warnings.simplefilter('ignore')
    df_chan = df[df['channel_name'] == chan]
    cols    = ['subject_id', group_var, run_covariate, 'bin_epoch_num', feat]
    d       = df_chan[cols].dropna(subset=[feat]).copy()
    d       = d.rename(columns={
        'subject_id': 'Subject', 'bin_epoch_num': 'Bin', feat: 'Value'
    })
    d['Bin'] = d['Bin'].astype('category')

    try:
        formula = f"Value ~ C({group_var}, Sum) * C(Bin, Sum) + {run_covariate}"
        model   = smf.mixedlm(formula, d, groups=d['Subject'])
        fit     = model.fit(reml=True, method='lbfgs', maxiter=500)

        converged = getattr(fit, 'converged', True)
        if not converged:
            fit = model.fit(reml=True, method='powell', maxiter=500)
            converged = getattr(fit, 'converged', True)

        # Extract random-intercept variance (sigma^2_subject)
        # In statsmodels MixedLM, random effects variance for intercept is in cov_re.iloc[0, 0]
        sigma2_subj = np.nan
        try:
            if hasattr(fit, 'cov_re') and fit.cov_re is not None:
                sigma2_subj = float(fit.cov_re.iloc[0, 0])
        except Exception:
            sigma2_subj = np.nan

        # Singular/boundary fit definition: random-intercept variance < 1e-6
        is_singular = bool(not np.isnan(sigma2_subj) and sigma2_subj < 1e-6)

        anova   = fit.wald_test_terms().summary_frame()
        p_inter = anova.loc[f'C({group_var}, Sum):C(Bin, Sum)', 'P>chi2'] if f'C({group_var}, Sum):C(Bin, Sum)' in anova.index else np.nan
        p_main  = anova.loc[f'C({group_var}, Sum)',             'P>chi2'] if f'C({group_var}, Sum)' in anova.index else np.nan
        p_bin   = anova.loc['C(Bin, Sum)',                      'P>chi2'] if 'C(Bin, Sum)' in anova.index else np.nan

        return {
            'chan': chan, 'feature': feat,
            'p_interaction': float(p_inter) if not np.isnan(p_inter) else None,
            'p_main':        float(p_main)  if not np.isnan(p_main)  else None,
            'p_bin':         float(p_bin)   if not np.isnan(p_bin)   else None,
            'converged':     bool(converged),
            'sigma2_subj':   float(sigma2_subj) if not np.isnan(sigma2_subj) else np.nan,
            'is_singular':   is_singular,
            'error':         None
        }
    except Exception as e:
        return {
            'chan': chan, 'feature': feat,
            'p_interaction': None, 'p_main': None, 'p_bin': None,
            'converged': False, 'sigma2_subj': np.nan, 'is_singular': False,
            'error': str(e)
        }

def run_null_simulation_k1(n_subjects, n_bins, n_reps=50, seed=42):
    """
    Simulate null data (no Group effect, no GxBin interaction) matching the k=1 dimension
    to compute Group x Bin type-I / rejection rate under the null.
    """
    rng = np.random.default_rng(seed)
    p_null_inter = []
    p_null_group = []

    for rep in range(n_reps):
        records = []
        for s in range(n_subjects):
            subj_id = f"s{s:02d}"
            group_val = s % 2  # balanced 2 groups
            subj_intercept = rng.normal(0, 0.3)
            # k=1 trial-level values
            vals = rng.normal(subj_intercept, 1.0, size=n_bins)
            for b_idx, val in enumerate(vals):
                records.append({
                    'Subject': subj_id,
                    'group': group_val,
                    'Bin': b_idx + 1,
                    'Value': val
                })
        df_sim = pd.DataFrame(records)
        df_sim['group'] = df_sim['group'].astype('category')
        df_sim['Bin']   = df_sim['Bin'].astype('category')

        try:
            model = smf.mixedlm("Value ~ C(group, Sum) * C(Bin, Sum)", df_sim, groups=df_sim['Subject'])
            fit = model.fit(reml=True, method='lbfgs', maxiter=300)
            anova = fit.wald_test_terms().summary_frame()
            if 'C(group, Sum):C(Bin, Sum)' in anova.index:
                p_null_inter.append(float(anova.loc['C(group, Sum):C(Bin, Sum)', 'P>chi2']))
            if 'C(group, Sum)' in anova.index:
                p_null_group.append(float(anova.loc['C(group, Sum)', 'P>chi2']))
        except Exception:
            continue

    rej_inter = float(np.mean(np.array(p_null_inter) < 0.05)) if p_null_inter else np.nan
    rej_group = float(np.mean(np.array(p_null_group) < 0.05)) if p_null_group else np.nan
    return rej_inter, rej_group, len(p_null_inter)

def run_lme(df, group_var, run_covariate, features, channels):
    tasks = [(chan, feat) for chan in channels for feat in features]
    n_attempted = len(tasks)

    results = Parallel(n_jobs=-1, backend='loky', verbose=5)(
        delayed(fit_one)(chan, feat, df, group_var, run_covariate)
        for chan, feat in tasks
    )
    raw_df = pd.DataFrame(results)

    # 1. Fit counts and percentages
    n_converged = int(raw_df['converged'].sum())
    pct_converged = (n_converged / n_attempted) * 100 if n_attempted > 0 else 0.0
    n_failed = n_attempted - n_converged
    pct_failed = (n_failed / n_attempted) * 100 if n_attempted > 0 else 0.0

    # 2. Random-intercept variance sigma^2_subject (median, IQR)
    valid_sigma = raw_df['sigma2_subj'].dropna()
    if not valid_sigma.empty:
        sigma_median = float(valid_sigma.median())
        sigma_q25    = float(valid_sigma.quantile(0.25))
        sigma_q75    = float(valid_sigma.quantile(0.75))
        sigma_iqr    = sigma_q75 - sigma_q25
    else:
        sigma_median, sigma_q25, sigma_q75, sigma_iqr = np.nan, np.nan, np.nan, np.nan

    # 3. Singular / boundary fits (< 10^-6)
    n_singular = int(raw_df['is_singular'].sum())
    pct_singular = (n_singular / n_attempted) * 100 if n_attempted > 0 else 0.0

    # Print summary of LME fits and stability metrics
    print(f"\n    {'='*65}")
    print(f"    LME FIT STABILITY & CONVERGENCE SUMMARY ({group_var})")
    print(f"    {'='*65}")
    print(f"    • Number of attempted LME fits : {n_attempted}")
    print(f"    • Converged / valid fits       : {n_converged} ({pct_converged:.1f}%)")
    print(f"    • Failed fits                  : {n_failed} ({pct_failed:.1f}%)")
    print(f"    • Estimated sigma^2_subject    : Median = {sigma_median:.6f}, IQR = {sigma_iqr:.6f} [Q25={sigma_q25:.6f}, Q75={sigma_q75:.6f}]")
    print(f"    • Singular/boundary fits (<1e-6): {n_singular} ({pct_singular:.1f}%)")

    # 4. Group x Bin type-I / rejection rate from null simulation
    n_subjects = int(df['subject_id'].nunique())
    n_bins_sample = int(df['bin_epoch_num'].nunique())
    print(f"    • Running null simulation (n_subjs={n_subjects}, n_bins={n_bins_sample}) for type-I error baseline...")
    null_rej_inter, null_rej_group, valid_sim_reps = run_null_simulation_k1(
        n_subjects=n_subjects,
        n_bins=min(n_bins_sample, 50),  # cap at 50 bins for simulation runtime
        n_reps=50,
        seed=42
    )
    print(f"    • Group x Bin type-I rejection rate (null simulation) : {null_rej_inter:.3f} (based on {valid_sim_reps} valid reps)")
    print(f"    • Group main effect type-I rejection rate (null sim)  : {null_rej_group:.3f}")
    print(f"    {'='*65}")

    # Process significant hits
    gate_df = raw_df.dropna(subset=['p_interaction', 'p_main']).copy()
    if not gate_df.empty:
        gate_df['p_interaction_fdr'] = multipletests(gate_df['p_interaction'], method='fdr_bh')[1]
        gate_df['p_main_fdr']        = multipletests(gate_df['p_main'],        method='fdr_bh')[1]
    else:
        gate_df['p_interaction_fdr'] = []
        gate_df['p_main_fdr']        = []

    # Attach summary stats to gate_df metadata
    gate_df.attrs['n_attempted'] = n_attempted
    gate_df.attrs['n_converged'] = n_converged
    gate_df.attrs['pct_converged'] = pct_converged
    gate_df.attrs['n_failed'] = n_failed
    gate_df.attrs['pct_failed'] = pct_failed
    gate_df.attrs['sigma_median'] = sigma_median
    gate_df.attrs['sigma_iqr'] = sigma_iqr
    gate_df.attrs['n_singular'] = n_singular
    gate_df.attrs['pct_singular'] = pct_singular
    gate_df.attrs['null_rej_inter'] = null_rej_inter

    return gate_df

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

# --- 6. Main pipeline: K sweep continuum [1, 5, 10, ..., 50] ---
all_sweep_records = []

for k_val in K_SWEEP:
    print(f"\n{'#'*75}")
    print(f"K SWEEP ITERATION: K = {k_val}")
    print(f"{'#'*75}")

    for dataset_name, cfg in dataset_configs.items():
        if not cfg.get('enabled', True):
            continue

        # Skip K=1 for giga (large dataset)
        if dataset_name == 'giga' and k_val == 1:
            print(f"\nSkipping K=1 for GIGA (dataset is too large).")
            continue

        print(f"\n{'='*65}")
        print(f"DATASET: {dataset_name.upper()} | K = {k_val}")
        print(f"{'='*65}")

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

        # --- Log bins per subject per run at this k to bin.log ---
        bin_cols = [col for col in cfg['group_cols'] if col in df_binned.columns]
        bins_per_subj_run = (
            df_binned.groupby(bin_cols)['bin_epoch_num']
            .nunique()
            .reset_index(name='n_bins')
        )

        with open("bin.log", "a") as f_log:
            f_log.write(f"\n{'='*70}\n")
            f_log.write(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write(f"Dataset     : {dataset_name.upper()}\n")
            f_log.write(f"K (bin size): {k_val}\n")
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
        print(f"  [bin.log] Logged bins per subject per run for {dataset_name} (K={k_val}) to bin.log")

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

        # Cast categoricals
        for col in ['subject_id', 'channel_name', 'run_group']:
            if col in df_binned.columns:
                df_binned[col] = df_binned[col].astype('category')
        df_binned['bin_epoch_num'] = df_binned['bin_epoch_num'].astype('category')

        # --- LME for each grouping variable ---
        for group_var, meta_dict in cfg['metadata'].items():
            print(f"\n  --- Group variable: {group_var} (K={k_val}) ---")

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

            gate_df = run_lme(df, group_var, 'run_group', available_feats, available_chans)

            passed_main  = gate_df[gate_df['p_main_fdr']        < 0.05]
            passed_inter = gate_df[gate_df['p_interaction_fdr'] < 0.05]

            n_sig_main  = len(passed_main)
            n_sig_inter = len(passed_inter)

            # Record stability and fit stats across the k sweep
            all_sweep_records.append({
                'dataset':          dataset_name,
                'k':                k_val,
                'group_var':        group_var,
                'n_attempted':      gate_df.attrs.get('n_attempted', 0),
                'n_converged':      gate_df.attrs.get('n_converged', 0),
                'pct_converged':    gate_df.attrs.get('pct_converged', 0.0),
                'n_failed':         gate_df.attrs.get('n_failed', 0),
                'pct_failed':       gate_df.attrs.get('pct_failed', 0.0),
                'sigma2_median':    gate_df.attrs.get('sigma_median', np.nan),
                'sigma2_iqr':       gate_df.attrs.get('sigma_iqr', np.nan),
                'n_singular':       gate_df.attrs.get('n_singular', 0),
                'pct_singular':     gate_df.attrs.get('pct_singular', 0.0),
                'null_rej_gxbin':   gate_df.attrs.get('null_rej_inter', np.nan),
                'n_sig_main':       n_sig_main,
                'n_sig_inter':      n_sig_inter,
            })

            print(f"\n    Significant main effects (FDR < 0.05): {n_sig_main}")
            if not passed_main.empty:
                print(passed_main[['chan','feature','p_main_fdr']].to_string(index=False))

            print(f"\n    Significant interactions ({group_var} × Bin, FDR < 0.05): {n_sig_inter}")
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

        del df_binned; gc.collect()

# ============================================================
# 7. Final Consolidated Summary Report Across K Sweep Continuum
# ============================================================
if all_sweep_records:
    df_all_sweep = pd.DataFrame(all_sweep_records)
    out_all_csv = "k_sweep_stability_and_effects_summary.csv"
    df_all_sweep.to_csv(out_all_csv, index=False)

    print(f"\n{'='*95}")
    print("FINAL CONSOLIDATED K-SWEEP SUMMARY REPORT (CONTINUUM K = [1, 5, 10, ..., 50])")
    print(f"{'='*95}")
    cols_display = [
        'dataset', 'k', 'group_var', 'n_converged', 'pct_converged',
        'sigma2_median', 'sigma2_iqr', 'n_singular', 'pct_singular',
        'null_rej_gxbin', 'n_sig_main', 'n_sig_inter'
    ]
    print(df_all_sweep[cols_display].to_string(index=False))
    print(f"\nDetailed sweep results exported to: {out_all_csv}")
    print(f"{'='*95}")

print("\nAll datasets and k sweeps complete.")
