# --- k_invariant_control_analysis.py ---
# Computes subject-level grand means directly from raw trial-level feature files (parquet_ft),
# completely bypassing any binned parquet files.
# Guaranteed 100% free of binning artifacts and k.
# Designed to be executed on your VM.

import os
import io
import gc
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

warnings.filterwarnings('ignore')

# --- 1. Azure Configuration ---
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

def download_blob(blob_path):
    buf = io.BytesIO()
    container.get_blob_client(blob_path).download_blob().readinto(buf)
    buf.seek(0)
    return buf

def list_blobs_prefix(prefix):
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]

# --- 2. Subject Metadata ---
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

# --- 3. Dataset Configs: Points directly to RAW feature extractions (parquet_ft) ---
dataset_configs = {
    'korean': {
        'enabled':        True,
        'ft_prefix':      'korean/parquet_ft',
        'epoch_type':     'target',
        'channels':       [
            'fp1', 'f7', 'f3', 'fc1', 'fc5', 'c4', 'cp2', 'fc6',
            'f8', 'f4', 'fc2', 'fp2', 'fz', 'cz', 'cp1', 'pz', 'oz', 'po4', 'p4'
        ],
        'metadata':       korean_meta,
    },
    'giga': {
        'enabled':        True,
        'ft_prefix':      'giga_preprocessed/parquet_ft',
        'epoch_type':     'non-target',
        'channels':       [
            'fc1', 'fc2', 'cz', 'c3', 'cp5', 'cp1', 'cp2', 'pz',
            'o1', 'oz', 'o2', 'c1', 'c2', 'cp3', 'cpz', 'cp4',
            'p1', 'p2', 'poz', 'c4',
        ],
        'metadata':       giga_meta,
    },
    'adaptive': {
        'enabled':        True,
        'ft_prefix':      'adaptiveP300/parquet_ft',
        'epoch_type':     'target',
        'channels':       [
            'cp1', 'cp2', 'cz', 'fp1', 'fz', 'f3', 'f4', 'fc1',
            'fc2', 'c4', 'c3',
        ],
        'metadata':       adaptive_meta,
    },
}

# --- 4. Helpers for Direct Raw Feature Aggregation ---

def build_meta_map(meta_dict, col_name):
    rows = [(subj, label) for label, subjs in meta_dict.items() for subj in subjs]
    return pd.DataFrame(rows, columns=['subject_id', col_name])

def cohens_d(x, y):
    x, y   = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    if vx == 0 or vy == 0 or nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled

def _process_one_raw_blob(blob_path, epoch_type, channels, feature_cols):
    """
    Worker: downloads one raw parquet file (trial-level), filters by epoch_type and channels,
    and aggregates sum and count for computing the exact unbinned trial grand mean.
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
        df = pd.read_parquet(buf)

        # Filter epoch_type
        if 'epoch_type' in df.columns:
            df = df[df['epoch_type'] == epoch_type]

        if df.empty:
            return None

        df['subject_id'] = df['subject_id'].astype(str).str.zfill(2)
        df['channel_name'] = df['channel_name'].astype(str).str.casefold()

        if channels is not None:
            allowed = {str(c).casefold() for c in channels}
            df = df[df['channel_name'].isin(allowed)]

        if df.empty:
            return None

        # Return subject-channel sum and count for exact weighted grand mean across files
        agg_funcs = {f: ['sum', 'count'] for f in feature_cols if f in df.columns}
        res = df.groupby(['subject_id', 'channel_name']).agg(agg_funcs)
        return res
    except Exception as e:
        print(f"  [error] {blob_path}: {e}")
        return None

def compute_raw_subject_grand_means(ft_prefix, epoch_type, channels):
    """
    Computes true subject-level grand means directly from raw trial-level feature files (parquet_ft).
    Completely bypasses any binning procedures or bin files.
    """
    blob_paths = [b for b in list_blobs_prefix(ft_prefix) if b.endswith('.parquet')]
    if not blob_paths:
        print(f"No raw feature parquet files found under {ft_prefix}")
        return None, []

    # Infer features from the first blob
    buf_first = download_blob(blob_paths[0])
    df_first = pd.read_parquet(buf_first)
    non_feat = {'subject_id','session_id','run_type','run_id','file_type',
                'split','channel_name','epoch_type','epoch_num',
                'bin_epoch_num','run_group'}
    feature_cols = [c for c in df_first.columns if c not in non_feat]
    del df_first, buf_first

    print(f"Found {len(blob_paths)} raw trial-level parquets under {ft_prefix}")
    print(f"Aggregating {len(feature_cols)} features directly across all trials (parallel loky)...")

    chunks = Parallel(n_jobs=-1, backend='loky', verbose=5)(
        delayed(_process_one_raw_blob)(b, epoch_type, channels, feature_cols)
        for b in blob_paths
    )
    chunks = [c for c in chunks if c is not None and not c.empty]

    if not chunks:
        return None, []

    # Combine sums and counts across chunks
    total_agg = pd.concat(chunks).groupby(level=['subject_id', 'channel_name']).sum()

    # Compute grand mean = total_sum / total_count
    records = []
    for (subj, chan), row in total_agg.iterrows():
        entry = {'subject_id': subj, 'channel_name': chan}
        for f in feature_cols:
            sum_val = row[(f, 'sum')]
            cnt_val = row[(f, 'count')]
            entry[f] = sum_val / cnt_val if cnt_val > 0 else np.nan
        records.append(entry)

    df_grand_means = pd.DataFrame(records)
    return df_grand_means, feature_cols

def run_k_invariant_analysis(df_subj_means, group_var, channels, features):
    groups = sorted(df_subj_means[group_var].dropna().unique())
    n_groups = len(groups)
    results = []

    for chan in channels:
        df_c = df_subj_means[df_subj_means['channel_name'] == chan]
        for feat in features:
            series_by_group = [
                df_c[df_c[group_var] == g][feat].dropna().values
                for g in groups
            ]
            if any(len(s) < 2 for s in series_by_group):
                continue

            if n_groups == 2:
                g0, g1 = series_by_group[0], series_by_group[1]
                stat, p_val = ttest_ind(g1, g0, equal_var=False)
                d_val = cohens_d(g1, g0)
            else:
                stat, p_val = f_oneway(*series_by_group)
                d_val = np.nan

            results.append({
                'channel': chan,
                'feature': feat,
                'test_stat': stat,
                'p_raw': p_val,
                'cohens_d': d_val,
                'n_per_group': [len(s) for s in series_by_group]
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['p_fdr'] = multipletests(res_df['p_raw'], method='fdr_bh')[1]
        res_df['significant'] = res_df['p_fdr'] < 0.05
    return res_df

# --- 5. Main Execution ---
all_summaries = []

for dataset_name, cfg in dataset_configs.items():
    if not cfg.get('enabled', True):
        continue

    print(f"\n{'='*70}")
    print(f"AIR-TIGHT K-INVARIANT CONTROL ANALYSIS: {dataset_name.upper()}")
    print("Computing grand means directly from trial-level parquet_ft (NO BINNING)")
    print(f"{'='*70}")

    df_subj_means, features = compute_raw_subject_grand_means(
        ft_prefix   = cfg['ft_prefix'],
        epoch_type  = cfg['epoch_type'],
        channels    = cfg['channels'],
    )

    if df_subj_means is None or df_subj_means.empty:
        print(f"No data extracted for {dataset_name}. Skipping.")
        continue

    available_chans = sorted(df_subj_means['channel_name'].unique())
    n_subjects = df_subj_means['subject_id'].nunique()
    print(f"\nSuccessfully aggregated raw trials: {n_subjects} subjects, {len(available_chans)} channels, {len(features)} features.")

    # Save the unbinned subject grand means table to disk
    subj_means_file = f"{dataset_name}_raw_subject_grand_means.csv"
    df_subj_means.to_csv(subj_means_file, index=False)
    print(f"Saved unbinned subject grand means to: {subj_means_file}")

    # Perform group comparisons for each metadata variable
    for group_var, meta_dict in cfg['metadata'].items():
        meta_map = build_meta_map(meta_dict, group_var)
        df_group = df_subj_means.merge(meta_map, on='subject_id', how='left').dropna(subset=[group_var])

        res_df = run_k_invariant_analysis(df_group, group_var, available_chans, features)
        sig_df = res_df[res_df['significant']].copy()

        print(f"\n  [Variable: {group_var}]")
        print(f"    Total tested combinations : {len(res_df)}")
        print(f"    Significant (FDR < 0.05)   : {len(sig_df)}")

        if not sig_df.empty:
            cols_show = ['channel', 'feature', 'test_stat', 'p_raw', 'p_fdr']
            if 'cohens_d' in sig_df.columns and not sig_df['cohens_d'].isna().all():
                cols_show.append('cohens_d')
            print(sig_df[cols_show].to_string(index=False))

        res_df['dataset'] = dataset_name
        res_df['group_var'] = group_var
        all_summaries.append(res_df)

    del df_subj_means
    gc.collect()

if all_summaries:
    final_table = pd.concat(all_summaries, ignore_index=True)
    out_csv = "k_invariant_control_results_raw.csv"
    final_table.to_csv(out_csv, index=False)
    print(f"\n results exported to: {out_csv}")
    print("Report this table alongside Table 6 to demonstrate unbinned robustness.")
