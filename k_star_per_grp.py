import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

warnings.filterwarnings('ignore')

# =============================================================
# 0. Azure Configuration
# =============================================================
AZURE_TENANT_ID         = os.environ['AZURE_TENANT_ID']
AZURE_CLIENT_ID         = os.environ['AZURE_CLIENT_ID']
AZURE_CLIENT_SECRET     = os.environ['AZURE_CLIENT_SECRET']
AZURE_STORAGE_ACCOUNT   = os.environ['AZURE_STORAGE_ACCOUNT']
AZURE_STORAGE_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']

PLOT_DIR = os.getcwd()

def get_container_client():
    cred = ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
    )
    svc = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=cred,
    )
    return svc.get_container_client(AZURE_STORAGE_CONTAINER)

container = get_container_client()

def download_blob_df(blob_path):
    buf = io.BytesIO()
    container.get_blob_client(blob_path).download_blob().readinto(buf)
    buf.seek(0)
    return pd.read_parquet(buf)

def load_all_under_prefix(prefix):
    blobs = [b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.endswith('.parquet')]
    if not blobs:
        return None
    print(f"  Downloading {len(blobs)} parquet files from '{prefix}'...")
    dfs = [download_blob_df(b) for b in blobs]
    return pd.concat(dfs, ignore_index=True)

def load_atpv_all_k(atpv_prefix, k_range):
    frames = []
    for k in k_range:
        prefix = f"{atpv_prefix}/k_{k}/"
        df_k = load_all_under_prefix(prefix)
        if df_k is not None and not df_k.empty:
            if 'k' not in df_k.columns:
                df_k['k'] = k
            frames.append(df_k)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# =============================================================
# 1. Metadata Mappings for Demographic Groups
# =============================================================
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
}

dataset_configs = {
    'korean': {
        'atpv_prefix': 'korean/parquet_meta/ddof_1',
        'snr_prefix':  'korean/parquet_snr',
        'metadata':    korean_meta,
    },
    'giga': {
        'atpv_prefix': 'giga_preprocessed/parquet_meta/ddof_1',
        'snr_prefix':  'giga_preprocessed/parquet_snr',
        'metadata':    giga_meta,
    },
    'adaptive': {
        'atpv_prefix': 'adaptiveP300/parquet_meta/ddof_1',
        'snr_prefix':  'adaptiveP300/parquet_snr',
        'metadata':    adaptive_meta,
    },
}

K_RANGE = [1, 2, 3, 4] + list(range(5, 51, 5))

# =============================================================
# 2. Mathematical Estimation Rules for k*(g)
# =============================================================
def estimate_k_star_snr(curve_df, threshold=0.85, kmax=50):
    """Smallest k where SNR(k) >= threshold * SNR(k_ref)."""
    s = curve_df.set_index('k')['snr'].sort_index()
    s = s[s.index <= kmax].dropna()
    if s.empty:
        return np.nan
    ref_k = s.index.max()
    ref_val = s.loc[ref_k]
    if ref_val <= 0:
        return np.nan
    target_val = threshold * ref_val
    satisfying = s[s >= target_val]
    if satisfying.empty:
        return np.nan
    return int(satisfying.index.min())

def estimate_k_star_knee(curve_df, val_col):
    """
    Geometric knee/elbow detection:
    Point of maximum distance from chord connecting first and last points.
    Works for both monotonic increasing (SNR) and decreasing (Variance) curves.
    """
    df = curve_df.sort_values('k').dropna().copy()
    if len(df) < 3:
        return np.nan
    x = df['k'].values.astype(float)
    y = df[val_col].values.astype(float)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return int(x[0])

    distances = [np.abs(np.cross(line_vec, np.array([x_norm[i], y_norm[i]]) - p1)) / line_len for i in range(len(x))]
    knee_idx = int(np.argmax(distances))
    return int(x[knee_idx])

# =============================================================
# 3. Main Analysis: Estimate k*(g) Per Group for SNR and ATPV
# =============================================================
results = []
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

for dataset_name, cfg in dataset_configs.items():
    print("\n" + "="*75)
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print("="*75)

    # 1. Load SNR
    print(f"Loading SNR data from: {cfg['snr_prefix']} ...")
    snr_df = load_all_under_prefix(cfg['snr_prefix'])
    if snr_df is not None and not snr_df.empty:
        snr_df['subject_id'] = snr_df['subject_id'].astype(str).str.extract(r'(\d+)')[0].str.zfill(2)

    # 2. Load ATPV
    print(f"Loading ATPV data from: {cfg['atpv_prefix']} ...")
    atpv_df = load_atpv_all_k(cfg['atpv_prefix'], K_RANGE)
    if atpv_df is not None and not atpv_df.empty:
        subj_col = [c for c in ['subject_id', 'subject', 'Subject'] if c in atpv_df.columns]
        if subj_col:
            atpv_df['subject_id'] = atpv_df[subj_col[0]].astype(str).str.extract(r'(\d+)')[0].str.zfill(2)
        val_col = 'atpv' if 'atpv' in atpv_df.columns else ('variance' if 'variance' in atpv_df.columns else atpv_df.columns[-1])
        atpv_df['atpv_val'] = atpv_df[val_col]

    for group_var, meta_dict in cfg['metadata'].items():
        print(f"\n--- Factor: '{group_var}' ({dataset_name.upper()}) ---")

        subj_to_group = {}
        for g_label, subjs in meta_dict.items():
            for s in subjs:
                s_clean = ''.join(filter(str.isdigit, str(s))).zfill(2)
                subj_to_group[s_clean] = g_label

        # ----------------- A. SNR Analysis per Group -----------------
        if snr_df is not None and not snr_df.empty:
            df_snr_var = snr_df.copy()
            df_snr_var['group'] = df_snr_var['subject_id'].map(subj_to_group)
            df_snr_var = df_snr_var.dropna(subset=['group'])
            df_snr_var['group'] = df_snr_var['group'].astype(int)

            groups_snr = sorted(df_snr_var['group'].unique())
            if len(groups_snr) >= 2:
                fig_snr, ax_snr = plt.subplots(figsize=(8, 5.5), dpi=200)

                for idx, g in enumerate(groups_snr):
                    sub_g = df_snr_var[df_snr_var['group'] == g]
                    n_subjs = sub_g['subject_id'].nunique()

                    curve_g = sub_g.groupby('k')['snr'].mean().reset_index()
                    curve_g = curve_g[curve_g['k'].isin(K_RANGE)].sort_values('k')

                    kstar_snr_85 = estimate_k_star_snr(curve_g, threshold=0.85, kmax=50)
                    kstar_snr_80 = estimate_k_star_snr(curve_g, threshold=0.80, kmax=50)
                    kstar_snr_knee = estimate_k_star_knee(curve_g, 'snr')
                    max_snr = curve_g['snr'].max()

                    print(f"  [SNR] Group {g} (N={n_subjs}): k*(85%)={kstar_snr_85} | k*(Knee)={kstar_snr_knee} | Max SNR={max_snr:.3f}")

                    results.append({
                        'dataset':      dataset_name,
                        'group_var':    group_var,
                        'group':        g,
                        'criterion':    'SNR',
                        'n_subjects':   n_subjs,
                        'k_star_rule1': kstar_snr_85,
                        'k_star_knee':  kstar_snr_knee,
                        'k_star_rule2': kstar_snr_80,
                        'metric_peak':  max_snr,
                    })

                    c = colors[idx % len(colors)]
                    ax_snr.plot(curve_g['k'], curve_g['snr'], marker='o', color=c, lw=2.0,
                                label=f"Group {g} (N={n_subjs}, k*={kstar_snr_85})")
                    if not np.isnan(kstar_snr_85):
                        val_k = curve_g.loc[curve_g['k'] == kstar_snr_85, 'snr'].values
                        if len(val_k):
                            ax_snr.scatter([kstar_snr_85], val_k, color=c, s=120, zorder=5, edgecolors='black')
                            ax_snr.axvline(kstar_snr_85, color=c, linestyle=':', alpha=0.5)

                ax_snr.set_title(f"{dataset_name.upper()} — Group-Specific SNR(k) Curves & k*(g)\nFactor: {group_var}",
                                 fontsize=12, fontweight='bold', pad=10)
                ax_snr.set_xlabel("k (bin size)", fontsize=11, fontweight='bold')
                ax_snr.set_ylabel("Mean SNR", fontsize=11, fontweight='bold')
                ax_snr.set_xticks(K_RANGE)
                ax_snr.grid(True, linestyle=':', alpha=0.6)
                ax_snr.legend(fontsize=9, loc='lower right', framealpha=0.95)

                plt.tight_layout()
                p_snr = os.path.join(PLOT_DIR, f"{dataset_name}_{group_var}_snr_kstar_per_group.png")
                plt.savefig(p_snr, dpi=200, bbox_inches='tight')
                plt.close()

        # ----------------- B. ATPV Analysis per Group -----------------
        if atpv_df is not None and not atpv_df.empty:
            df_atpv_var = atpv_df.copy()
            df_atpv_var['group'] = df_atpv_var['subject_id'].map(subj_to_group)
            df_atpv_var = df_atpv_var.dropna(subset=['group'])
            df_atpv_var['group'] = df_atpv_var['group'].astype(int)

            groups_atpv = sorted(df_atpv_var['group'].unique())
            if len(groups_atpv) >= 2:
                fig_atpv, ax_atpv = plt.subplots(figsize=(8, 5.5), dpi=200)

                for idx, g in enumerate(groups_atpv):
                    sub_g = df_atpv_var[df_atpv_var['group'] == g]
                    n_subjs = sub_g['subject_id'].nunique()

                    # Compute empirical SD of ATPV across bins and channels per k
                    sd_curve = sub_g.groupby('k')['atpv_val'].std().reset_index()
                    sd_curve = sd_curve[sd_curve['k'].isin(K_RANGE)].sort_values('k')
                    sd_curve.rename(columns={'atpv_val': 'atpv_sd'}, inplace=True)

                    kstar_atpv_knee = estimate_k_star_knee(sd_curve, 'atpv_sd')
                    mean_atpv = sub_g['atpv_val'].mean()

                    print(f"  [ATPV] Group {g} (N={n_subjs}): k*(ATPV Knee)={kstar_atpv_knee} | Mean ATPV={mean_atpv:.3f}")

                    results.append({
                        'dataset':      dataset_name,
                        'group_var':    group_var,
                        'group':        g,
                        'criterion':    'ATPV_SD_Knee',
                        'n_subjects':   n_subjs,
                        'k_star_rule1': kstar_atpv_knee,
                        'k_star_knee':  kstar_atpv_knee,
                        'k_star_rule2': np.nan,
                        'metric_peak':  mean_atpv,
                    })

                    c = colors[idx % len(colors)]
                    ax_atpv.plot(sd_curve['k'], sd_curve['atpv_sd'], marker='s', color=c, lw=2.0,
                                 label=f"Group {g} (N={n_subjs}, k*={kstar_atpv_knee})")
                    if not np.isnan(kstar_atpv_knee):
                        val_k = sd_curve.loc[sd_curve['k'] == kstar_atpv_knee, 'atpv_sd'].values
                        if len(val_k):
                            ax_atpv.scatter([kstar_atpv_knee], val_k, color=c, s=120, zorder=5, edgecolors='black')
                            ax_atpv.axvline(kstar_atpv_knee, color=c, linestyle=':', alpha=0.5)

                ax_atpv.set_title(f"{dataset_name.upper()} — Group-Specific ATPV Spread & k*(g)\nFactor: {group_var}",
                                  fontsize=12, fontweight='bold', pad=10)
                ax_atpv.set_xlabel("k (bin size)", fontsize=11, fontweight='bold')
                ax_atpv.set_ylabel("Empirical SD of ATPV", fontsize=11, fontweight='bold')
                ax_atpv.set_xticks(K_RANGE)
                ax_atpv.grid(True, linestyle=':', alpha=0.6)
                ax_atpv.legend(fontsize=9, loc='upper right', framealpha=0.95)

                plt.tight_layout()
                p_atpv = os.path.join(PLOT_DIR, f"{dataset_name}_{group_var}_atpv_kstar_per_group.png")
                plt.savefig(p_atpv, dpi=200, bbox_inches='tight')
                plt.close()

# =============================================================
# 4. Final Comparison Table
# =============================================================
if results:
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(PLOT_DIR, "k_star_per_group_results.csv")
    res_df.to_csv(out_csv, index=False)

    print("\n" + "="*85)
    print("="*85)
    print(res_df[[
        'dataset', 'group_var', 'criterion', 'group', 'n_subjects',
        'k_star_rule1', 'k_star_knee', 'metric_peak'
    ]].to_string(index=False))
    print("="*85)
    print(f"\nAll results exported to: {out_csv}")
    print("Plots saved for both SNR and ATPV per group.")
