# -*- coding: utf-8 -*-
"""
analytic_curve.py

Evaluates Section 5 (k-selection) sensitivity:

1. ATPV sampling-noise check
   -------------------------
   The unbiased ATPV estimator (ddof=1) is, in expectation, INVARIANT to k --
   it targets the true per-timepoint variance sigma^2 regardless of bin size.
   What DOES change with k is the estimator's own sampling variability: the
   variance of a sample-variance estimate computed from k observations shrinks
   as k grows, following (for approximately Gaussian data):

       SD_analytic(k) ~= sigma^2 * sqrt(2 / (k-1))

   We overlay this analytic curve (anchored to the empirical variance at
   large k) against the empirical spread of ATPV across bins/channels/subjects
   at each k. If the empirical curve tracks the analytic curve, the observed
   "elbow" is just estimator noise shrinking -- not a real change in signal
   structure. If the empirical curve departs from the analytic curve (flattens
   earlier, or sits above it at large k due to genuine non-stationarity),
   that departure is the evidence that k* reflects real data structure, not
   only sampling variance.

2. SNR rule sensitivity to (threshold, kmax)
   ------------------------------------------
   k* under the SNR rule is defined as the smallest k with
       SNR(k) >= threshold * SNR(kmax)
   Both threshold (default 0.85) and kmax (default 50) are free parameters.
   We report k* over a grid of both, per dataset, to show how much k* moves.

Usage:
    export AZURE_TENANT_ID=...
    export AZURE_CLIENT_ID=...
    export AZURE_CLIENT_SECRET=...
    export AZURE_STORAGE_ACCOUNT=...
    export AZURE_STORAGE_CONTAINER=...
    pip install azure-identity azure-storage-blob pandas numpy matplotlib pyarrow
    python analytic_curve.py
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

# =============================================================
# 0. Azure configuration
# =============================================================
AZURE_TENANT_ID         = os.environ['AZURE_TENANT_ID']
AZURE_CLIENT_ID         = os.environ['AZURE_CLIENT_ID']
AZURE_CLIENT_SECRET     = os.environ['AZURE_CLIENT_SECRET']
AZURE_STORAGE_ACCOUNT   = os.environ['AZURE_STORAGE_ACCOUNT']
AZURE_STORAGE_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']

PLOT_DIR = os.getcwd()

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

def list_blobs_prefix(prefix):
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]

def download_blob_df(blob_path):
    buf = io.BytesIO()
    container.get_blob_client(blob_path).download_blob().readinto(buf)
    buf.seek(0)
    return pd.read_parquet(buf)

def load_all_under_prefix(prefix):
    paths = [b for b in list_blobs_prefix(prefix) if b.endswith('.parquet')]
    if not paths:
        return None
    dfs = [download_blob_df(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def savefig(name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")

# =============================================================
# 1. Dataset configs
#    ATPV metadata lives under {prefix}/ddof_1/k_{k}/...  (all k in one tree)
#    SNR metadata lives under a flat prefix, one parquet per (subject, run)
#    covering all k values in the 'k' column.
# =============================================================
dataset_configs = {
    'korean': {
        'atpv_prefix': 'korean/parquet_meta/ddof_1',
        'snr_prefix':  'korean/parquet_snr',
    },
    'giga': {
        'atpv_prefix': 'giga_preprocessed/parquet_meta/ddof_1',
        'snr_prefix':  'giga_preprocessed/parquet_snr',
    },
    'adaptive': {
        'atpv_prefix': 'adaptiveP300/parquet_meta/ddof_1',
        'snr_prefix':  'adaptiveP300/parquet_snr',
    },
}

K_RANGE = [1, 2, 3, 4] + list(range(5, 51, 5))

# =============================================================
# 2. Part 1 — ATPV vs analytic sampling-SD curve
# =============================================================

def load_atpv_all_k(atpv_prefix, k_range):
    """
    ATPV metadata is stored as {atpv_prefix}/k_{k}/<...>/subject_xx.parquet.
    Load every k, concatenate with a 'k' column (already present per-file,
    but we double check/attach if missing), return one long DataFrame.
    """
    frames = []
    for k in k_range:
        prefix = f"{atpv_prefix}/k_{k}/"
        df_k   = load_all_under_prefix(prefix)
        if df_k is None or df_k.empty:
            continue
        if 'k' not in df_k.columns:
            df_k['k'] = k
        frames.append(df_k)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def analytic_sampling_sd(k_range, sigma2_ref):
    """
    Analytic SD of a ddof=1 sample-variance estimator built from k
    (approximately Gaussian, iid) observations, anchored to sigma2_ref
    (taken as the empirical ATPV mean at the largest available k, our
    best estimate of the true per-timepoint variance sigma^2).

        SD_analytic(k) = sigma2_ref * sqrt(2 / (k - 1)),   k >= 2
        SD_analytic(1)  = NaN  (ddof=1 undefined at k=1)
    """
    sd = np.full(len(k_range), np.nan)
    for i, k in enumerate(k_range):
        if k >= 2:
            sd[i] = sigma2_ref * np.sqrt(2.0 / (k - 1))
    return sd

def plot_atpv_vs_analytic(dataset_name, atpv_df, k_range):
    """
    Empirical ATPV(k): mean and SD across (group_key, channel, bin) at each k.
    Analytic SD(k): sampling-noise-only prediction, anchored at sigma2_ref
    = empirical ATPV mean at the largest k available (assumed most stable).
    """
    empirical_mean = []
    empirical_sd   = []
    for k in k_range:
        vals = atpv_df.loc[atpv_df['k'] == k, 'atpv'].dropna().values
        if len(vals) == 0:
            empirical_mean.append(np.nan)
            empirical_sd.append(np.nan)
        else:
            empirical_mean.append(np.mean(vals))
            empirical_sd.append(np.std(vals, ddof=1))

    empirical_mean = np.array(empirical_mean)
    empirical_sd   = np.array(empirical_sd)

    # Anchor sigma2_ref at the largest k with valid data
    valid_k_idx = np.where(~np.isnan(empirical_mean))[0]
    if len(valid_k_idx) == 0:
        print(f"  [skip] {dataset_name}: no valid ATPV data")
        return None
    sigma2_ref = empirical_mean[valid_k_idx[-1]]

    analytic_sd = analytic_sampling_sd(k_range, sigma2_ref)

    # --- Plot: empirical mean +/- SD, with analytic SD envelope overlaid ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: empirical mean ATPV (should be ~flat if estimator is unbiased)
    ax = axes[0]
    ax.plot(k_range, empirical_mean, marker='o', color='C0', label='Empirical mean ATPV(k)')
    ax.axhline(sigma2_ref, color='gray', lw=1, linestyle='--',
               label=f'sigma² reference (ATPV at k={k_range[valid_k_idx[-1]]})')
    ax.set_xlabel("k (bin size)")
    ax.set_ylabel("Mean ATPV")
    ax.set_title(f"{dataset_name}: Empirical mean ATPV(k)\n"
                 f"(flat line = ddof=1 estimator is unbiased in k, as expected)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right panel: empirical SD vs analytic sampling-SD
    ax = axes[1]
    ax.plot(k_range, empirical_sd, marker='o', color='C1',
            label='Empirical SD of ATPV across (group, channel, bin)')
    ax.plot(k_range, analytic_sd, marker='s', color='C2', linestyle='--',
            label='Analytic sampling SD  ( sigma² · sqrt(2/(k−1)) )')
    ax.set_xlabel("k (bin size)")
    ax.set_ylabel("SD of ATPV")
    ax.set_title(f"{dataset_name}: Empirical vs analytic sampling-noise SD\n"
                 f"(departure = real structure beyond sampling noise)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(f"{dataset_name}_atpv_vs_analytic_sampling_sd")

    # --- Quantify departure: ratio of empirical to analytic SD ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = empirical_sd / analytic_sd

    departure_df = pd.DataFrame({
        'k':               k_range,
        'empirical_mean':  empirical_mean,
        'empirical_sd':    empirical_sd,
        'analytic_sd':     analytic_sd,
        'ratio_emp_over_analytic': ratio,
    })
    print(f"\n  {dataset_name}: empirical / analytic SD ratio by k")
    print(f"  (ratio ~= 1 → sampling noise explains the curve; "
          f"ratio >> 1 at large k → real non-stationarity remains)")
    print(departure_df.round(4).to_string(index=False))

    out_csv = os.path.join(PLOT_DIR, f"{dataset_name}_atpv_analytic_departure.csv")
    departure_df.to_csv(out_csv, index=False)
    print(f"  [saved] {out_csv}")

    return departure_df

# =============================================================
# 3. Part 2 — SNR rule sensitivity to (threshold, kmax)
# =============================================================

def compute_k_star_snr(snr_curve_df, threshold, kmax):
    """
    snr_curve_df: DataFrame with columns ['k', 'snr'] -- one row per k,
    already averaged across ROI channels / subjects / groups.
    Returns smallest k such that SNR(k) >= threshold * SNR(kmax),
    restricted to k <= kmax. Returns np.nan if kmax not available or
    no k satisfies the condition.
    """
    curve = snr_curve_df.set_index('k')['snr']
    curve = curve[curve.index <= kmax].sort_index()

    if kmax not in curve.index or not np.isfinite(curve.loc[kmax]):
        # fall back to the largest available k <= kmax as the reference
        valid = curve.dropna()
        if valid.empty:
            return np.nan
        ref_k = valid.index.max()
    else:
        ref_k = kmax

    ref_val = curve.loc[ref_k]
    if not np.isfinite(ref_val) or ref_val == 0:
        return np.nan

    target = threshold * ref_val
    satisfying = curve[curve >= target].dropna()
    if satisfying.empty:
        return np.nan
    return int(satisfying.index.min())

def build_mean_snr_curve(snr_df, k_range):
    """Collapse raw SNR rows to one mean-SNR-per-k curve (all subjects/channels pooled)."""
    rows = []
    for k in k_range:
        vals = snr_df.loc[snr_df['k'] == k, 'snr'].dropna().values
        rows.append({'k': k, 'snr': np.mean(vals) if len(vals) else np.nan})
    return pd.DataFrame(rows)

def sensitivity_analysis(dataset_name, snr_df, k_range,
                         thresholds=(0.80, 0.85, 0.90, 0.95),
                         kmax_grid=(20, 30, 40, 50)):
    """
    For each (threshold, kmax) combination, compute k*.
    kmax_grid is intersected with what's actually available in k_range.
    """
    kmax_grid = [k for k in kmax_grid if k in k_range]
    if not kmax_grid:
        print(f"  [skip] {dataset_name}: none of the requested kmax values "
              f"are present in k_range={k_range}")
        return None

    snr_curve = build_mean_snr_curve(snr_df, k_range)

    rows = []
    for kmax in kmax_grid:
        for thr in thresholds:
            k_star = compute_k_star_snr(snr_curve, thr, kmax)
            rows.append({'threshold': thr, 'kmax': kmax, 'k_star': k_star})

    sens_df = pd.DataFrame(rows)

    print(f"\n  {dataset_name}: k* sensitivity to (threshold, kmax)")
    pivot = sens_df.pivot(index='threshold', columns='kmax', values='k_star')
    print(pivot.to_string())

    out_csv = os.path.join(PLOT_DIR, f"{dataset_name}_snr_kstar_sensitivity.csv")
    sens_df.to_csv(out_csv, index=False)
    print(f"  [saved] {out_csv}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("kmax")
    ax.set_ylabel("threshold (fraction of SNR(kmax))")
    ax.set_title(f"{dataset_name}: k* as a function of (threshold, kmax)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            label = f"{val:.0f}" if np.isfinite(val) else "n/a"
            ax.text(j, i, label, ha='center', va='center',
                    color='white' if np.isfinite(val) else 'red', fontsize=9)
    fig.colorbar(im, ax=ax, label='k*')
    plt.tight_layout()
    savefig(f"{dataset_name}_snr_kstar_sensitivity_heatmap")

    # Line plot: k* vs threshold, one line per kmax
    plt.figure(figsize=(8, 5))
    for kmax in kmax_grid:
        sub = sens_df[sens_df['kmax'] == kmax].sort_values('threshold')
        plt.plot(sub['threshold'], sub['k_star'], marker='o', label=f"kmax={kmax}")
    plt.xlabel("threshold (fraction of SNR(kmax))")
    plt.ylabel("k*")
    plt.title(f"{dataset_name}: k* vs threshold, by kmax")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(f"{dataset_name}_snr_kstar_vs_threshold")

    return sens_df

# =============================================================
# 4. Main
# =============================================================
all_departure = {}
all_sensitivity = {}

for dataset_name, cfg in dataset_configs.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    # --- Part 1: ATPV vs analytic sampling SD ---
    print(f"\n  Loading ATPV metadata from {cfg['atpv_prefix']} ...")
    atpv_df = load_atpv_all_k(cfg['atpv_prefix'], K_RANGE)
    if atpv_df is None or atpv_df.empty:
        print(f"  [skip] no ATPV data found for {dataset_name}")
    else:
        departure_df = plot_atpv_vs_analytic(dataset_name, atpv_df, K_RANGE)
        all_departure[dataset_name] = departure_df
    del atpv_df

    # --- Part 2: SNR k* sensitivity ---
    print(f"\n  Loading SNR metadata from {cfg['snr_prefix']} ...")
    snr_df = load_all_under_prefix(cfg['snr_prefix'])
    if snr_df is None or snr_df.empty:
        print(f"  [skip] no SNR data found for {dataset_name}")
    else:
        sens_df = sensitivity_analysis(dataset_name, snr_df, K_RANGE)
        all_sensitivity[dataset_name] = sens_df
    del snr_df

# =============================================================
# 5. Combined summary table across datasets (for the paper)
# =============================================================
print(f"\n{'='*60}")
print("COMBINED SUMMARY")
print(f"{'='*60}")

if all_sensitivity:
    combined = []
    for dataset_name, sens_df in all_sensitivity.items():
        if sens_df is None:
            continue
        d = sens_df.copy()
        d['dataset'] = dataset_name
        combined.append(d)
    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        out_csv = os.path.join(PLOT_DIR, "all_datasets_kstar_sensitivity.csv")
        combined_df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")

        # k* at the paper's actual reported operating point (threshold=0.85, kmax=50)
        default_row = combined_df[
            (combined_df['threshold'] == 0.85) & (combined_df['kmax'] == 50)
        ]
        if not default_row.empty:
            print("\nk* at reported operating point (threshold=0.85, kmax=50):")
            print(default_row[['dataset', 'k_star']].to_string(index=False))

        # Range of k* across the full grid, per dataset
        print("\nRange of k* across full (threshold, kmax) grid, per dataset:")
        print(combined_df.groupby('dataset')['k_star'].agg(['min', 'max']).to_string())

print("\nAll done.")
print(f"Outputs written to: {PLOT_DIR}")
