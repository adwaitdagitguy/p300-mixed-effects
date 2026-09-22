# -*- coding: utf-8 -*-
"""
giga_task_performance_sensitivity.py

Addresses Reviewer Comment 3:
"In particular, the GIGA task-performance score uses manually chosen weights of 1, −0.5
 and −0.3 for accuracy, misses and drowsiness. The authors themselves acknowledge that
 these weights are heuristic. Sensitivity analysis would help demonstrate that the
 conclusions are not dependent on these particular choices."

Methodology:
  1. Load and transpose GigaDB questionnaire/behavioral data (54 subjects).
  2. Define the baseline composite score:
       Score = 1.0 * norm(BCI_performance) - 0.5 * norm(trials_missed) - 0.3 * norm(nodded_off)
     yielding 3 clusters (Low, Mid, High performers) with KMeans (k=3).
  3. Grid Sweep across a broad parameter space of weights:
       w_accuracy in [0.5, 0.8, 1.0, 1.2, 1.5]
       w_misses   in [0.1, 0.3, 0.5, 0.7, 1.0]
       w_drowsy   in [0.0, 0.1, 0.3, 0.5, 0.7]   (including 0 = no drowsiness penalty)
     Plus Data-Driven Objective Weighting Schemes:
       - PCA (PC1 projection of the 3 components)
       - Equal weighting (1, -1, -1)
       - Accuracy only (1, 0, 0)
       - Accuracy + Misses only (1, -1, 0)
  4. Stability Metrics for each weight configuration:
       - Adjusted Rand Index (ARI) against the baseline clustering
       - Normalized Mutual Information (NMI) against the baseline clustering
       - Spearman rank correlation of composite scores vs baseline score
       - Percentage of subjects retaining their exact baseline cluster assignment
  5. Outputs:
       - Sensitivity summary table & statistics
       - Publication-quality heatmap / distribution figures:
           giga_task_performance_sensitivity_heatmap.png
           giga_weight_sweep_correlations.png
       - Saved CSV report: giga_task_performance_weight_sensitivity.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mode as scipy_mode
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# 0. Load and Preprocess Data
# =============================================================
def load_and_prepare_data(csv_path="giga.csv"):
    raw = pd.read_csv(csv_path)
    subject_cols = [c for c in raw.columns if str(c).startswith("subject")]
    raw = raw[["Feature"] + subject_cols].dropna(subset=["Feature"])

    df = raw.set_index("Feature")[subject_cols].T
    df.index.name = "subject"
    df.index = subject_cols
    df = df.apply(pd.to_numeric, errors="coerce")

    # Impute missing values with column mode
    for col in df.columns:
        if df[col].isna().any():
            m = scipy_mode(df[col].dropna(), keepdims=True).mode
            fill_val = m[0] if len(m) > 0 else 0
            df[col].fillna(fill_val, inplace=True)

    perf_cols   = [c for c in df.columns if "bci_performance" in c.lower()]
    missed_cols = [c for c in df.columns if "trials_missed"   in c.lower()]
    nodded_cols = [c for c in df.columns if "nodded_off"      in c.lower()]

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Average over sessions for each subject
    s_perf   = minmax(df[perf_cols].mean(axis=1))
    s_missed = minmax(df[missed_cols].mean(axis=1))
    s_nodded = minmax(df[nodded_cols].mean(axis=1))

    features_df = pd.DataFrame({
        'accuracy': s_perf,
        'misses':   s_missed,
        'drowsy':   s_nodded
    }, index=df.index)

    return features_df

def cluster_score(score_series, n_clusters=3, random_state=42):
    score_norm = (score_series - score_series.min()) / (score_series.max() - score_series.min() + 1e-9)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(score_norm.values.reshape(-1, 1))

    # Order cluster labels so 0=Low, 1=Mid, 2=High
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    label_map = {old: new for new, old in enumerate(order)}
    ordered_labels = np.array([label_map[l] for l in raw_labels])
    return ordered_labels, score_norm

# =============================================================
# 1. Run Comprehensive Sensitivity Analysis
# =============================================================
def run_sensitivity_analysis(csv_path="giga.csv", out_dir="."):
    features_df = load_and_prepare_data(csv_path)

    # Baseline calculation (w_acc=1.0, w_miss=0.5, w_drowsy=0.3)
    base_score = 1.0 * features_df['accuracy'] - 0.5 * features_df['misses'] - 0.3 * features_df['drowsy']
    base_labels, base_norm_score = cluster_score(base_score)

    print("="*80)
    print("GIGA TASK-PERFORMANCE WEIGHT SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Baseline weights: Accuracy = 1.0, Misses = -0.5, Drowsiness = -0.3")
    print(f"Baseline cluster counts: Low={sum(base_labels==0)}, Mid={sum(base_labels==1)}, High={sum(base_labels==2)}")

    # 1. Grid of weights
    w_acc_vals   = [0.5, 0.8, 1.0, 1.2, 1.5]
    w_miss_vals  = [0.1, 0.3, 0.5, 0.7, 1.0]
    w_drowsy_vals = [0.0, 0.1, 0.3, 0.5, 0.7]

    records = []

    for w_acc in w_acc_vals:
        for w_miss in w_miss_vals:
            for w_drowsy in w_drowsy_vals:
                score = w_acc * features_df['accuracy'] - w_miss * features_df['misses'] - w_drowsy * features_df['drowsy']
                labels, _ = cluster_score(score)

                ari = adjusted_rand_score(base_labels, labels)
                nmi = normalized_mutual_info_score(base_labels, labels)
                rho, _ = spearmanr(base_score, score)
                pct_match = np.mean(base_labels == labels) * 100.0

                records.append({
                    'type':       'Grid Sweep',
                    'w_acc':      w_acc,
                    'w_miss':     w_miss,
                    'w_drowsy':   w_drowsy,
                    'ARI':        ari,
                    'NMI':        nmi,
                    'Spearman_r': rho,
                    'Pct_Match':  pct_match,
                })

    # 2. Objective / Data-Driven Schemes
    # PCA Scheme: First principal component of (accuracy, -misses, -drowsiness)
    pca = PCA(n_components=1, random_state=42)
    X_matrix = np.column_stack([
        features_df['accuracy'],
        -features_df['misses'],
        -features_df['drowsy']
    ])
    pca_comp = pca.fit_transform(X_matrix).flatten()
    pca_loadings = pca.components_[0]
    # Ensure positive correlation with accuracy
    if pca_loadings[0] < 0:
        pca_comp = -pca_comp
        pca_loadings = -pca_loadings

    pca_score = pd.Series(pca_comp, index=features_df.index)
    pca_labels, _ = cluster_score(pca_score)
    records.append({
        'type':       f'Objective (PCA: {pca_loadings[0]:.2f}, {pca_loadings[1]:.2f}, {pca_loadings[2]:.2f})',
        'w_acc':      abs(pca_loadings[0]),
        'w_miss':     abs(pca_loadings[1]),
        'w_drowsy':   abs(pca_loadings[2]),
        'ARI':        adjusted_rand_score(base_labels, pca_labels),
        'NMI':        normalized_mutual_info_score(base_labels, pca_labels),
        'Spearman_r': spearmanr(base_score, pca_score)[0],
        'Pct_Match':  np.mean(base_labels == pca_labels) * 100.0,
    })

    # Equal Weights Scheme (1, -1, -1)
    eq_score = 1.0 * features_df['accuracy'] - 1.0 * features_df['misses'] - 1.0 * features_df['drowsy']
    eq_labels, _ = cluster_score(eq_score)
    records.append({
        'type':       'Equal Weights (1.0, -1.0, -1.0)',
        'w_acc':      1.0,
        'w_miss':     1.0,
        'w_drowsy':   1.0,
        'ARI':        adjusted_rand_score(base_labels, eq_labels),
        'NMI':        normalized_mutual_info_score(base_labels, eq_labels),
        'Spearman_r': spearmanr(base_score, eq_score)[0],
        'Pct_Match':  np.mean(base_labels == eq_labels) * 100.0,
    })

    # Accuracy Only Scheme (1, 0, 0)
    acc_score = features_df['accuracy']
    acc_labels, _ = cluster_score(acc_score)
    records.append({
        'type':       'Accuracy Only (1.0, 0.0, 0.0)',
        'w_acc':      1.0,
        'w_miss':     0.0,
        'w_drowsy':   0.0,
        'ARI':        adjusted_rand_score(base_labels, acc_labels),
        'NMI':        normalized_mutual_info_score(base_labels, acc_labels),
        'Spearman_r': spearmanr(base_score, acc_score)[0],
        'Pct_Match':  np.mean(base_labels == acc_labels) * 100.0,
    })

    # Accuracy + Misses Only (No Drowsiness Penalty)
    nomiss_score = 1.0 * features_df['accuracy'] - 0.5 * features_df['misses']
    nomiss_labels, _ = cluster_score(nomiss_score)
    records.append({
        'type':       'Accuracy + Misses Only (1.0, -0.5, 0.0)',
        'w_acc':      1.0,
        'w_miss':     0.5,
        'w_drowsy':   0.0,
        'ARI':        adjusted_rand_score(base_labels, nomiss_labels),
        'NMI':        normalized_mutual_info_score(base_labels, nomiss_labels),
        'Spearman_r': spearmanr(base_score, nomiss_score)[0],
        'Pct_Match':  np.mean(base_labels == nomiss_labels) * 100.0,
    })

    res_df = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "giga_task_performance_weight_sensitivity.csv")
    res_df.to_csv(out_csv, index=False)

    # =============================================================
    # 2. Print Summary Statistics for Reviewer Rebuttal
    # =============================================================
    grid_res = res_df[res_df['type'] == 'Grid Sweep']

    print("\n" + "-"*80)
    print("SENSITIVITY SUMMARY ACROSS 125 WEIGHT COMBINATIONS:")
    print("-"*80)
    print(f"  • Spearman Rank Correlation (rho): mean = {grid_res['Spearman_r'].mean():.3f} (min = {grid_res['Spearman_r'].min():.3f}, max = {grid_res['Spearman_r'].max():.3f})")
    print(f"  • Adjusted Rand Index (ARI):       mean = {grid_res['ARI'].mean():.3f} (min = {grid_res['ARI'].min():.3f}, max = {grid_res['ARI'].max():.3f})")
    print(f"  • Normalized Mutual Info (NMI):    mean = {grid_res['NMI'].mean():.3f} (min = {grid_res['NMI'].min():.3f}, max = {grid_res['NMI'].max():.3f})")
    print(f"  • Subject Cluster Match (%):       mean = {grid_res['Pct_Match'].mean():.1f}% (min = {grid_res['Pct_Match'].min():.1f}%, max = {grid_res['Pct_Match'].max():.1f}%)")

    print("\nALTERNATIVE & OBJECTIVE WEIGHTING SCHEMES:")
    alt_res = res_df[res_df['type'] != 'Grid Sweep']
    print(alt_res[['type', 'ARI', 'NMI', 'Spearman_r', 'Pct_Match']].to_string(index=False))

    # =============================================================
    # 3. Visualizations (Single Standalone Heatmaps for Print)
    # =============================================================
    # Standalone Figure 1: Single ARI Heatmap (Optimized for publication print size)
    fixed_acc_df = grid_res[grid_res['w_acc'] == 1.0]
    pivot_ari = fixed_acc_df.pivot(index='w_drowsy', columns='w_miss', values='ARI')
    pivot_pct = fixed_acc_df.pivot(index='w_drowsy', columns='w_miss', values='Pct_Match')

    fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=300)
    im = ax.imshow(pivot_ari.values, cmap='YlGnBu', vmin=0.5, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(pivot_ari.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot_ari.columns], fontsize=15, fontweight='bold')
    ax.set_yticks(range(len(pivot_ari.index)))
    ax.set_yticklabels([f"{i:.1f}" for i in pivot_ari.index], fontsize=15, fontweight='bold')
    ax.set_xlabel(r"Weight: Missed Trials ($|w_{\mathrm{miss}}|)$", fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Weight: Drowsiness ($|w_{\mathrm{drowsy}}|)$", fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(r"Cluster Stability (Adjusted Rand Index, $w_{\mathrm{acc}}=1.0$)", fontsize=16, fontweight='bold', pad=14)

    for i in range(len(pivot_ari.index)):
        for j in range(len(pivot_ari.columns)):
            val = pivot_ari.values[i, j]
            is_baseline = (pivot_ari.index[i] == 0.3) and (pivot_ari.columns[j] == 0.5)
            txt = f"{val:.2f}"
            if is_baseline:
                txt += "\n(Base)"
                # Add highlighting rectangle around baseline cell
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                     edgecolor='#d62728', lw=3.0, linestyle='-')
                ax.add_patch(rect)

            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if val > 0.72 else 'black',
                    fontweight='bold', fontsize=17)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Adjusted Rand Index (ARI)', fontsize=15, fontweight='bold', labelpad=10)

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "giga_task_performance_sensitivity_heatmap.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[saved] Single ARI Heatmap -> {fig1_path}")

    # Also save standalone Pct Match Heatmap separately
    fig_pct, ax_pct = plt.subplots(figsize=(8.0, 6.8), dpi=300)
    im_pct = ax_pct.imshow(pivot_pct.values, cmap='Blues', vmin=60, vmax=100, aspect='auto')

    ax_pct.set_xticks(range(len(pivot_pct.columns)))
    ax_pct.set_xticklabels([f"{c:.1f}" for c in pivot_pct.columns], fontsize=15, fontweight='bold')
    ax_pct.set_yticks(range(len(pivot_pct.index)))
    ax_pct.set_yticklabels([f"{i:.1f}" for i in pivot_pct.index], fontsize=15, fontweight='bold')
    ax_pct.set_xlabel(r"Weight: Missed Trials ($|w_{\mathrm{miss}}|)$", fontsize=16, fontweight='bold', labelpad=10)
    ax_pct.set_ylabel(r"Weight: Drowsiness ($|w_{\mathrm{drowsy}}|)$", fontsize=16, fontweight='bold', labelpad=10)
    ax_pct.set_title(r"Subject Cluster Retention Rate (%, $w_{\mathrm{acc}}=1.0$)", fontsize=16, fontweight='bold', pad=14)

    for i in range(len(pivot_pct.index)):
        for j in range(len(pivot_pct.columns)):
            val = pivot_pct.values[i, j]
            is_baseline = (pivot_pct.index[i] == 0.3) and (pivot_pct.columns[j] == 0.5)
            txt = f"{val:.0f}%"
            if is_baseline:
                txt += "\n(Base)"
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                     edgecolor='#d62728', lw=3.0, linestyle='-')
                ax_pct.add_patch(rect)

            ax_pct.text(j, i, txt, ha='center', va='center',
                        color='white' if val > 80 else 'black',
                        fontweight='bold', fontsize=18)

    cbar_pct = plt.colorbar(im_pct, ax=ax_pct, fraction=0.046, pad=0.04)
    cbar_pct.ax.tick_params(labelsize=13)
    cbar_pct.set_label('Retained Baseline Cluster (%)', fontsize=15, fontweight='bold', labelpad=10)

    plt.tight_layout()
    pct_path = os.path.join(out_dir, "giga_task_performance_pct_match_heatmap.png")
    plt.savefig(pct_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] Single Pct Match Heatmap -> {pct_path}")

    # Figure 2: Spearman Rank Correlation Distribution across all combinations
    plt.figure(figsize=(7, 4.5), dpi=200)
    plt.hist(grid_res['Spearman_r'], bins=20, color='skyblue', edgecolor='black', alpha=0.85)
    plt.axvline(grid_res['Spearman_r'].mean(), color='red', linestyle='--', lw=2,
                label=f"Mean rho = {grid_res['Spearman_r'].mean():.3f}")
    plt.axvline(grid_res['Spearman_r'].min(), color='darkorange', linestyle=':', lw=1.5,
                label=f"Min rho = {grid_res['Spearman_r'].min():.3f}")
    plt.title("Distribution of Spearman Rank Correlations vs Baseline Score\n(125 Weight Combinations)",
              fontsize=11, fontweight='bold')
    plt.xlabel("Spearman Rank Correlation (rho)", fontsize=10, fontweight='bold')
    plt.ylabel("Frequency", fontsize=10, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "giga_weight_sweep_correlations.png")
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] Correlation distribution plot -> {fig2_path}")

    print("\nAnalysis complete. Use these statistics and figures in your response to Reviewer Comment 3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="giga.csv", help="Path to questionnaire CSV")
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()

    # If giga.csv not in current dir, check common paths
    csv_file = args.csv
    if not os.path.exists(csv_file):
        candidates = [
            "d:/eeg_datasets/questionnaire/files_giga/giga.csv",
            "d:/eeg_datasets/questionnaire/giga.csv",
            "../giga.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_file = c
                break

    run_sensitivity_analysis(csv_file, args.out)

