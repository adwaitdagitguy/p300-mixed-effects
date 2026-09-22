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
    # 3. Visualizations
    # =============================================================
    # Figure 1: Heatmap of ARI when varying w_miss and w_drowsy (with w_acc fixed at 1.0)
    fixed_acc_df = grid_res[grid_res['w_acc'] == 1.0]
    pivot_ari = fixed_acc_df.pivot(index='w_drowsy', columns='w_miss', values='ARI')
    pivot_pct = fixed_acc_df.pivot(index='w_drowsy', columns='w_miss', values='Pct_Match')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Left: ARI Heatmap
    im1 = axes[0].imshow(pivot_ari.values, cmap='YlGnBu', vmin=0.5, vmax=1.0, aspect='auto')
    axes[0].set_xticks(range(len(pivot_ari.columns)))
    axes[0].set_xticklabels([f"{c:.1f}" for c in pivot_ari.columns])
    axes[0].set_yticks(range(len(pivot_ari.index)))
    axes[0].set_yticklabels([f"{i:.1f}" for i in pivot_ari.index])
    axes[0].set_xlabel("Weight: Missed Trials (|w_miss|)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Weight: Drowsiness (|w_drowsy|)", fontsize=11, fontweight='bold')
    axes[0].set_title("Cluster Stability (Adjusted Rand Index)\nw_acc = 1.0", fontsize=12, fontweight='bold')

    for i in range(len(pivot_ari.index)):
        for j in range(len(pivot_ari.columns)):
            val = pivot_ari.values[i, j]
            # Mark baseline position
            is_baseline = (pivot_ari.index[i] == 0.3) and (pivot_ari.columns[j] == 0.5)
            txt = f"{val:.2f}" + ("\n(Base)" if is_baseline else "")
            axes[0].text(j, i, txt, ha='center', va='center',
                         color='white' if val > 0.75 else 'black',
                         fontweight='bold' if is_baseline else 'normal', fontsize=9)
    plt.colorbar(im1, ax=axes[0], label='ARI')

    # Right: Pct Match Heatmap
    im2 = axes[1].imshow(pivot_pct.values, cmap='Blues', vmin=60, vmax=100, aspect='auto')
    axes[1].set_xticks(range(len(pivot_pct.columns)))
    axes[1].set_xticklabels([f"{c:.1f}" for c in pivot_pct.columns])
    axes[1].set_yticks(range(len(pivot_pct.index)))
    axes[1].set_yticklabels([f"{i:.1f}" for i in pivot_pct.index])
    axes[1].set_xlabel("Weight: Missed Trials (|w_miss|)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Weight: Drowsiness (|w_drowsy|)", fontsize=11, fontweight='bold')
    axes[1].set_title("Percentage of Subjects Retaining Baseline Cluster (%)\nw_acc = 1.0", fontsize=12, fontweight='bold')

    for i in range(len(pivot_pct.index)):
        for j in range(len(pivot_pct.columns)):
            val = pivot_pct.values[i, j]
            is_baseline = (pivot_pct.index[i] == 0.3) and (pivot_pct.columns[j] == 0.5)
            txt = f"{val:.0f}%" + ("\n(Base)" if is_baseline else "")
            axes[1].text(j, i, txt, ha='center', va='center',
                         color='white' if val > 80 else 'black',
                         fontweight='bold' if is_baseline else 'normal', fontsize=9)
    plt.colorbar(im2, ax=axes[1], label='% Match')

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "giga_task_performance_sensitivity_heatmap.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"\n[saved] Heatmap plot -> {fig1_path}")

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

