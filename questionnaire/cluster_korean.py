"""
cluster_korean.py
==================
Clusters Korean P300 BCI subjects along 4 independent axes:

  1. Gender         – gender M vs F
  2. BCI Experience – biofeedback_exp N (naive) vs Y (experienced)
  3. Subjective     – aggregate 7 RE features → PCA → KMeans (elbow) → KNN smooth
  4. Task-Based     – composite score from perf_pred, too_fast, too_difficult,
                      concentration_diff → KMeans k=3

Scale polarity (per dataset author):
  NORMAL  (higher = better): anxiety, boredom, physical_state, mental_state,
                              eye_state, concentration, perf_pred, concentration_diff
  INVERTED (higher = worse): sleepy, too_fast, too_difficult

Usage
-----
    python cluster_korean.py [--csv PATH]

Default: korean.csv in the same directory.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RENAME_MAP = {
    'Unnamed: 0': 'subject', '1': 'handedness', '2': 'disease', '3': 'age',
    '4': 'gender', '5': 'corrected_vision', '6': 'biofeedback_exp',
    '16': 'anxiety', '17': 'boredom', '18': 'physical_state', '19': 'mental_state',
    '20': 'eye_state', '21': 'performance_prediction', '22': 'continue_YN',
    '23': 'anxiety_2', '24': 'boredom_2', '25': 'physical_state_2',
    '26': 'mental_state_2', '27': 'concentration', '28': 'eye_state_2',
    '29': 'sleepy', '30': 'too_fast', '31': 'too_difficult',
    '32': 'concentration_diff', '33': 'last_run_perf_pred',
    '34': 'next_run_perf_pred', '35': 'continue_YN_2', '36': 'anxiety_3',
    '37': 'boredom_3', '38': 'concentration_2', '39': 'physical_state_3',
    '40': 'mental_state_3', '41': 'eye_state_3', '42': 'sleepy_2',
    '43': 'too_fast_2', '44': 'too_difficult_2', '45': 'concentration_diff_2',
    '47': 'next_run_perf_pred_2', '48': 'continue_YN_3', '49': 'anxiety_4',
    '50': 'boredom_4', '51': 'concentration_3', '52': 'physical_state_4',
    '53': 'mental_state_4', '54': 'eye_state_4', '55': 'sleepy_3',
    '56': 'too_fast_3', '57': 'too_difficult_3', '58': 'concentration_diff_3',
    '60': 'next_run_perf_pred_3', '61': 'continue_YN_4', '62': 'anxiety_5',
    '63': 'boredom_5', '64': 'concentration_4', '65': 'physical_state_5',
    '66': 'mental_state_5', '67': 'eye_state_5', '68': 'sleepy_4',
    '69': 'too_fast_4', '70': 'too_difficult_4', '71': 'concentration_diff_4',
    '73': 'next_run_perf_pred_4', '74': 'anxiety_6', '75': 'boredom_6',
    '76': 'concentration_5', '77': 'physical_state_6', '78': 'mental_state_6',
    '79': 'eye_state_6', '80': 'sleepy_5', '81': 'too_fast_5',
    '82': 'too_difficult_5', '84': 'concentration_diff_5', '85': 'duration',
    '86': 'evaluation', '87': 'experimental_environment', '88': 'task_difficulty'
}

# Aggregation groups
SUBJ_GROUPS = {
    'Anxiety':        lambda c: 'anxiety'        in c and 'diff' not in c,
    'Boredom':        lambda c: 'boredom'        in c,
    'Physical_State': lambda c: 'physical_state' in c,
    'Mental_State':   lambda c: 'mental_state'   in c,
    'Eye_State':      lambda c: 'eye_state'      in c,
    'Concentration':  lambda c: 'concentration'  in c and 'diff' not in c,
    'Sleepy':         lambda c: 'sleepy'         in c,
}

TASK_GROUPS = {
    'Perf_Pred':    lambda c: 'perf_pred' in c or c == 'performance_prediction',
    'Too_Fast':     lambda c: 'too_fast'      in c,
    'Too_Difficult':lambda c: 'too_difficult' in c,
    'Conc_Diff':    lambda c: 'concentration_diff' in c,
}

# Scale polarity for interpretation
INVERTED_SUBJ = {'Sleepy'}
INVERTED_TASK = {'Too_Fast', 'Too_Difficult'}


# ──────────────────────────────────────────────────────────────────────────────
# 0.  LOAD
# ──────────────────────────────────────────────────────────────────────────────

def load(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw.rename(columns=RENAME_MAP, inplace=True)

    # Format subject ID
    raw['subject'] = (raw['subject'].str.extract(r'(\d+)$')[0]
                      .astype(int).apply(lambda x: f's{x:02d}'))
    raw.set_index('subject', inplace=True)

    # Keep only mapped columns (ignore unmapped numeric cols like 7-15)
    mapped_names = list(RENAME_MAP.values())[1:]   # exclude 'subject'
    raw = raw[[c for c in mapped_names if c in raw.columns]]

    # Mode-impute missing values
    for col in raw.columns:
        if raw[col].isna().any():
            try:
                fill = scipy_mode(raw[col].dropna(), keepdims=True).mode[0]
            except Exception:
                fill = raw[col].dropna().iloc[0] if len(raw[col].dropna()) else 0
            raw[col] = raw[col].fillna(fill)

    print(f"[load]  Shape: {raw.shape}  |  Subjects: {len(raw)}")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GENDER
# ──────────────────────────────────────────────────────────────────────────────

def gender_clusters(df: pd.DataFrame) -> dict:
    g = df['gender'].astype(str)
    return {
        'Gender_Male':   list(g[g == 'M'].index),
        'Gender_Female': list(g[g == 'F'].index),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  BCI EXPERIENCE
# ──────────────────────────────────────────────────────────────────────────────

def bciexp_clusters(df: pd.DataFrame) -> dict:
    b = df['biofeedback_exp'].astype(str)
    return {
        'BCIExp_Naive':       list(b[b == 'N'].index),
        'BCIExp_Experienced': list(b[b == 'Y'].index),
    }


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Mean-aggregate session columns per feature group."""
    agg = {}
    for grp, fn in groups.items():
        cols = [c for c in df.columns if fn(c)]
        # coerce to numeric
        sub = df[cols].apply(pd.to_numeric, errors='coerce')
        if not sub.empty:
            agg[grp] = sub.mean(axis=1)
    return pd.DataFrame(agg, index=df.index).fillna(
        lambda x: x.median()
    )

def _optimal_k_elbow(X: np.ndarray, k_range: range, random_state: int = 42):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    inertias = np.array(inertias)
    diffs2 = np.diff(np.diff(inertias))
    best_idx = int(np.argmax(diffs2)) + 1
    return list(k_range)[best_idx], list(k_range), inertias

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SUBJECTIVE STATE  (PCA + KMeans + KNN)
# ──────────────────────────────────────────────────────────────────────────────

def subjective_clusters(df: pd.DataFrame,
                        n_pca_components: int = 5,
                        n_top_loadings: int = 5,
                        random_state: int = 42) -> dict:
    # Aggregate → 7 mean scores
    sub_df = _aggregate(df, SUBJ_GROUPS)
    feat_names = sub_df.columns.tolist()
    print(f"\n[subj]  Aggregated to {len(feat_names)} mean scores: {feat_names}")

    X_scaled = StandardScaler().fit_transform(sub_df.values)

    # PCA
    pca_full = PCA(random_state=random_state).fit(X_scaled)
    cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp   = max(2, int(np.searchsorted(cum_var, 0.90) + 1))
    n_comp   = min(n_comp, n_pca_components, X_scaled.shape[1])

    pca   = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    ev    = pca.explained_variance_ratio_
    ev_c  = np.cumsum(ev)

    # Print PCA table
    print(f"\n[subj]  PCA components used: {n_comp}")
    comp_df = pd.DataFrame(pca.components_, columns=feat_names,
                           index=[f'PC{i+1}' for i in range(n_comp)])
    print(f"\n  {'PC':<6}  {'Var %':>7}  {'Cum %':>7}  Top-{n_top_loadings} features (loading)")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*55}")
    pc_labels = []
    for i, pc in enumerate(comp_df.index):
        row     = comp_df.loc[pc]
        top_idx = row.abs().nlargest(n_top_loadings).index
        top_vals= row[top_idx]
        loading_str = '  '.join(f'{f} ({v:+.3f})' for f, v in zip(top_idx, top_vals))
        print(f"  {pc:<6}  {ev[i]*100:>6.1f}%  {ev_c[i]*100:>6.1f}%  {loading_str}")
        dominant = top_idx[0].lower()
        if   'mental'   in dominant or 'physical' in dominant: lbl = 'Cognitive/physical state'
        elif 'anxiety'  in dominant:                           lbl = 'Anxiety level'
        elif 'boredom'  in dominant:                           lbl = 'Boredom'
        elif 'eye'      in dominant or 'sleepy' in dominant:   lbl = 'Alertness / wakefulness'
        elif 'concentr' in dominant:                           lbl = 'Concentration'
        else:                                                  lbl = dominant.split('_')[0].capitalize()
        pc_labels.append(f'PC{i+1}: {lbl} ({ev[i]*100:.1f}%)')

    print(f"\n  Semantic axis summary:")
    for lbl in pc_labels:
        print(f"    • {lbl}")

    # Elbow → KMeans → KNN
    k_range   = range(2, min(8, len(df)))
    optimal_k, ks, inertias = _optimal_k_elbow(X_pca, k_range, random_state)
    print(f"\n[subj]  Optimal k (elbow): {optimal_k}")

    km_labels  = KMeans(n_clusters=optimal_k, random_state=random_state,
                        n_init=10).fit_predict(X_pca)
    knn        = KNeighborsClassifier(n_neighbors=min(5, len(df) - 1))
    knn.fit(X_pca, km_labels)
    knn_labels = knn.predict(X_pca)

    # Centroid profiles
    X_scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=feat_names)
    X_scaled_df['_cluster'] = knn_labels
    cluster_means = X_scaled_df.groupby('_cluster')[feat_names].mean()

    print(f"\n  Cluster centroid profiles (mean z-score of aggregated features):")
    header = f"  {'Feature':<18}" + ''.join(f'  C{c:>5}' for c in range(optimal_k))
    print(header)
    print('  ' + '-' * (18 + optimal_k * 8))
    for feat in feat_names:
        row = f"  {feat:<18}" + ''.join(
            f"  {cluster_means.loc[c, feat]:+.3f}" for c in range(optimal_k))
        print(row)

    # Polarity-aware interpretation
    print(f"\n  Scale note: Sleepy is INVERTED (higher = worse). Others: higher = better.")
    print(f"\n  Cluster interpretations:")
    for c in range(optimal_k):
        centroid  = cluster_means.loc[c]
        positives, negatives = [], []
        for feat, val in centroid.items():
            if abs(val) <= 0.3:
                continue
            inv = feat in INVERTED_SUBJ
            if val > 0:
                (negatives if inv else positives).append(
                    f"{feat}↑{'(worse)' if inv else ''}")
            else:
                (positives if inv else negatives).append(
                    f"{feat}↓{'(better)' if inv else ''}")
        print(f"    C{c}:  POSITIVE → {', '.join(positives) or 'nothing notable'}")
        print(f"         NEGATIVE → {', '.join(negatives) or 'nothing notable'}")

    clusters = {
        f'Subjective_C{c}': [df.index[i] for i, l in enumerate(knn_labels) if l == c]
        for c in range(optimal_k)
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors = plt.cm.tab10.colors

    # Plot 1: Elbow + scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ks, inertias, 'bo-', linewidth=2)
    axes[0].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Curve – Subjective State Clustering'); axes[0].legend()
    for cid in range(optimal_k):
        mask = knn_labels == cid
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[cid],
                        label=f'C{cid} (n={mask.sum()})', s=70, edgecolors='k', lw=0.6)
    axes[1].set_xlabel(f'PC1 ({ev[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({ev[1]*100:.1f}% var)')
    axes[1].set_title(f'PCA scatter – {optimal_k} clusters (KNN-smoothed)')
    axes[1].legend(fontsize=9)
    plt.tight_layout(); plt.savefig('korean_subjective_cluster_plot.png', dpi=150); plt.close()

    # Plot 2: Loadings heatmap
    abs_max      = comp_df.abs().max(axis=0)
    top_features = abs_max.nlargest(len(feat_names)).index.tolist()
    heat_data    = comp_df[top_features]
    fig2, ax2    = plt.subplots(figsize=(max(10, len(top_features)*0.8), n_comp*1.1 + 1.5))
    im = ax2.imshow(heat_data.values, aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    ax2.set_xticks(range(len(top_features)))
    ax2.set_xticklabels(top_features, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(n_comp))
    ax2.set_yticklabels([f'PC{i+1}  ({ev[i]*100:.1f}%)' for i in range(n_comp)], fontsize=10)
    ax2.set_title('PCA Loadings Heatmap – Subjective State Features\n(red=positive, blue=negative)', fontsize=11)
    for i in range(n_comp):
        for j, f in enumerate(top_features):
            val = heat_data.iloc[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=8, color='white' if abs(val) > 0.35 else 'black')
    plt.colorbar(im, ax=ax2, label='Loading')
    plt.tight_layout(); plt.savefig('korean_pca_loadings_heatmap.png', dpi=150); plt.close()

    # Plot 3: Variance explained
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bars = ax3.bar([f'PC{i+1}' for i in range(n_comp)], ev*100,
                   color='steelblue', edgecolor='k', alpha=0.8)
    ax3t = ax3.twinx()
    ax3t.plot([f'PC{i+1}' for i in range(n_comp)], ev_c*100, 'ro-', lw=2, label='Cumulative %')
    ax3t.set_ylabel('Cumulative variance (%)', color='red'); ax3t.tick_params(axis='y', colors='red')
    ax3t.set_ylim(0, 100)
    for bar, v in zip(bars, ev*100):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=9)
    ax3.set_xlabel('Principal Component'); ax3.set_ylabel('Explained variance (%)')
    ax3.set_title('PCA Explained Variance – Korean Subjective States')
    ax3t.legend(loc='center right', fontsize=9)
    plt.tight_layout(); plt.savefig('korean_pca_variance_explained.png', dpi=150); plt.close()

    print(f"\n[subj]  Plots saved → korean_subjective_cluster_plot.png, "
          f"korean_pca_loadings_heatmap.png, korean_pca_variance_explained.png")
    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TASK-BASED  (composite score → KMeans k=3)
# ──────────────────────────────────────────────────────────────────────────────

def task_clusters(df: pd.DataFrame, n_clusters: int = 3,
                  random_state: int = 42):
    """
    Composite score (higher = better):
        score = norm(mean perf_pred)
              + norm(mean conc_diff)          [higher = better]
              - 0.5 * norm(mean too_fast)     [higher = worse]
              - 0.5 * norm(mean too_difficult)[higher = worse]
    """
    task_df = _aggregate(df, TASK_GROUPS)

    score = pd.Series(0.0, index=df.index)
    if 'Perf_Pred'    in task_df: score += _minmax(task_df['Perf_Pred'])
    if 'Conc_Diff'    in task_df: score += _minmax(task_df['Conc_Diff'])
    if 'Too_Fast'     in task_df: score -= 0.5 * _minmax(task_df['Too_Fast'])
    if 'Too_Difficult'in task_df: score -= 0.5 * _minmax(task_df['Too_Difficult'])
    score = _minmax(score)

    km     = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(score.values.reshape(-1, 1))

    # Sort by centroid ascending → Low / Mid / High
    order   = np.argsort(km.cluster_centers_.flatten())
    lbl_map = {old: new for new, old in enumerate(order)}
    labels  = np.array([lbl_map[l] for l in labels])
    names   = ['Low_Performer', 'Mid_Performer', 'High_Performer']

    clusters = {
        f'Task_{names[c]}': [df.index[i] for i, l in enumerate(labels) if l == c]
        for c in range(n_clusters)
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for c, name in enumerate(names):
        idx = [i for i, l in enumerate(labels) if l == c]
        ax.scatter(idx, score.values[idx], label=name,
                   color=plt.cm.tab10.colors[c], s=60, edgecolors='k')
    ax.set_xlabel('Subject index'); ax.set_ylabel('Composite task score (normalised)')
    ax.set_title('Task-Based Clustering – Korean Dataset'); ax.legend()
    plt.tight_layout(); plt.savefig('korean_task_cluster_plot.png', dpi=150); plt.close()
    print(f"[task]  Score plot saved → korean_task_cluster_plot.png")

    return clusters, score


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PRINT SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(title: str, clusters: dict):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for name, members in clusters.items():
        print(f"\n  [{name}]  ({len(members)} subjects)")
        for i in range(0, len(members), 10):
            print('    ' + '  '.join(members[i:i+10]))


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='korean.csv')
    args = parser.parse_args()

    df = load(args.csv)

    g_clusters = gender_clusters(df)
    print_summary('CLUSTER TYPE 1: GENDER', g_clusters)

    b_clusters = bciexp_clusters(df)
    print_summary('CLUSTER TYPE 2: BCI EXPERIENCE', b_clusters)

    s_clusters = subjective_clusters(df)
    print_summary('CLUSTER TYPE 3: SUBJECTIVE STATES (PCA + KMeans + KNN)', s_clusters)

    t_clusters, scores = task_clusters(df)
    print_summary('CLUSTER TYPE 4: TASK-BASED PERFORMANCE', t_clusters)

    # Save CSV
    result = pd.DataFrame(index=df.index)
    result['Gender_cluster']      = df['gender'].map({'M': 'Male', 'F': 'Female'})
    result['BCIExp_cluster']      = df['biofeedback_exp'].map({'N': 'Naive', 'Y': 'Experienced'})
    subj_lbl = {m: k for k, v in s_clusters.items() for m in v}
    result['Subjective_cluster']  = result.index.map(subj_lbl)
    task_lbl = {m: k for k, v in t_clusters.items() for m in v}
    result['Task_cluster']        = result.index.map(task_lbl)
    result['Task_score']          = scores.values
    result.to_csv('korean_subject_clusters.csv')
    print(f"\n[done]  Cluster table saved → korean_subject_clusters.csv")
    print(result.to_string())


if __name__ == '__main__':
    main()
