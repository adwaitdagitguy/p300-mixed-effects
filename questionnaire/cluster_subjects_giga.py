"""
cluster_subjects.py
====================
Clusters GigaDB BCI subjects along 4 independent axes:

  1. Gender       – Sex == 0 vs Sex == 1
  2. BCI Exp      – BCI_exp == 0 vs BCI_exp > 0
  3. Subjective   – PCA on RE features, optimal k via KElbowVisualizer (KMeans)
  4. Task-Based   – composite score → KMeans (k=3: low / mid / high performers)

Usage
-----
    python cluster_subjects.py [--csv PATH]

Default CSV path: giga.csv  (place in the same directory, or pass --csv)
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

# ──────────────────────────────────────────────────────────────────────────────
# 0.  LOAD & TRANSPOSE
# ──────────────────────────────────────────────────────────────────────────────

def load_and_transpose(csv_path: str) -> pd.DataFrame:
    """
    Raw file  →  (54 subjects) × (features) DataFrame.
    Drops the 'average' and 'std' columns that are not subjects.
    Coerces all values to numeric; mode-imputes remaining NaN.
    """
    raw = pd.read_csv(csv_path)

    # Keep only subject columns (drop 'average', 'std', and any unnamed cols)
    subject_cols = [c for c in raw.columns if str(c).startswith("subject")]
    raw = raw[["Feature"] + subject_cols]

    # Drop rows with no Feature label (trailing NaN rows at the bottom)
    raw = raw.dropna(subset=["Feature"])

    # Transpose: rows → subjects, columns → features
    df = raw.set_index("Feature")[subject_cols].T
    df.index.name = "subject"
    df.index = subject_cols  # e.g. ["subject1", ..., "subject54"]

    # Coerce everything to numeric (handles stray strings like '0', '2')
    df = df.apply(pd.to_numeric, errors="coerce")

    # Mode imputation column-wise
    for col in df.columns:
        if df[col].isna().any():
            col_mode = scipy_mode(df[col].dropna(), keepdims=True).mode
            fill_val = col_mode[0] if len(col_mode) > 0 else 0
            df[col].fillna(fill_val, inplace=True)

    print(f"[load]  Shape after transpose & impute: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GENDER CLUSTERS
# ──────────────────────────────────────────────────────────────────────────────

def gender_clusters(df: pd.DataFrame) -> dict:
    """
    Cluster 0: Sex == 0  (female, per GigaDB coding)
    Cluster 1: Sex == 1  (male)
    """
    sex = df["Sex"].astype(int)
    clusters = {
        "Gender_0 (Female)": list(sex[sex == 0].index),
        "Gender_1 (Male)":   list(sex[sex == 1].index),
    }
    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# 2.  BCI EXPERIENCE CLUSTERS
# ──────────────────────────────────────────────────────────────────────────────

def bciexp_clusters(df: pd.DataFrame) -> dict:
    """
    Cluster 0: BCI_exp == 0  (no prior experience)
    Cluster 1: BCI_exp  > 0  (some prior experience)
    """
    bci = df["BCI_exp"].astype(float)
    clusters = {
        "BCIExp_0 (Naive)":       list(bci[bci == 0].index),
        "BCIExp_1 (Experienced)": list(bci[bci  > 0].index),
    }
    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SUBJECTIVE STATE CLUSTERS  (PCA + KMeans, k chosen by elbow / inertia)
# ──────────────────────────────────────────────────────────────────────────────

SUBJ_KEYWORDS = [
    "relaxed", "exciting", "attention", "eyestate",
    "somnolence", "body_state", "mental_state", "concentration",
]

def _get_subjective_cols(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if any(c.lower().startswith(kw) or f"_{kw.split('_')[0]}" in c.lower()
               for kw in SUBJ_KEYWORDS)
    ]

def _optimal_k_elbow(X_scaled: np.ndarray,
                     k_range: range = range(2, 8),
                     random_state: int = 42) -> int:
    """
    Compute inertia for k in k_range; pick k at the elbow using the
    'kneedle' heuristic (largest second-difference of inertia).
    Falls back to k=3 if the curve is monotone without a clear elbow.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Second difference to find sharpest bend
    inertias = np.array(inertias)
    diffs2 = np.diff(np.diff(inertias))
    best_idx = int(np.argmax(diffs2)) + 1   # offset for two diff() calls
    optimal_k = list(k_range)[best_idx]
    return optimal_k, list(k_range), inertias

def subjective_clusters(df: pd.DataFrame,
                        n_pca_components: int = 5,
                        n_top_loadings: int = 5,
                        random_state: int = 42) -> dict:
    """
    Steps:
      a) Aggregate all session-wise RE columns -> 8 mean scores per subject
         (Relaxed, Exciting, Attention, EyeState, Somnolence,
          Body_State, Mental_State, Concentration)
      b) StandardScaler -> PCA on the 8 aggregated features
      c) Print explained variance + loadings per PC for interpretation
      d) Elbow method on KMeans inertia to choose optimal k
      e) KMeans clustering in PCA space
      f) KNN (k=5) smooth label mapper to validate boundary assignments
      g) Print per-cluster centroid profiles in original aggregated space
      h) Save: elbow+scatter plot, loadings heatmap, variance chart
    """
    # Aggregate session columns -> one mean score per feature type
    GROUPS = {
        "Relaxed":       [c for c in df.columns if "relaxed"       in c.lower()],
        "Exciting":      [c for c in df.columns if "exciting"      in c.lower()],
        "Attention":     [c for c in df.columns if "attention"     in c.lower()],
        "EyeState":      [c for c in df.columns if "eyestate"      in c.lower()],
        "Somnolence":    [c for c in df.columns if "somnolence"    in c.lower()],
        "Body_State":    [c for c in df.columns if "body_state"    in c.lower()],
        "Mental_State":  [c for c in df.columns if "mental_state"  in c.lower()],
        "Concentration": [c for c in df.columns if "concentration" in c.lower()],
    }
    agg = {}
    for grp, gcols in GROUPS.items():
        gcols = [c for c in gcols if c in df.columns]
        if gcols:
            agg[grp] = df[gcols].mean(axis=1)
    sub_df = pd.DataFrame(agg, index=df.index)
    sub_df = sub_df.fillna(sub_df.median())
    feat_names = sub_df.columns.tolist()   # 8 clean group names

    print(f"[subj]  Aggregated to {len(feat_names)} mean scores: {feat_names}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_df.values)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(2, int(np.searchsorted(cum_var, 0.90) + 1))
    n_comp = min(n_comp, n_pca_components, X_scaled.shape[1])

    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    ev   = pca.explained_variance_ratio_
    ev_c = np.cumsum(ev)

    # ── Print PCA interpretation ──────────────────────────────────────────────
    print(f"\n[subj]  PCA components used: {n_comp}")
    print(f"\n  {'PC':<6}  {'Var %':>7}  {'Cum %':>7}  Top-{n_top_loadings} features (loading)")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*50}")
    comp_df = pd.DataFrame(pca.components_, columns=feat_names,
                           index=[f"PC{i+1}" for i in range(n_comp)])

    # Human-readable axis labels derived from dominant loadings
    pc_labels = []
    for i, pc in enumerate(comp_df.index):
        row = comp_df.loc[pc]
        top_idx  = row.abs().nlargest(n_top_loadings).index
        top_vals = row[top_idx]
        loading_str = "  ".join(
            f"{f} ({v:+.3f})" for f, v in zip(top_idx, top_vals)
        )
        print(f"  {pc:<6}  {ev[i]*100:>6.1f}%  {ev_c[i]*100:>6.1f}%  {loading_str}")

        # Derive a short semantic label for the PC axis
        dominant = top_idx[0].lower()
        if "mental" in dominant or "body" in dominant:
            label = "Overall cognitive/body load"
        elif "relaxed" in dominant:
            label = "Relaxation level"
        elif "exciting" in dominant or "attention" in dominant:
            label = "Arousal / engagement"
        elif "somnolence" in dominant or "eyestate" in dominant:
            label = "Alertness / wakefulness"
        elif "concentration" in dominant:
            label = "Concentration"
        else:
            label = dominant.split("_")[0].capitalize()
        pc_labels.append(f"{pc}: {label} ({ev[i]*100:.1f}%)")

    print(f"\n  Semantic axis summary:")
    for lbl in pc_labels:
        print(f"    • {lbl}")

    # ── Optimal k via elbow ───────────────────────────────────────────────────
    k_range = range(2, min(8, len(df)))
    optimal_k, ks, inertias = _optimal_k_elbow(X_pca, k_range, random_state)
    print(f"\n[subj]  Optimal k (elbow): {optimal_k}")

    # ── KMeans ───────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    km_labels = km.fit_predict(X_pca)

    # KNN smoothing
    knn = KNeighborsClassifier(n_neighbors=min(5, len(df) - 1))
    knn.fit(X_pca, km_labels)
    knn_labels = knn.predict(X_pca)

    # ── Cluster centroid profiles in aggregated z-score space ────────────────
    # feat_names are already the 8 group names, so no further grouping needed
    print(f"\n  Cluster centroid profiles (mean z-score of aggregated features):")
    X_scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=feat_names)
    X_scaled_df["_cluster"] = knn_labels
    cluster_means = X_scaled_df.groupby("_cluster")[feat_names].mean()

    header = f"  {'Feature':<16}" + "".join(f"  C{c:>5}" for c in range(optimal_k))
    print(header)
    print("  " + "-" * (16 + optimal_k * 8))
    for feat in feat_names:
        row = f"  {feat:<16}" + "".join(
            f"  {cluster_means.loc[c, feat]:+.3f}" for c in range(optimal_k)
        )
        print(row)

    # Also print a one-line interpretation per cluster
    # Scale polarity (based on dataset author's coding):
    #   INVERTED (1=good, higher=worse): Somnolence, EyeState, Body_State, Mental_State
    #   NORMAL   (1=low, higher=more):   Relaxed, Exciting, Attention, Concentration
    INVERTED = {"Somnolence", "EyeState", "Body_State", "Mental_State"}

    print(f"\n  Scale note: Somnolence/EyeState/Body_State/Mental_State are INVERTED")
    print(f"  (1=good/alert, higher=worse). Interpretations below account for this.\n")
    print(f"  Cluster interpretations (based on centroid z-scores):")
    for c in range(optimal_k):
        centroid = cluster_means.loc[c]
        positives = []
        negatives = []
        for feat, val in centroid.items():
            if abs(val) <= 0.3:
                continue
            is_inv = feat in INVERTED
            if val > 0:
                # High raw score: good for normal features, bad for inverted
                if is_inv:
                    negatives.append(f"{feat}↑(worse)")
                else:
                    positives.append(f"{feat}↑")
            else:
                # Low raw score: bad for normal features, good for inverted
                if is_inv:
                    positives.append(f"{feat}↓(better)")
                else:
                    negatives.append(f"{feat}↓")
        pos_str = ", ".join(positives) if positives else "nothing notably positive"
        neg_str = ", ".join(negatives) if negatives else "nothing notably negative"
        print(f"    C{c}:  POSITIVE → {pos_str}")
        print(f"         NEGATIVE → {neg_str}")

    # ── Build clusters dict ───────────────────────────────────────────────────
    clusters = {}
    for cluster_id in range(optimal_k):
        members = [df.index[i] for i, lbl in enumerate(knn_labels) if lbl == cluster_id]
        clusters[f"Subjective_C{cluster_id}"] = members

    # ── PLOT 1: Elbow + PCA scatter ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ks, inertias, "bo-", linewidth=2)
    axes[0].axvline(optimal_k, color="red", linestyle="--",
                    label=f"Optimal k={optimal_k}")
    axes[0].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[0].set_ylabel("Inertia", fontsize=11)
    axes[0].set_title("Elbow Curve – Subjective State Clustering", fontsize=12)
    axes[0].legend()

    colors = plt.cm.tab10.colors
    for cid in range(optimal_k):
        mask = knn_labels == cid
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=colors[cid], label=f"C{cid} (n={mask.sum()})",
                        s=70, edgecolors="k", linewidths=0.6)
    axes[1].set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
    axes[1].set_title(f"PCA scatter – {optimal_k} clusters (KNN-smoothed)", fontsize=12)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("subjective_cluster_plot.png", dpi=150)
    plt.close()

    # ── PLOT 2: Loadings heatmap ──────────────────────────────────────────────
    TOP_N = 8   # show top-N features per PC (by max absolute loading across PCs)
    abs_max = comp_df.abs().max(axis=0)
    top_features = abs_max.nlargest(TOP_N * n_comp).index.tolist()
    # deduplicate while preserving order
    seen = set()
    top_features = [f for f in top_features if not (f in seen or seen.add(f))]
    top_features = top_features[:min(40, len(top_features))]

    heat_data = comp_df[top_features]

    fig2, ax2 = plt.subplots(figsize=(max(12, len(top_features) * 0.55), n_comp * 1.1 + 1.5))
    im = ax2.imshow(heat_data.values, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax2.set_xticks(range(len(top_features)))
    ax2.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(n_comp))
    ax2.set_yticklabels(
        [f"PC{i+1}  ({ev[i]*100:.1f}%)" for i in range(n_comp)], fontsize=10
    )
    ax2.set_title("PCA Loadings Heatmap – Subjective State Features\n"
                  "(colour = loading magnitude; red = positive, blue = negative)",
                  fontsize=11)
    for i in range(n_comp):
        for j, f in enumerate(top_features):
            val = heat_data.iloc[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if abs(val) > 0.22 else "black")
    plt.colorbar(im, ax=ax2, label="Loading")
    plt.tight_layout()
    plt.savefig("pca_loadings_heatmap.png", dpi=150)
    plt.close()
    print(f"\n[subj]  Plots saved → subjective_cluster_plot.png, pca_loadings_heatmap.png")

    # ── PLOT 3: Variance explained bar chart ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bars = ax3.bar([f"PC{i+1}" for i in range(n_comp)],
                   ev * 100, color="steelblue", edgecolor="k", alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot([f"PC{i+1}" for i in range(n_comp)], ev_c * 100,
                  "ro-", linewidth=2, label="Cumulative %")
    ax3_twin.set_ylabel("Cumulative variance (%)", color="red", fontsize=10)
    ax3_twin.tick_params(axis="y", colors="red")
    ax3_twin.set_ylim(0, 100)
    for bar, v in zip(bars, ev * 100):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_xlabel("Principal Component", fontsize=11)
    ax3.set_ylabel("Explained variance (%)", fontsize=11)
    ax3.set_title("PCA Explained Variance – Subjective States", fontsize=12)
    ax3_twin.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig("pca_variance_explained.png", dpi=150)
    plt.close()
    print(f"[subj]  Variance chart saved → pca_variance_explained.png")

    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TASK-BASED CLUSTERS  (composite score → KMeans k=3)
# ──────────────────────────────────────────────────────────────────────────────

TASK_KEYWORDS = ["bci_performance", "trials_missed", "nodded_off"]

def _get_task_cols(df: pd.DataFrame) -> tuple:
    perf_cols    = [c for c in df.columns if "bci_performance" in c.lower()]
    missed_cols  = [c for c in df.columns if "trials_missed"   in c.lower()]
    nodded_cols  = [c for c in df.columns if "nodded_off"      in c.lower()]
    return perf_cols, missed_cols, nodded_cols

def _minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)

def task_clusters(df: pd.DataFrame,
                  n_clusters: int = 3,
                  random_state: int = 42) -> dict:
    """
    Composite score (0–1, higher = better performer):
        score = mean(BCI_performance_sessions)        [higher → better]
              - mean(trials_missed_sessions)  * 0.5  [higher → worse]
              - mean(nodded_off_sessions)     * 0.3  [higher → worse]

    All sub-scores are min-max normalised before combining.
    KMeans k=3: Low / Mid / High performer clusters.
    """
    perf_cols, missed_cols, nodded_cols = _get_task_cols(df)

    score = pd.Series(0.0, index=df.index)
    if perf_cols:
        score += _minmax(df[perf_cols].mean(axis=1))
    if missed_cols:
        score -= 0.5 * _minmax(df[missed_cols].mean(axis=1))
    if nodded_cols:
        score -= 0.3 * _minmax(df[nodded_cols].mean(axis=1))

    # Normalise composite to [0, 1]
    score = _minmax(score)

    # KMeans on 1D score
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(score.values.reshape(-1, 1))

    # Sort cluster IDs by centroid (ascending score → Low, Mid, High)
    centroid_order = np.argsort(km.cluster_centers_.flatten())
    label_map = {old: new for new, old in enumerate(centroid_order)}
    labels = np.array([label_map[l] for l in labels])
    label_names = ["Low_Performer", "Mid_Performer", "High_Performer"]

    clusters = {}
    for cluster_id, name in enumerate(label_names[:n_clusters]):
        members = [df.index[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        clusters[f"Task_{name}"] = members

    # ── Save score distribution plot ──
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors
    for cluster_id, name in enumerate(label_names[:n_clusters]):
        members_idx = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        ax.scatter(members_idx, score.values[members_idx],
                   label=name, color=colors[cluster_id], s=60, edgecolors="k")
    ax.set_xlabel("Subject index")
    ax.set_ylabel("Composite task score (normalised)")
    ax.set_title("Task-Based Clustering – Composite Score")
    ax.legend()
    plot_path = "task_cluster_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[task]  Score plot saved → {plot_path}")

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
        # Pretty-print in rows of 10
        for i in range(0, len(members), 10):
            print("    " + "  ".join(members[i:i+10]))


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="giga.csv",
                        help="Path to the GigaDB subject metadata CSV")
    args = parser.parse_args()

    # ── Load ──
    df = load_and_transpose(args.csv)

    # ── Cluster 1: Gender ──
    g_clusters = gender_clusters(df)
    print_summary("CLUSTER TYPE 1: GENDER", g_clusters)

    # ── Cluster 2: BCI Experience ──
    b_clusters = bciexp_clusters(df)
    print_summary("CLUSTER TYPE 2: BCI EXPERIENCE", b_clusters)

    # ── Cluster 3: Subjective States ──
    s_clusters = subjective_clusters(df)
    print_summary("CLUSTER TYPE 3: SUBJECTIVE STATES (PCA + KMeans + KNN)", s_clusters)

    # ── Cluster 4: Task-Based ──
    t_clusters, scores = task_clusters(df)
    print_summary("CLUSTER TYPE 4: TASK-BASED PERFORMANCE", t_clusters)

    # ── Save all cluster assignments to CSV ──
    result = pd.DataFrame(index=df.index)
    result["Gender_cluster"]    = df["Sex"].astype(int).map({0: "Female", 1: "Male"})
    result["BCIExp_cluster"]    = (df["BCI_exp"] > 0).map({False: "Naive", True: "Experienced"})

    subj_label = {}
    for cname, members in s_clusters.items():
        for m in members:
            subj_label[m] = cname
    result["Subjective_cluster"] = result.index.map(subj_label)

    task_label = {}
    for cname, members in t_clusters.items():
        for m in members:
            task_label[m] = cname
    result["Task_cluster"]      = result.index.map(task_label)
    result["Task_score"]        = scores.values

    out_csv = "subject_clusters.csv"
    result.to_csv(out_csv)
    print(f"\n[done]  Full cluster table saved → {out_csv}")
    print(result.to_string())


if __name__ == "__main__":
    main()
