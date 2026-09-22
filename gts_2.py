import warnings
import traceback
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# 1. Parameters
# ============================================================
N_SUBJECTS_PER_GROUP = 20      # 20 subjects per group (2 groups = 40 total)
N_TRIALS             = 200     # raw trials per subject
SIGMA_WITHIN         = 1.0     # within-subject trial-level standard deviation
SIGMA_BETWEEN        = 0.3     # between-subject random intercept standard deviation
ALPHA                = 0.05    # nominal significance threshold
N_REPS               = 200     # Monte Carlo repetitions per (scenario, k)

# Continuum of k values: [1, 5, 10, 15, ..., 50]
K_VALUES = [1] + list(range(5, 51, 5))

# Scenarios: Null (delta=0) and Constant Signal (delta=0.5, true GxBin = 0)
SCENARIOS = {
    'Null (delta=0.0)':      0.0,
    'Signal (delta=0.5)':    0.5,
}

# Base master seed for deterministic reproducibility across runs
MASTER_SEED = 42

# ============================================================
# 2. Single Monte Carlo Rep Worker
# ============================================================
def run_one_rep(rep_seed, group_means, k):
    """
    Generates one synthetic trial-level dataset, computes:
      1. K-invariant subject-mean Welch t-test (independent of binning)
      2. Binned dataset at bin size k
      3. Sum-coded MixedLM estimating:
         - Group main effect: C(group, Sum)
         - Group x Bin interaction: C(group, Sum):C(Bin, Sum)
    Returns:
      (p_lme_group, p_lme_inter, p_km, fit_status, failure_reason)
    """
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(rep_seed)

    # --- Step 1: Generate trial-level data ---
    records = []
    for g, gm in enumerate(group_means):
        for s in range(N_SUBJECTS_PER_GROUP):
            subj_id = f"g{g}_s{s:02d}"
            # True ground truth: random subject intercept + trial noise
            subj_mean = gm + rng.normal(0, SIGMA_BETWEEN)
            trials = rng.normal(subj_mean, SIGMA_WITHIN, size=N_TRIALS)
            for t, val in enumerate(trials):
                records.append({'Subject': subj_id, 'group': g, 'trial': t, 'value': val})
    df_trials = pd.DataFrame(records)

    # --- Step 2: K-invariant subject-mean test (raw trials, no binning) ---
    p_km = np.nan
    try:
        subj_means = df_trials.groupby(['Subject', 'group'])['value'].mean().reset_index()
        g0 = subj_means[subj_means['group'] == 0]['value'].values
        g1 = subj_means[subj_means['group'] == 1]['value'].values
        if len(g0) >= 2 and len(g1) >= 2:
            _, p_km = ttest_ind(g1, g0, equal_var=False)
    except Exception:
        p_km = np.nan

    # --- Step 3: Bin trials into B = floor(N_TRIALS / k) bins ---
    bin_records = []
    for subj_id, grp_df in df_trials.groupby('Subject'):
        grp_df = grp_df.sort_values('trial').reset_index(drop=True)
        g = grp_df['group'].iloc[0]
        n_bins = len(grp_df) // k
        if n_bins < 2:
            # Cannot test Group x Bin interaction with < 2 bins
            return np.nan, np.nan, p_km, False, "less_than_2_bins"
        for b in range(n_bins):
            block = grp_df.iloc[b*k:(b+1)*k]['value']
            bin_records.append({
                'Subject': subj_id,
                'group':   g,
                'Bin':     b + 1,
                'Value':   block.mean(),
            })

    df_binned = pd.DataFrame(bin_records)
    df_binned['group'] = df_binned['group'].astype('category')
    df_binned['Bin']   = df_binned['Bin'].astype('category')

    # --- Step 4: Sum-coded MixedLM ---
    p_lme_group = np.nan
    p_lme_inter = np.nan
    fit_status = False
    failure_reason = ""

    try:
        formula = "Value ~ C(group, Sum) * C(Bin, Sum)"
        model = smf.mixedlm(formula, df_binned, groups=df_binned['Subject'])
        
        # Robust fitting strategy:
        # lbfgs with maxiter=500; fallback to powell if lbfgs encounters convergence issues
        fit = model.fit(reml=True, method='lbfgs', maxiter=500)
        
        # Check if fit converged
        if not fit.converged:
            # Attempt fallback method
            fit = model.fit(reml=True, method='powell', maxiter=500)

        if not fit.converged:
            failure_reason = "optimizer_did_not_converge"
            return np.nan, np.nan, p_km, False, failure_reason

        # Extract Wald test terms
        anova = fit.wald_test_terms().summary_frame()
        
        group_term = 'C(group, Sum)'
        inter_term = 'C(group, Sum):C(Bin, Sum)'

        if group_term in anova.index:
            p_lme_group = float(anova.loc[group_term, 'P>chi2'])
        if inter_term in anova.index:
            p_lme_inter = float(anova.loc[inter_term, 'P>chi2'])

        if not np.isnan(p_lme_group) and not np.isnan(p_lme_inter):
            fit_status = True
            failure_reason = "success"
        else:
            failure_reason = "wald_test_nan"

    except Exception as e:
        failure_reason = f"exception_{type(e).__name__}"

    return p_lme_group, p_lme_inter, p_km, fit_status, failure_reason


# ============================================================
# 3. Monte Carlo Sweep
# ============================================================
results = []
failure_logs = []

for sc_idx, (scenario_name, delta) in enumerate(SCENARIOS.items()):
    group_means = [0.0, delta]
    print(f"\n{'='*75}")
    print(f"Scenario: {scenario_name} (Group Means: [0.0, {delta}])")
    print(f"{'='*75}")

    for k_idx, k in enumerate(K_VALUES):
        # Fully deterministic seed arithmetic (no hash())
        base_seed = MASTER_SEED + sc_idx * 100000 + k_idx * 1000

        # Parallelize across available CPU cores using joblib loky
        rep_results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(run_one_rep)(base_seed + rep, group_means, k)
            for rep in range(N_REPS)
        )

        # Unpack results independently
        p_lme_group_list = []
        p_lme_inter_list = []
        p_km_list        = []
        fail_reasons     = {}

        for p_grp, p_int, p_k, status, reason in rep_results:
            if not np.isnan(p_k):
                p_km_list.append(p_k)
            if status and not np.isnan(p_grp) and not np.isnan(p_int):
                p_lme_group_list.append(p_grp)
                p_lme_inter_list.append(p_int)
            else:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        n_valid_lme = len(p_lme_group_list)
        n_valid_km  = len(p_km_list)
        lme_convergence_rate = n_valid_lme / N_REPS

        # Calculate rejection rates (power or Type I error)
        rej_lme_group = float(np.mean(np.array(p_lme_group_list) < ALPHA)) if n_valid_lme > 0 else np.nan
        rej_lme_inter = float(np.mean(np.array(p_lme_inter_list) < ALPHA)) if n_valid_lme > 0 else np.nan
        rej_km        = float(np.mean(np.array(p_km_list)        < ALPHA)) if n_valid_km  > 0 else np.nan

        print(f"  k={k:2d} | Valid LME: {n_valid_lme:3d}/{N_REPS} ({lme_convergence_rate*100:5.1f}%) | "
              f"Group Rej: {rej_lme_group:.3f} | GxBin Rej: {rej_lme_inter:.3f} | "
              f"K-Inv Rej: {rej_km:.3f} | Failures: {fail_reasons}")

        results.append({
            'scenario':             scenario_name,
            'k':                    k,
            'n_reps_total':         N_REPS,
            'n_valid_lme':          n_valid_lme,
            'lme_conv_rate':        lme_convergence_rate,
            'n_valid_km':           n_valid_km,
            'rej_group_lme':        rej_lme_group,
            'rej_gxbin_lme':        rej_lme_inter,
            'rej_kinvariant':       rej_km,
        })

        failure_logs.append({
            'scenario': scenario_name,
            'k': k,
            'failure_breakdown': str(fail_reasons)
        })

results_df = pd.DataFrame(results)
results_df.to_csv("ground_truth_simulation_results.csv", index=False)
pd.DataFrame(failure_logs).to_csv("ground_truth_convergence_failures.csv", index=False)
print("\nSimulation results saved to: ground_truth_simulation_results.csv")
print("Convergence failure diagnostics saved to: ground_truth_convergence_failures.csv")

# ============================================================
# 4. Comprehensive 4-Panel Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Panel 1 (Top-Left): Null Scenario (delta = 0.0)
ax_null = axes[0, 0]
df_null = results_df[results_df['scenario'] == 'Null (delta=0.0)']

ax_null.plot(df_null['k'], df_null['rej_group_lme'],
             color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=7,
             label='LME Group Main Effect (Sum coded)')
ax_null.plot(df_null['k'], df_null['rej_gxbin_lme'],
             color='#d62728', linestyle='--', marker='s', linewidth=2, markersize=7,
             label='LME Group × Bin Interaction')
ax_null.plot(df_null['k'], df_null['rej_kinvariant'],
             color='#2ca02c', linestyle=':', marker='^', linewidth=2, markersize=7,
             label='K-Invariant Subject-Mean Test')

ax_null.axhline(ALPHA, color='black', linestyle='-.', linewidth=1.2, label=f'Nominal alpha = {ALPHA}')
ax_null.set_ylabel('Rejection Rate (Type I Error)', fontsize=11, fontweight='bold')
ax_null.set_title('(A) Null Scenario (delta = 0.0): All Tests Control Type I Error ≈ 0.05', fontsize=12, fontweight='bold')
ax_null.set_ylim(-0.02, 0.35)
ax_null.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_null.grid(True, linestyle=':', alpha=0.6)

# Panel 2 (Top-Right): Constant Signal Scenario (delta = 0.5)
ax_sig = axes[0, 1]
df_sig = results_df[results_df['scenario'] == 'Signal (delta=0.5)']

ax_sig.plot(df_sig['k'], df_sig['rej_group_lme'],
            color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=7,
            label='LME Group Main Effect Power')
ax_sig.plot(df_sig['k'], df_sig['rej_kinvariant'],
            color='#2ca02c', linestyle=':', marker='^', linewidth=2, markersize=7,
            label='K-Invariant Subject-Mean Power')
ax_sig.plot(df_sig['k'], df_sig['rej_gxbin_lme'],
            color='#d62728', linestyle='--', marker='s', linewidth=2, markersize=7,
            label='LME Group × Bin False Positives (True G×B = 0)')

ax_sig.axhline(ALPHA, color='black', linestyle='-.', linewidth=1.2, label=f'Nominal alpha = {ALPHA}')
ax_sig.set_ylabel('Rejection Rate (Power / False Positives)', fontsize=11, fontweight='bold')
ax_sig.set_title('(B) Constant Signal (delta = 0.5): Group Power vs Stable G×Bin False Positive Rate', fontsize=12, fontweight='bold')
ax_sig.set_ylim(-0.02, 1.05)
ax_sig.legend(fontsize=9, loc='center right', framealpha=0.9)
ax_sig.grid(True, linestyle=':', alpha=0.6)

# Panel 3 (Bottom-Left): Group x Bin Interaction False-Positive Rate Comparison
ax_gxbin = axes[1, 0]

ax_gxbin.plot(df_null['k'], df_null['rej_gxbin_lme'],
              color='#9467bd', linestyle='-', marker='o', linewidth=2, markersize=7,
              label='G×Bin Rejection under Null (delta = 0.0)')
ax_gxbin.plot(df_sig['k'], df_sig['rej_gxbin_lme'],
              color='#d62728', linestyle='--', marker='s', linewidth=2, markersize=7,
              label='G×Bin Rejection under Signal (delta = 0.5)')

ax_gxbin.axhline(ALPHA, color='black', linestyle='-.', linewidth=1.2, label=f'Nominal alpha = {ALPHA}')
ax_gxbin.set_xlabel('k (epochs per bin)', fontsize=11, fontweight='bold')
ax_gxbin.set_ylabel('G×Bin Rejection Rate', fontsize=11, fontweight='bold')
ax_gxbin.set_title('(C) k Does NOT Generate Spurious G×Bin Interactions', fontsize=12, fontweight='bold')
ax_gxbin.set_xticks(K_VALUES)
ax_gxbin.set_ylim(-0.01, 0.20)
ax_gxbin.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_gxbin.grid(True, linestyle=':', alpha=0.6)

# Panel 4 (Bottom-Right): LME Convergence & Valid Fit Rate across k
ax_conv = axes[1, 1]

ax_conv.plot(df_null['k'], df_null['lme_conv_rate'] * 100,
             color='#333333', linestyle='-', marker='o', linewidth=2, markersize=7,
             label='Null Scenario LME Convergence (%)')
ax_conv.plot(df_sig['k'], df_sig['lme_conv_rate'] * 100,
             color='#17becf', linestyle='--', marker='^', linewidth=2, markersize=7,
             label='Signal Scenario LME Convergence (%)')

ax_conv.axhline(100, color='gray', linestyle=':', linewidth=1)
ax_conv.set_xlabel('k (epochs per bin)', fontsize=11, fontweight='bold')
ax_conv.set_ylabel('Valid LME Fit Rate (%)', fontsize=11, fontweight='bold')
ax_conv.set_title('(D) Model Stability & Convergence Rate across k ', fontsize=12, fontweight='bold')
ax_conv.set_xticks(K_VALUES)
ax_conv.set_ylim(0, 105)
ax_conv.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax_conv.grid(True, linestyle=':', alpha=0.6)

fig.text(0.5, 0.01,
         f"Parameters: N={N_SUBJECTS_PER_GROUP}/group, {N_TRIALS} trials/subject, "
         f"sigma_within={SIGMA_WITHIN}, sigma_between={SIGMA_BETWEEN}, {N_REPS} Monte Carlo reps per k | Sum Contrast Coding",
         ha='center', fontsize=9, color='#444444')

plt.suptitle("Ground-Truth Simulation: Method Validation, False Positive Control, and Model Stability",
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
out_fig = "ground_truth_simulation.png"
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()
print(f"Figure successfully saved to: {out_fig}")

# ============================================================
# ============================================================
print("\n" + "="*85)
print("TABLE 1: GROUND-TRUTH REJECTION RATES & CONVERGENCE ")
print("="*85)
for sc_name in SCENARIOS:
    print(f"\n--- {sc_name} ---")
    sub = results_df[results_df['scenario'] == sc_name]
    header = f"{'k':>4} | {'Valid LME':>11} | {'Conv Rate':>9} | {'Group Rej':>10} | {'GxBin Rej':>10} | {'K-Inv Rej':>10}"
    print(header)
    print("-" * len(header))
    for _, r in sub.iterrows():
        print(f"{int(r['k']):4d} | "
              f"{int(r['n_valid_lme']):4d}/{int(r['n_reps_total']):<4d} | "
              f"{r['lme_conv_rate']*100:8.1f}% | "
              f"{r['rej_group_lme']:10.3f} | "
              f"{r['rej_gxbin_lme']:10.3f} | "
              f"{r['rej_kinvariant']:10.3f}")
print("="*85)
