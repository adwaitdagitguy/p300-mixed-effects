# -*- coding: utf-8 -*-
"""
giga_grand_erp.py

Computes grand-averaged ERPs for the Giga dataset from preprocessed
.set/.fdt files stored on Azure Blob Storage.

Key Updates:
  1. Azure Blob Storage integration (ClientSecretCredential + BlobServiceClient)
  2. STRICT explicit file naming: exactly mirrors the reference code:
       clean_epoched_subj{subj}_sess{sess}_EEG_ERP_{split}.set
     under container prefix: 'giga_preprocessed' (54 subjects x 2 sessions x 2 splits = 216 files)
     This excludes any extraneous files, raw files, or files from other experiments.
  3. Corrected label extraction:
       Original stimulus codes in Giga: 1 = target, 2 = non-target
       Inverted event_id mapping ensures:
         target_idx    = [i for i, l in enumerate(labels) if l == 1]
         nontarget_idx = [i for i, l in enumerate(labels) if l == 2]
  4. Generates single, publication-quality ROI averaged ERP plot (Target vs Non-target)
     using canonical P300 ROI channels: ['Pz', 'CPz', 'Cz', 'P3', 'P4', 'POz', 'Fz']
  5. Saves:
     - giga_grand_erp_roi.png: Single ROI averaged plot
     - giga_grand_erp_grid.png: Grid plot of all channels
     - erp_channels/erp_<ch>.png: Individual channel waveforms
     - giga_grand_erp.npz: Saved ERP arrays for downstream testing
"""

import os
import gc
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from joblib import Parallel, delayed
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

mne.set_log_level('WARNING')

# =============================================================
# 0. Configuration & Azure Credentials
# =============================================================
AZURE_TENANT_ID      = os.environ.get('AZURE_TENANT_ID')
AZURE_CLIENT_ID      = os.environ.get('AZURE_CLIENT_ID')
AZURE_CLIENT_SECRET  = os.environ.get('AZURE_CLIENT_SECRET')
AZURE_STORAGE_ACCOUNT   = os.environ.get('AZURE_STORAGE_ACCOUNT')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER')

missing_env = [k for k, v in [
    ('AZURE_TENANT_ID', AZURE_TENANT_ID),
    ('AZURE_CLIENT_ID', AZURE_CLIENT_ID),
    ('AZURE_CLIENT_SECRET', AZURE_CLIENT_SECRET),
    ('AZURE_STORAGE_ACCOUNT', AZURE_STORAGE_ACCOUNT),
    ('AZURE_STORAGE_CONTAINER', AZURE_STORAGE_CONTAINER)
] if not v]

if missing_env:
    raise RuntimeError(f"Missing required Azure environment variables: {', '.join(missing_env)}")

# Matches giga_snr_from_set.py exact path on Azure: 'giga_preprocessed'
AZURE_INPUT_PREFIX = os.environ.get('GIGA_INPUT_PREFIX', 'giga_preprocessed')
PLOT_DIR           = os.environ.get('PLOT_DIR', os.getcwd())
N_JOBS             = int(os.environ.get('N_JOBS', 16))

# Exact lists from reference code: 54 subjects, 2 sessions, 2 splits = 216 files
subjects_list = [f"{i:02d}" for i in range(1, 55)]   # 01 to 54
sessions_list = ['01', '02']
split_list    = ['train', 'test']

T_START = 0.0   # Giga epochs start at 0ms — no pre-stimulus baseline
T_END   = 0.8

# ROI channels matching dataset config in testing_gb_interaction_sept19.py (20 channels)
ROI_CHANNELS = [
    'fc1', 'fc2', 'cz', 'c3', 'cp5', 'cp1', 'cp2', 'pz',
    'o1', 'oz', 'o2', 'c1', 'c2', 'cp3', 'cpz', 'cp4',
    'p1', 'p2', 'poz', 'c4',
]

tmp_dir = tempfile.mkdtemp(prefix="giga_erp_")
os.makedirs(os.path.join(PLOT_DIR, "erp_channels"), exist_ok=True)

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

def savefig(name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")

# =============================================================
# 1. Parallel File Loading Worker
# =============================================================
def process_one_file(run_name):
    """
    Download one clean preprocessed .set/.fdt from Azure, parse epochs,
    and compute per-condition averages with corrected label extraction:
      1 = Target
      2 = Non-target
    """
    container = get_container_client()
    pid     = os.getpid()
    tmp_set = os.path.join(tmp_dir, f"{run_name}_{pid}.set")
    tmp_fdt = os.path.join(tmp_dir, f"{run_name}_{pid}.fdt")

    try:
        # .set is required
        blob_set = f"{AZURE_INPUT_PREFIX}/{run_name}.set"
        try:
            with open(tmp_set, 'wb') as f:
                container.get_blob_client(blob_set).download_blob().readinto(f)
        except Exception as e:
            return run_name, None, None, f".set not found: {e}"

        # .fdt is optional
        blob_fdt = f"{AZURE_INPUT_PREFIX}/{run_name}.fdt"
        try:
            with open(tmp_fdt, 'wb') as f:
                container.get_blob_client(blob_fdt).download_blob().readinto(f)
        except Exception:
            pass

        epochs = mne.io.read_epochs_eeglab(tmp_set, verbose=False)

        # Label extraction via event_id inversion:
        # Original stimulus codes in Giga:
        #   '1' = target
        #   '2' = non-target
        code_to_label = {v: int(k) for k, v in epochs.event_id.items()}
        labels = [code_to_label[code] for code in epochs.events[:, 2]]

        # CORRECTED TARGET vs NON-TARGET ASSIGNMENT:
        # 1 = Target, 2 = Non-target
        target_idx    = [i for i, l in enumerate(labels) if l == 1]
        nontarget_idx = [i for i, l in enumerate(labels) if l == 2]

        evoked_t  = epochs[target_idx].average()    if target_idx    else None
        evoked_nt = epochs[nontarget_idx].average() if nontarget_idx else None

        return run_name, evoked_t, evoked_nt, None

    except Exception as e:
        return run_name, None, None, str(e)

    finally:
        for path in [tmp_set, tmp_fdt]:
            if os.path.exists(path):
                os.remove(path)
        gc.collect()

# =============================================================
# 2. Build Run List and Dispatch Parallel Jobs
# =============================================================
print("\n" + "="*70)
print(f"STAGE 1: Parallel epoch loading from '{AZURE_INPUT_PREFIX}' (n_jobs={N_JOBS})")
print("="*70)

all_runs = [
    f"clean_epoched_subj{subj}_sess{sess}_EEG_ERP_{split}"
    for sess in sessions_list
    for split in split_list
    for subj in subjects_list
]
print(f"  Explicit target preprocessed runs: {len(all_runs)} (54 subjects x 2 sessions x 2 splits)")

results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
    delayed(process_one_file)(run_name)
    for run_name in all_runs
)

# =============================================================
# 3. Collect Results
# =============================================================
evokeds = {'target': [], 'non-target': []}
skipped = []
loaded  = 0

for run_name, evoked_t, evoked_nt, error in results:
    if error is not None:
        skipped.append((run_name, error))
        continue
    if evoked_t  is not None: evokeds['target'].append(evoked_t)
    if evoked_nt is not None: evokeds['non-target'].append(evoked_nt)
    loaded += 1

print(f"\nLoaded {loaded} files successfully, skipped {len(skipped)}")
if skipped and len(skipped) <= 10:
    print("Skipped:")
    for s, err in skipped:
        print(f"  {s}: {err}")
elif skipped:
    print(f"  (First 5 skipped: {[s[0] for s in skipped[:5]]}...)")

if not evokeds['target'] or not evokeds['non-target']:
    raise RuntimeError("No valid evoked objects accumulated -- cannot compute grand average.")

# =============================================================
# 4. Compute Grand Averages
# =============================================================
print("\n" + "="*70)
print("STAGE 2: Computing grand averages")
print("="*70)

# Intersect channels across all loaded files to ensure strict alignment
common_chans = None
for cond in ['target', 'non-target']:
    for ev in evokeds[cond]:
        if common_chans is None:
            common_chans = set(ev.ch_names)
        else:
            common_chans = common_chans.intersection(set(ev.ch_names))

common_chans = sorted(list(common_chans))
print(f"Common channels across all files ({len(common_chans)}): {', '.join(common_chans[:10])} ...")

for cond in ['target', 'non-target']:
    evokeds[cond] = [ev.copy().pick_channels(common_chans) for ev in evokeds[cond]]

grand = {
    'target':     mne.grand_average(evokeds['target']),
    'non-target': mne.grand_average(evokeds['non-target']),
}

print(f"  • Target:     grand average over {len(evokeds['target'])} evokeds")
print(f"  • Non-target: grand average over {len(evokeds['non-target'])} evokeds")

ch_names = grand['target'].ch_names
times    = grand['target'].times    # seconds
times_ms = times * 1000             # milliseconds

# =============================================================
# 5. Save ERP Data as .npz for Later Reuse
# =============================================================
print("\n" + "="*70)
print("STAGE 3: Saving ERP data (.npz)")
print("="*70)

npz_path = os.path.join(PLOT_DIR, "giga_grand_erp.npz")
np.savez(
    npz_path,
    target             = grand['target'].data,      # (n_ch, n_times) Volts
    non_target         = grand['non-target'].data,
    times              = times,                      # seconds
    ch_names           = np.array(ch_names),
    sfreq              = grand['target'].info['sfreq'],
    n_evoked_target    = len(evokeds['target']),
    n_evoked_nontarget = len(evokeds['non-target']),
)
print(f"  [saved] {npz_path}")
print(f"  Shape: {grand['target'].data.shape} (channels x timepoints)")

# =============================================================
# 6. Standalone Single ROI-Averaged ERP Plot (Target vs Non-Target)
# =============================================================
print("\n" + "="*70)
print("STAGE 4: Generating Single ROI-Averaged ERP Plot")
print("="*70)

# Match ROI channels case-insensitively
ch_names_cf = [c.casefold() for c in ch_names]
roi_cf      = [r.casefold() for r in ROI_CHANNELS]
roi_indices = [i for i, c in enumerate(ch_names_cf) if c in roi_cf]

if not roi_indices:
    roi_indices = [i for i, c in enumerate(ch_names_cf) if any(k in c for k in ['cz', 'pz', 'cp'])]

matched_roi_names = [ch_names[i] for i in roi_indices]
print(f"Matched ROI channels ({len(matched_roi_names)}): {', '.join(matched_roi_names)}")

# Calculate spatial average across ROI channels in microvolts
roi_target_uv    = grand['target'].data[roi_indices, :].mean(axis=0) * 1e6
roi_nontarget_uv = grand['non-target'].data[roi_indices, :].mean(axis=0) * 1e6
roi_diff_uv      = roi_target_uv - roi_nontarget_uv

# Publication-grade single plot
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
ax.plot(times_ms, roi_target_uv,    color='#d62728', lw=2.4, label='Target')
ax.plot(times_ms, roi_nontarget_uv, color='#1f77b4', lw=2.4, label='Non-target')
ax.plot(times_ms, roi_diff_uv,      color='#2ca02c', lw=1.6, linestyle='--', label='Difference (Target − Non-target)')

ax.axhline(0, color='black', lw=0.8, linestyle=':')
ax.axvline(0, color='black', lw=0.8, linestyle=':')
ax.axvspan(250, 500, color='#d62728', alpha=0.10, label='P300 Window (250–500 ms)')

ax.set_title(f"Giga Dataset — Grand-Averaged ERP (ROI Average)\nROI: {', '.join(matched_roi_names)}",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
ax.set_ylabel("Amplitude (µV)", fontsize=11, fontweight='bold')
ax.set_xlim(times_ms[0], times_ms[-1])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=10, loc='upper right', framealpha=0.95)

plt.tight_layout()
roi_plot_path = os.path.join(PLOT_DIR, "giga_grand_erp_roi.png")
plt.savefig(roi_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [saved] ROI Average Plot -> {roi_plot_path}")

# =============================================================
# 7. Per-Channel Waveform Plots
# =============================================================
print("\n" + "="*70)
print("STAGE 5: Per-channel waveform plots")
print("="*70)

p300_mask        = (times >= 0.25) & (times < 0.50)
mean_p300_target = grand['target'].data[:, p300_mask].mean(axis=1) * 1e6

for ch_idx, ch_name in enumerate(ch_names):
    target_uv    = grand['target'].data[ch_idx]     * 1e6
    nontarget_uv = grand['non-target'].data[ch_idx] * 1e6
    diff_uv      = target_uv - nontarget_uv

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times_ms, target_uv,    color='#d62728', lw=1.5, label='Target')
    ax.plot(times_ms, nontarget_uv, color='#1f77b4', lw=1.5, label='Non-target')
    ax.plot(times_ms, diff_uv,      color='#2ca02c', lw=1.0, linestyle='--', label='Difference (T − NT)')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvspan(250, 500, alpha=0.08, color='#d62728', label='P300 window')
    ax.set_title(
        f"Giga Grand ERP — {ch_name}  "
        f"(mean P300 250–500ms: {mean_p300_target[ch_idx]:+.2f} µV)"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_ch = ch_name.replace('/', '_')
    path    = os.path.join(PLOT_DIR, "erp_channels", f"erp_{safe_ch}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

print(f"  [saved] {len(ch_names)} channel PNGs -> {PLOT_DIR}/erp_channels/")

# =============================================================
# 8. Grid Plot of All Channels
# =============================================================
print("\n" + "="*70)
print("STAGE 6: Grid plot (all channels)")
print("="*70)

n_cols    = 8
n_rows    = int(np.ceil(len(ch_names) / n_cols))
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 2.5, n_rows * 2.0),
    sharex=True
)
axes_flat = axes.flatten()

for ch_idx, ch_name in enumerate(ch_names):
    ax           = axes_flat[ch_idx]
    target_uv    = grand['target'].data[ch_idx]     * 1e6
    nontarget_uv = grand['non-target'].data[ch_idx] * 1e6
    ax.plot(times_ms, target_uv,    color='#d62728', lw=0.8)
    ax.plot(times_ms, nontarget_uv, color='#1f77b4', lw=0.8)
    ax.axhline(0, color='k', lw=0.3)
    ax.axvspan(250, 500, alpha=0.06, color='#d62728')
    ax.set_title(ch_name, fontsize=7, pad=2)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.2)

for idx in range(len(ch_names), len(axes_flat)):
    axes_flat[idx].set_visible(False)

handles = [
    plt.Line2D([0], [0], color='#d62728', lw=1.5, label='Target'),
    plt.Line2D([0], [0], color='#1f77b4', lw=1.5, label='Non-target'),
]
fig.legend(handles=handles, loc='lower right', fontsize=9, ncol=2)
fig.suptitle("Giga Dataset — Grand ERP (all channels)", fontsize=13, y=1.01)
plt.tight_layout()
savefig("giga_grand_erp_grid")

# =============================================================
# 9. Empirical Peak Channel Report
# =============================================================
print("\n" + "="*70)
print("STAGE 7: Empirical P300 peak channel report")
print("="*70)

peak_order = np.argsort(mean_p300_target)[::-1]
print("Top 10 channels by mean amplitude in P300 window (250–500ms, target):")
for rank, ch_idx in enumerate(peak_order[:10]):
    print(f"  {rank+1:2d}. {ch_names[ch_idx]:<8s}  {mean_p300_target[ch_idx]:+.3f} µV")

peak_ch     = ch_names[peak_order[0]]
peak_ch_idx = peak_order[0]
peak_lat_ms = times_ms[np.argmax(grand['target'].data[peak_ch_idx])]
print(f"\nPeak channel: {peak_ch} (peak latency: {peak_lat_ms:.1f} ms)")

print("\n" + "="*70)
print("All done successfully.")
print(f"Outputs written to: {PLOT_DIR}")
print(f"  • giga_grand_erp_roi.png     — Standalone single ROI averaged ERP plot")
print(f"  • giga_grand_erp.npz         — ERP data array for downstream tests")
print(f"  • giga_grand_erp_grid.png    — All-channels grid plot")
print(f"  • erp_channels/erp_<ch>.png  — Per-channel waveforms ({len(ch_names)} files)")
print("="*70)
