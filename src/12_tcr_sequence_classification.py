"""
Module: 12_tcr_sequence_classification

Description:
Performs sequence-level classification of TCRs based on their immunological age associations.
1. Encodes TCR amino acid sequences into physical-chemical properties using Atchley Factors 
   with center-padding to handle variable lengths.
2. Creates strictly balanced (1:1) cohorts of Young-associated, Old-associated, and 
   Non-significant (background) TCR sequences.
3. Uses H2O AutoML to train models to distinguish between these biological groups.
4. Validates the biological signal by testing against negative controls (Non vs. Non).
5. Generates a Nature-style figure containing ROC curves and a pairwise separability heatmap.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import h2o
from h2o.automl import H2OAutoML

# ==========================================
# 1. STYLE & CONSTANTS
# ==========================================
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5
})

DATA_FILE = "./updated_tcr_age_lists_with_all_significance.csv"
SUMMARY_FILE = "sequence_classification_summary.csv"
SEED = 42

# Atchley Factors for Amino Acids
ATCHLEY_DICT = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
    'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512]
}

# ==========================================
# 2. ENCODING LOGIC
# ==========================================
def encode_atchley_center_padded(sequences, max_len=None):
    """Encodes TCR sequences into fixed-size numeric matrices using Atchley factors and center padding."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    
    n_factors = 5
    encoded_matrix = []
    
    for seq in sequences:
        mat = np.zeros((max_len, n_factors), dtype=np.float32)
        seq_len = min(len(seq), max_len)
        seq = seq[:max_len]
        mid_point = seq_len // 2
        
        # Encode Head
        for i in range(mid_point):
            if seq[i] in ATCHLEY_DICT: mat[i, :] = ATCHLEY_DICT[seq[i]]
        
        # Encode Tail (Pushed to the end)
        remaining = seq_len - mid_point
        for i in range(remaining):
            aa = seq[seq_len - 1 - i]
            target_pos = max_len - 1 - i
            if aa in ATCHLEY_DICT: mat[target_pos, :] = ATCHLEY_DICT[aa]
            
        encoded_matrix.append(mat.flatten())
        
    col_names = [f"Pos{i}_F{f}" for i in range(max_len) for f in range(1, 6)]
    return np.array(encoded_matrix), col_names

def save_metrics(metrics_dict):
    df_new = pd.DataFrame([metrics_dict])
    if not os.path.isfile(SUMMARY_FILE):
        df_new.to_csv(SUMMARY_FILE, index=False)
    else:
        df_new.to_csv(SUMMARY_FILE, mode='a', header=False, index=False)

# ==========================================
# 3. DATA PREPARATION & BALANCING
# ==========================================
def prepare_balanced_cohorts():
    print("Loading and filtering TCR data...")
    df = pd.read_csv(DATA_FILE)
    df = df[df['TCR'].apply(lambda x: isinstance(x, str) and all(c in ATCHLEY_DICT for c in x))].copy()

    z_col = "component_zscore"
    age_threshold = 1.96

    # Identify pools
    idx_young = df[(df['signed_wasserstein_significant']==True) & (df[z_col] < -age_threshold)].index
    idx_old   = df[(df['signed_wasserstein_significant']==True) & (df[z_col] > age_threshold)].index
    idx_non   = df[df['signed_wasserstein_significant']==False].index

    # Enforce strict maximum balanced size across all cohorts
    sample_size = min(len(idx_young), len(idx_old), 5000)
    np.random.seed(SEED)

    young_grp = df.loc[np.random.choice(idx_young, sample_size, replace=False)]
    old_grp   = df.loc[np.random.choice(idx_old, sample_size, replace=False)]
    
    # We need multiple distinct 'Non' groups for various tests
    non_indices = np.random.choice(idx_non, sample_size * 4, replace=False)
    non1_grp = df.loc[non_indices[:sample_size]]
    non2_grp = df.loc[non_indices[sample_size:sample_size*2]]
    
    # For the combined Age vs Non experiment
    age_grp = pd.concat([young_grp, old_grp])
    non_combined_grp = df.loc[non_indices[sample_size*2 : sample_size*4]] # Exactly 2x sample_size

    comparisons = [
        ("Young_vs_Old",  young_grp, old_grp,  "Young", "Old"),
        ("Non1_vs_Non2",  non1_grp,  non2_grp, "Non1",  "Non2"),
        ("Young_vs_Non1", young_grp, non1_grp, "Young", "Non1"),
        ("Young_vs_Non2", young_grp, non2_grp, "Young", "Non2"),
        ("Old_vs_Non1",   old_grp,   non1_grp, "Old",   "Non1"),
        ("Old_vs_Non2",   old_grp,   non2_grp, "Old",   "Non2"),
        ("Age_vs_Non",    age_grp,   non_combined_grp, "Age", "Non") # Balanced 1:1 (2*size vs 2*size)
    ]
    
    print(f"Cohorts stabilized. Base size per class: {sample_size}")
    return comparisons

# ==========================================
# 4. H2O AUTOML EXECUTION
# ==========================================
def run_automl_pipeline(comparisons):
    h2o.init(max_mem_size="12G", nthreads=-1)

    for task_name, g1, g2, l1, l2 in comparisons:
        out_file = f"preds_{task_name.lower()}.csv"
        print(f"\n--- Task: {task_name} ({l1} vs {l2}) ---")
        
        # Resume mechanism
        if os.path.exists(out_file) and os.path.exists(SUMMARY_FILE):
            existing = pd.read_csv(SUMMARY_FILE)
            if task_name in existing['Task'].values:
                print(f"Skipping {task_name} (Already completed).")
                continue

        # Combine and encode
        full_df = pd.concat([
            pd.DataFrame({'TCR': g1['TCR'], 'target': l1}),
            pd.DataFrame({'TCR': g2['TCR'], 'target': l2})
        ], ignore_index=True)
        
        X, cols = encode_atchley_center_padded(full_df['TCR'])
        
        # Build H2O Frame
        hf = h2o.H2OFrame(pd.concat([full_df, pd.DataFrame(X, columns=cols)], axis=1))
        hf['target'] = hf['target'].asfactor()
        train, test = hf.split_frame(ratios=[0.75], seed=SEED)
        
        # Train
        print(f"Running AutoML for {task_name}...")
        aml = H2OAutoML(max_models=10, max_runtime_secs=300, seed=SEED, balance_classes=True, sort_metric="AUC")
        aml.train(x=cols, y='target', training_frame=train, leaderboard_frame=test)
        
        # Predict & Save
        preds = aml.leader.predict(test).as_data_frame()
        result_df = pd.concat([test.as_data_frame()[['TCR', 'target']], preds], axis=1)
        result_df.to_csv(out_file, index=False)
        
        # Calculate Metrics
        pos_class = l2 if l2 in result_df.columns else result_df.columns[-1]
        y_true = (result_df['target'] == l2).astype(int)
        y_score = result_df[pos_class]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        acc = np.mean(result_df['target'] == result_df['predict'])
        
        save_metrics({
            "Task": task_name, "Class1": l1, "Class2": l2, 
            "AUC": roc_auc, "Accuracy": acc, 
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"Completed {task_name}. AUC: {roc_auc:.3f} | Acc: {acc:.3f}")

# ==========================================
# 5. RESULTS VISUALIZATION
# ==========================================
def plot_results():
    print("\nGenerating Figure...")
    if not os.path.exists(SUMMARY_FILE):
        print("No summary file found. Run pipeline first.")
        return

    summary_df = pd.read_csv(SUMMARY_FILE).set_index("Task")
    
    fig = plt.figure(figsize=(10, 4), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # Panel A: ROC Curves
    ax1 = fig.add_subplot(gs[0])
    plot_tasks = {'Young_vs_Old': '#E69F00', 'Age_vs_Non': '#CC79A7', 'Non1_vs_Non2': 'grey'}
    
    for task, color in plot_tasks.items():
        fname = f"preds_{task.lower()}.csv"
        if os.path.exists(fname):
            res = pd.read_csv(fname)
            lbl = summary_df.loc[task, "Class2"]
            pos_col = lbl if lbl in res.columns else res.columns[-1]
            
            y_t = (res['target'] == lbl).astype(int)
            fpr, tpr, _ = roc_curve(y_t, res[pos_col])
            auc_val = summary_df.loc[task, "AUC"]
            
            ax1.plot(fpr, tpr, color=color, lw=2, label=f"{task.replace('_',' ')} (AUC = {auc_val:.2f})")

    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.legend(loc="lower right", fontsize=8, frameon=True)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title('a  Sequence Classification ROC', loc='left')

    # Panel B: Pairwise Separability Heatmap
    ax2 = fig.add_subplot(gs[1])
    
    def get_acc(t): return summary_df.loc[t, "Accuracy"] if t in summary_df.index else 0.50
    
    # Aggregate mirrored tasks
    acc_yo = get_acc("Young_vs_Old")
    acc_yn = (get_acc("Young_vs_Non1") + get_acc("Young_vs_Non2")) / 2.0
    acc_on = (get_acc("Old_vs_Non1") + get_acc("Old_vs_Non2")) / 2.0
    acc_nn = get_acc("Non1_vs_Non2")
    
    # 4x4 Matrix: Young, Old, Non1, Non2
    matrix = np.array([
        [1.00,   acc_yo, acc_yn, acc_yn],
        [acc_yo, 1.00,   acc_on, acc_on],
        [acc_yn, acc_on, 1.00,   acc_nn],
        [acc_yn, acc_on, acc_nn, 1.00]
    ])
    
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis', 
                xticklabels=['Young', 'Old', 'Non1', 'Non2'],
                yticklabels=['Young', 'Old', 'Non1', 'Non2'],
                cbar_kws={'label': 'Model Accuracy'}, ax=ax2)
    ax2.set_title('b  Pairwise Accuracy Heatmap', loc='left')

    plt.tight_layout()
    plt.savefig("Figure3_Sequence_Classification.pdf", bbox_inches='tight', transparent=True)
    print("Saved Figure3_Sequence_Classification.pdf")

# ==========================================
# 6. RUN
# ==========================================
if __name__ == "__main__":
    comparisons = prepare_balanced_cohorts()
    run_automl_pipeline(comparisons)
    plot_results()