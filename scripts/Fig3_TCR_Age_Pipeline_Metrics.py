"""
Figure 3: Continuous Pipeline and Sequence Distinctiveness
----------------------------------------------------------
This script generates a Nature-compliant multi-panel figure demonstrating:
(A) The conceptual pipeline: Data extraction, KDE construction, EMD force, 
    scoring, and GMM statistical selection.
(B) ROC curves for Sequence Distinctiveness across multiple classification tasks.
(C) Confusion matrices paired with conceptual mini-schematics (Bifurcation vs. Contrast).
(D) Pairwise Separability heatmap.

Outputs: Figure3_Continuous_Nature_Final_Fixed.pdf (180mm standard width)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.patches import FancyArrowPatch
import os

# ==========================================
# 1. Nature Style & Settings
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 6  # Compliant with Nature's 5-7pt requirement
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded as vectors
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# 2. Helper Functions
# ==========================================

def get_metric(task_name, metric='Accuracy'):
    """
    Reads a specific metric for a given classification task from the results summary.
    Returns 0.5 as a fallback if the file or task is missing.
    """
    summary_file = 'results_summary.csv'
    if not os.path.exists(summary_file): return 0.5
    try:
        summ_df = pd.read_csv(summary_file)
        summ_df['Task'] = summ_df['Task'].str.strip()
        summ_df = summ_df.set_index('Task')
        return summ_df.loc[task_name, metric]
    except: 
        return 0.5

def plot_mini_schem(ax, type='bifurcation'):
    """
    Plots a synthetic schematic representing the theoretical models 
    underlying the confusion matrices (either 'bifurcation' or 'distinctiveness').
    """
    np.random.seed(42)
    # Generate mock tri-modal data
    data = np.concatenate([np.random.normal(0, 1.2, 5000), 
                           np.random.normal(-3.5, 1, 1000), 
                           np.random.normal(3.5, 1, 1000)])
    
    sns.kdeplot(data, color="lightgrey", fill=True, alpha=0.3, linewidth=0, ax=ax, rasterized=True)
    line = sns.kdeplot(data, color="grey", linewidth=0.8, ax=ax).get_lines()[-1]
    x, y = line.get_data()
    thresh_low, thresh_high = -1.96, 1.96
    
    if type == 'bifurcation':
        ax.fill_between(x, 0, y, where=(x <= thresh_low), color='#0072B2', alpha=0.9)
        ax.fill_between(x, 0, y, where=(x >= thresh_high), color='#D55E00', alpha=0.9)
        y_arrow = max(y) * 0.4
        arrow = FancyArrowPatch((-3, y_arrow), (3, y_arrow), arrowstyle='<|-|>', mutation_scale=6, color='black', lw=0.8)
        ax.add_patch(arrow)
        ax.text(0, y_arrow + 0.01, 'VS', ha='center', va='bottom', fontweight='bold', fontsize=5)
    else:
        ax.fill_between(x, 0, y, where=(x <= thresh_low), color='#663399', alpha=0.7)
        ax.fill_between(x, 0, y, where=(x >= thresh_high), color='#663399', alpha=0.7)
        ax.fill_between(x, 0, y, where=((x > thresh_low) & (x < thresh_high)), color='grey', alpha=0.4, hatch='///')
        ax.annotate("", xy=(-3.5, max(y)*0.5), xytext=(-1.5, max(y)*0.5), arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        ax.annotate("", xy=(3.5, max(y)*0.5), xytext=(1.5, max(y)*0.5), arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        
    ax.set_yticks([])
    ax.set_xlim(-5, 5)
    ax.set_xticks([])
    ax.axis('off')

def add_panel_letter(ax, letter, x=-0.08, y=1.05):
    """
    Adds a standardized lowercase bold letter to a subplot for Nature formatting.
    """
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom', ha='right')

# ==========================================
# 3. Layout Configuration (GridSpec)
# ==========================================
# 180mm full width standard (7.08 inches). Adjusted height to 5.6 for perfect aspect ratio.
fig = plt.figure(figsize=(7.08, 5.6), dpi=300)

# Main Grid: Two rows. Top row for the pipeline, bottom row for metrics.
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.45, 
                             left=0.09, right=0.99, top=0.96, bottom=0.08)

# Top row nested grid: 5 panels
gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], 
                                          width_ratios=[1.3, 0.9, 0.9, 0.9, 1.3], wspace=0.15)

# Bottom row nested grid: 3 main panels
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], 
                                             width_ratios=[1, 1.8, 1], wspace=0.3)

# ==========================================
# 4. Top Panel Generation (Pipeline)
# ==========================================

# --- a1. Data Extraction (Visual Matrix) ---
ax1 = fig.add_subplot(gs_top[0])
ax1.set_xlim(0, 6.5); ax1.set_ylim(0, 12); ax1.axis('off')
rows, cols, cw, ch, sx, sy = 8, 5, 1.0, 1.15, 0.2, 0.5 
ages, presence = [22, 65, 24, 70, 28, 68, 25, 72], [1, 0, 1, 1, 0, 0, 1, 0]

for j in range(cols):
    ax1.text(sx + j*cw + cw/2, sy + rows*ch + 0.3, f'TCR{j+1}', ha='center', color='black' if j==2 else 'grey', fontsize=4)
for i in range(rows):
    y_p = sy + (rows-1-i)*ch
    ax1.text(sx - 0.1, y_p + ch/2, f'D{i+1} ({ages[i]})', ha='right', va='center', fontsize=4)
    for j in range(cols):
        fc = '#0072B2' if (j == 2 and presence[i] == 1) else '#f4f4f4'
        ax1.add_patch(patches.Rectangle((sx + j*cw, y_p), cw, ch, lw=0.4, ec='white', fc=fc, alpha=0.8))
        if j == 2 and presence[i] == 1: 
            ax1.text(sx + j*cw + cw/2, y_p + ch/2, '✓', ha='center', va='center', color='white', fontsize=5)

ax1.add_patch(patches.Rectangle((sx + 2*cw, sy), cw, rows*ch, lw=1, ec='#0072B2', fc='none', zorder=2))
ax1.text(sx + 3.8*cw, sy + 4*ch, "Age Vector:\n[22, 24, 70, 25]", ha='center', va='center', fontsize=5, fontweight='bold', zorder=5, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="#0072B2", lw=0.8, alpha=0.95))
ax1.set_title("Data Extraction", loc='center', fontsize=6, pad=8)
add_panel_letter(ax1, 'a1')

# --- a2. Construct Distribution (KDEs) ---
ax2 = fig.add_subplot(gs_top[1])
x_grid = np.linspace(0, 100, 200)
y_young, y_null = norm.pdf(x_grid, 28, 7), norm.pdf(x_grid, 50, 15)
ax2.plot(x_grid, y_null, 'k--', alpha=0.3, lw=0.7)
ax2.plot(x_grid, y_young, color='#0072B2', lw=1)
ax2.fill_between(x_grid, y_young, color='#0072B2', alpha=0.15, rasterized=True)
ax2.set_yticks([]); ax2.set_xlim(0, 100); ax2.set_xlabel("Age", fontsize=5)
ax2.set_title("Construct Dist.", loc='center', fontsize=6, pad=8)
add_panel_letter(ax2, 'a2')
sns.despine(ax=ax2, left=True)

# --- a3. Earth Mover's Force (Transport Schematic) ---
ax3 = fig.add_subplot(gs_top[2])
ax3.plot(x_grid, y_null, 'k--', alpha=0.3, lw=0.7)
ax3.plot(x_grid, y_young, color='#0072B2', lw=1, alpha=0.7)
for i in range(35, 95, 15): 
    ax3.arrow(i, norm.pdf(i, 50, 15), -10, 0, head_width=0.0015, head_length=2.5, fc='#0072B2', ec='#0072B2', alpha=0.6)
ax3.set_title("EMD Force", loc='center', fontsize=6, pad=8)
ax3.set_yticks([]); ax3.set_xlim(0, 100); ax3.set_xlabel("Age", fontsize=5)
add_panel_letter(ax3, 'a3')
sns.despine(ax=ax3, left=True)

# --- a4. Calculate Score (EMD Computation) ---
ax4 = fig.add_subplot(gs_top[3])
ax4.plot(x_grid, y_null, 'k--', alpha=0.3, lw=0.7)
ax4.plot(x_grid, y_young, color='#0072B2', lw=1)
ax4.annotate("", xy=(28, 0.015), xytext=(50, 0.015), arrowprops=dict(arrowstyle="<|-|>", lw=0.7))
ax4.text(39, 0.018, "Distance", ha='center', fontsize=5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8))
ax4.text(50, max(y_young)*0.6, "Score = -2.8", ha='center', va='center', fontsize=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#0072B2", lw=0.5, alpha=0.9))
ax4.set_title("Calculate Score", loc='center', fontsize=6, pad=8)
ax4.set_yticks([]); ax4.set_xlim(0, 100); ax4.set_xlabel("Age", fontsize=5)
add_panel_letter(ax4, 'a4')
sns.despine(ax=ax4, left=True)

# --- a5. GMM Statistical Selection ---
ax5 = fig.add_subplot(gs_top[4])
np.random.seed(42)
selection_data = np.concatenate([np.random.normal(0, 0.8, 153194), np.random.normal(-3, 1.2, 3732), np.random.normal(3, 1.2, 3771)])
sns.kdeplot(selection_data, color="grey", fill=True, alpha=0.15, linewidth=0, ax=ax5, rasterized=True)
line_kde = sns.kdeplot(selection_data, color="grey", linewidth=0.8, ax=ax5).get_lines()[-1]
x_kde, y_kde = line_kde.get_data()

ax5.fill_between(x_kde, 0, y_kde, where=(x_kde <= -1.96), color='#0072B2', alpha=0.85)
ax5.fill_between(x_kde, 0, y_kde, where=(x_kde >= 1.96), color='#D55E00', alpha=0.85)
ax5.axvline(-1.96, color='k', ls='--', lw=0.5, alpha=0.4)
ax5.axvline(1.96, color='k', ls='--', lw=0.5, alpha=0.4)

ax5.annotate('n = 3,732', xy=(-3.5, max(y_kde)*0.1), xytext=(-5.2, max(y_kde)*0.35), arrowprops=dict(arrowstyle='->', color='#0072B2', lw=0.6), color='#0072B2', ha='center', fontsize=5, fontweight='bold')
ax5.annotate('n = 3,771', xy=(3.5, max(y_kde)*0.1), xytext=(5.2, max(y_kde)*0.35), arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.6), color='#D55E00', ha='center', fontsize=5, fontweight='bold')
ax5.annotate('Non-associated\nn = 153,194', xy=(0, max(y_kde)*0.6), xytext=(0, max(y_kde)*0.85), arrowprops=dict(arrowstyle='->', color='black', lw=0.5, alpha=0.8), ha='center', va='center', color='black', fontsize=5, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))

ax5.set_title("GMM Selection", loc='center', fontsize=6, pad=8)
ax5.set_xlabel("Signed Wasserstein Score", fontsize=5.5)
ax5.set_xlim(-7, 7); ax5.set_yticks([])
add_panel_letter(ax5, 'a5')
sns.despine(ax=ax5, left=True, right=True)

# ==========================================
# 5. Bottom Panel Generation (Metrics)
# ==========================================

# --- b. Sequence Distinctiveness (ROC Curves) ---
ax_d = fig.add_subplot(gs_bottom[0])
roc_tasks = [('Young_vs_Old', 'preds_young_vs_old.csv', '#E69F00', '-', 'Old'), 
             ('Age_vs_Non_Balanced', 'preds_age_vs_non_balanced.csv', '#CC79A7', '-', 'Age'), 
             ('Non1_vs_Non2', 'preds_non1_vs_non2.csv', 'grey', '--', 'Non2')]

for task, fname, color, style, pos_label in roc_tasks:
    if os.path.exists(fname):
        df_roc = pd.read_csv(fname)
        potential_cols = [c for c in df_roc.columns if c not in ['TCR', 'target', 'predict', 'Unnamed: 0']]
        score_col = next((c for c in potential_cols if c.lower() in [f"p{pos_label.lower()}", pos_label.lower()]), potential_cols[-1])
        y_true = (df_roc['target'] == pos_label).astype(int)
        fpr, tpr, _ = roc_curve(y_true, df_roc[score_col])
        auc_val = auc(fpr, tpr)
        label_text = task.replace('_', ' ').replace('Balanced', '').replace('Non1 vs Non2', 'Non vs Non').strip()
        ax_d.plot(fpr, tpr, color=color, linestyle=style, lw=1.2, label=f"{label_text} ({auc_val:.2f})")

ax_d.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.4)
ax_d.set_xlabel('False Positive Rate', fontsize=5)
ax_d.set_ylabel('True Positive Rate', fontsize=5)
ax_d.set_title('Sequence Distinctiveness', loc='center', fontsize=7)
add_panel_letter(ax_d, 'b', y=1.05)
ax_d.legend(frameon=False, fontsize=5, loc='lower right')
sns.despine(ax=ax_d)

# --- c. Confusion Matrices (Nested GridSpec) ---
gs_e = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_bottom[1], width_ratios=[1, 0.85], hspace=0.5, wspace=0.7)

# c1. Bifurcation Model Schematic
ax_e1 = fig.add_subplot(gs_e[0, 0])
plot_mini_schem(ax_e1, 'bifurcation')
ax_e1.set_title('Bifurcation Model', loc='center', fontsize=6)
add_panel_letter(ax_e1, 'c1', x=-0.1)

# c2. Young vs Old Heatmap
ax_e2 = fig.add_subplot(gs_e[0, 1])
fname = 'preds_young_vs_old.csv'
if os.path.exists(fname):
    df_cm = pd.read_csv(fname)
    labels = ['Young', 'Old']
    cm = confusion_matrix(df_cm['target'], df_cm['predict'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_e2, annot_kws={"size": 6}) 
    ax_e2.set_title('Young vs Old', loc='center', fontsize=6)
    ax_e2.set_xticklabels(labels, fontsize=5)
    ax_e2.set_yticklabels(labels, rotation=0, fontsize=5)
    ax_e2.set_ylabel('True', fontsize=5)
add_panel_letter(ax_e2, 'c2', x=-0.2)

# c3. Distinctiveness Model Schematic
ax_e3 = fig.add_subplot(gs_e[1, 0])
plot_mini_schem(ax_e3, 'distinctiveness')
ax_e3.set_title('Signal Contrast', loc='center', fontsize=6)
add_panel_letter(ax_e3, 'c3', x=-0.1)

# c4. Age vs Non Heatmap
ax_e4 = fig.add_subplot(gs_e[1, 1])
fname = 'preds_age_vs_non_balanced.csv'
if os.path.exists(fname):
    df_cm2 = pd.read_csv(fname)
    labels = ['Age', 'Non']
    cm2 = confusion_matrix(df_cm2['target'], df_cm2['predict'], labels=labels)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax_e4, annot_kws={"size": 6})
    ax_e4.set_title('Age vs Non', loc='center', fontsize=6)
    ax_e4.set_xticklabels(['Age', 'Non'], fontsize=5)
    ax_e4.set_yticklabels(['Age', 'Non'], rotation=0, fontsize=5)
    ax_e4.set_ylabel('True', fontsize=5)
    ax_e4.set_xlabel('Predicted', fontsize=5)
add_panel_letter(ax_e4, 'c4', x=-0.2)

# --- d. Pairwise Separability (Heatmap) ---
ax_f = fig.add_subplot(gs_bottom[2])
acc_yo = get_metric('Young_vs_Old')
acc_yn = (get_metric('Young_vs_Non1') + get_metric('Young_vs_Non2')) / 2
acc_on = (get_metric('Old_vs_Non1') + get_metric('Old_vs_Non2')) / 2

matrix_data = np.array([
    [1.0, acc_yo, acc_yn], 
    [acc_yo, 1.0, acc_on], 
    [acc_yn, acc_on, 1.0]
])

sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='viridis', cbar=False, ax=ax_f, 
            annot_kws={"size": 6}, linewidths=0.5, linecolor='white')
labels_f = ['Young', 'Old', 'Non']
ax_f.set_xticklabels(labels_f, rotation=45, ha='right', fontsize=6)
ax_f.set_yticklabels(labels_f, rotation=0, fontsize=6)
ax_f.set_title('Pairwise Separability', loc='center', fontsize=7)
add_panel_letter(ax_f, 'd', y=1.05)

# ==========================================
# 6. Save and Export
# ==========================================
# Saving cleanly without tight_layout to preserve exact specified dimensions.
plt.savefig('Figure3_Continuous_Nature_Final_Fixed.pdf', format='pdf', dpi=300, transparent=True)
plt.show()