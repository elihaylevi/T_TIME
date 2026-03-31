"""
Figure 4: Functional Landscape and Age-Bias Spectrum
----------------------------------------------------
This script generates a Nature-compliant multi-panel figure demonstrating:
(A) Functional Landscape (MDS) of species based on age-bias Wasserstein distances.
(B) Species Age-Bias Spectrum showing the distribution (quartiles/medians) per category.
(C) Antigenic Resolution (ECDF) for top target species and their respective genes.
(D) Cumulative Shift Integral isolating specific gene influences.
(E) Intra-Species Gene Shifts demonstrating variance within specific pathogens.

Outputs: Figure4_Landscape_Final_Nature_V2.pdf (180mm standard width)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.manifold import MDS
from matplotlib.patches import Patch
import os

# ==========================================
# 1. Nature Style & Settings
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 7  
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def add_panel_letter(ax, letter, x=-0.05, y=1.10):
    """
    Adds a standardized lowercase bold letter to a subplot for Nature formatting.
    """
    ax.text(x, y, letter, transform=ax.transAxes, 
            fontsize=7, fontweight='bold', va='bottom', ha='right')

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
file_path = 'adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv'
df = pd.read_csv(file_path)

# Filter out Human entries to focus on pathogens/self-antigens
df = df[df['Epitope_species_norm'] != "Human (Homo sapiens)"].copy()

# Standardize nomenclature for plotting
name_map = {
    'Cytomegalovirus (CMV)': 'CMV',
    'Influenza': 'Influenza',
    'M. tuberculosis': 'TB',
    'EBV': 'EBV',
    'Cancer/Self antigen': 'Cancer/Self',
    'COVID-19 / SARS-CoV-2 / SARS-CoV': 'SARS-CoV-2',
    'HIV / Human immunodeficiency virus (HIV)': 'HIV',
    'T1D': 'T1D',
    'Neoantigen': 'Neoantigen'
}

category_map = {
    'CMV': 'Viral', 'Influenza': 'Viral', 'TB': 'Bacterial',
    'EBV': 'Viral', 'Cancer/Self': 'Self/Cancer', 'SARS-CoV-2': 'Viral',
    'HIV': 'Viral', 'T1D': 'Self/Cancer', 'Neoantigen': 'Self/Cancer'
}

CAT_COLORS = {'Viral': '#4878D0', 'Bacterial': '#6ACC64', 'Self/Cancer': '#D65F5F'}

df['species'] = df['Epitope_species_norm'].replace(name_map)
df['gene'] = df['Epitope_gene_merged'].fillna('Unknown')
metric_col = 'signed_wasserstein'

# Identify top 9 species and target species for in-depth analysis
top_9_list = df['species'].value_counts().head(9).index.tolist()
target_species = ['EBV', 'SARS-CoV-2', 'Influenza', 'CMV']
gene_limits = {'EBV': 3, 'SARS-CoV-2': 3, 'Influenza': 2, 'CMV': 2}
SPECIES_COLORS_E = dict(zip(target_species, sns.color_palette("husl", len(target_species))))

# Filter to top genes within the target species
filtered_rows = []
for sp in target_species:
    sp_df = df[df['species'] == sp]
    top_genes = sp_df['gene'].value_counts().head(gene_limits[sp]).index.tolist()
    filtered_rows.append(sp_df[sp_df['gene'].isin(top_genes)])
df_genes = pd.concat(filtered_rows)

# ==========================================
# 3. Auxiliary Computations (MDS & Integrals)
# ==========================================

# 3.1 MDS computation based on Wasserstein distances
distributions = {s: df[df['species'] == s][metric_col].values for s in top_9_list}
dist_matrix = np.zeros((9, 9))
for i in range(9):
    for j in range(i+1, 9):
        # Symmetrical distance matrix calculation
        dist_matrix[i, j] = dist_matrix[j, i] = stats.wasserstein_distance(
            distributions[top_9_list[i]], distributions[top_9_list[j]]
        )
coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)

# 3.2 Integral computations for ECDF shifts
x_grid = np.linspace(-6, 6, 200)
integral_vals = []
for sp in target_species:
    sp_vals = np.sort(df[df['species'] == sp][metric_col].values)
    sp_ecdf = np.searchsorted(sp_vals, x_grid) / len(sp_vals)
    for g in df_genes[df_genes['species'] == sp]['gene'].unique():
        g_vals = np.sort(df_genes[(df_genes['species'] == sp) & (df_genes['gene'] == g)][metric_col].values)
        integral_vals.append(np.cumsum(sp_ecdf - np.searchsorted(g_vals, x_grid) / len(g_vals)))

y_min_int, y_max_int = min(map(min, integral_vals))*1.1, max(map(max, integral_vals))*1.1

# ==========================================
# 4. Figure Construction (180mm width)
# ==========================================
fig = plt.figure(figsize=(7.08, 6.5), dpi=300)

gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1.3, 1.0], height_ratios=[1, 1.2], 
                       hspace=0.5, wspace=0.20, left=0.03, right=0.98, top=0.92, bottom=0.08)

# ---------------------------------------------------------
# Panel A: Functional Landscape (MDS)
# ---------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(coords[:, 0], coords[:, 1], 
             s=[len(distributions[s])/12.0 for s in top_9_list], 
             c=[CAT_COLORS[category_map[s]] for s in top_9_list], 
             alpha=0.9, edgecolors='black', lw=0.7)

x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
x_range, y_range = x_max - x_min, y_max - y_min

for i, txt in enumerate(top_9_list): 
    ax_a.text(coords[i, 0], coords[i, 1] + 0.08 * y_range, txt, 
              fontsize=5, fontweight='bold', ha='center', va='bottom')

ax_a.set_xlim(x_min - 0.15 * x_range, x_max + 0.15 * x_range)
ax_a.set_ylim(y_min - 0.15 * y_range, y_max + 0.20 * y_range)
ax_a.set_xticks([]); ax_a.set_yticks([])

for spine in ax_a.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color('black')

ax_a.set_title("Functional Landscape (MDS)", loc='center', pad=10)
add_panel_letter(ax_a, 'a', x=-0.02)
ax_a.legend(handles=[Patch(facecolor=CAT_COLORS[cat], alpha=0.6, label=cat) for cat in CAT_COLORS], 
            loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=5)

# ---------------------------------------------------------
# Panel B: Species Age-Bias Spectrum (Strip Plot)
# ---------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])

# Calculate descriptive statistics for the strip plot
stats_df = pd.DataFrame([{
    's': s, 'm': np.median(distributions[s]), 
    'q1': np.percentile(distributions[s], 25), 
    'q3': np.percentile(distributions[s], 75), 
    'p5': np.percentile(distributions[s], 5), 
    'p95': np.percentile(distributions[s], 95), 
    'n': len(distributions[s]), 
    'cat': category_map[s]
} for s in top_9_list]).sort_values('m')

y_pos = np.arange(len(stats_df))

ax_b.hlines(y_pos, stats_df['p5'], stats_df['p95'], color='grey', alpha=0.3, lw=0.8)
ax_b.hlines(y_pos, stats_df['q1'], stats_df['q3'], color='#2C3E50', alpha=0.7, lw=2.5)
ax_b.scatter(stats_df['m'], y_pos, s=stats_df['n']/15.0, 
             c=[CAT_COLORS[c] for c in stats_df['cat']], 
             edgecolor='black', lw=0.5, zorder=5)

ax_b.set_yticks(y_pos); ax_b.set_yticklabels([])
ax_b.tick_params(axis='y', length=0); ax_b.set_xlim(-5, 3)

for y, label in zip(y_pos, stats_df['s']):
    ax_b.text(-5, y, label, ha='center', va='center', fontsize=5.5, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9), zorder=20)

ax_b.axvline(0, color='black', ls='--', alpha=0.5, lw=0.5)
ax_b.set_title('Species Age-Bias Spectrum', loc='center', pad=10)
add_panel_letter(ax_b, 'b', x=-0.02)
ax_b.set_xlabel("Signed-Wasserstein Score", fontsize=6)
sns.despine(ax=ax_b)

# ---------------------------------------------------------
# Panel C: Antigenic Resolution (ECDF)
# ---------------------------------------------------------
ax_c_main = fig.add_subplot(gs[1, 0])
ax_c_main.axis('off')
ax_c_main.set_title("Antigenic Resolution (ECDF)", loc='center', pad=20)
add_panel_letter(ax_c_main, 'c', x=-0.05, y=1.15)

gs_c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0], hspace=0.6, wspace=0.1)
for i, sp in enumerate(target_species):
    ax = fig.add_subplot(gs_c[divmod(i, 2)])
    sp_df = df_genes[df_genes['species'] == sp]
    for g in sp_df.groupby('gene')[metric_col].median().sort_values().index:
        v = np.sort(sp_df[sp_df['gene'] == g][metric_col].values)
        ax.plot(v, np.linspace(0, 1, len(v)), lw=1.2, label=g)
    
    ax.set_xlim(-5, 5)
    if i % 2 != 0: ax.tick_params(labelleft=False)
    ax.set_title(sp, fontweight='bold', fontsize=6, pad=2)
    ax.tick_params(axis='both', labelsize=5, pad=1)
    if i >= 2: ax.set_xlabel("Age Score", fontsize=6, labelpad=2)
    
    ax.legend(fontsize=4.5, frameon=False, loc='upper left', handlelength=1.0)
    sns.despine(ax=ax)

# ---------------------------------------------------------
# Panel D: Cumulative Shift Integral
# ---------------------------------------------------------
ax_d_main = fig.add_subplot(gs[1, 1])
ax_d_main.axis('off')
ax_d_main.set_title("Cumulative Shift Integral", loc='center', pad=20)
add_panel_letter(ax_d_main, 'd', x=-0.02, y=1.15)

gs_d = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1], hspace=0.6, wspace=0.1)
for i, sp in enumerate(target_species):
    ax = fig.add_subplot(gs_d[divmod(i, 2)])
    sp_vals = np.sort(df[df['species'] == sp][metric_col].values)
    sp_ecdf = np.searchsorted(sp_vals, x_grid) / len(sp_vals)
    for g in df_genes[df_genes['species'] == sp]['gene'].unique():
        g_vals = np.sort(df_genes[(df_genes['species'] == sp) & (df_genes['gene'] == g)][metric_col].values)
        ax.plot(x_grid, np.cumsum(sp_ecdf - np.searchsorted(g_vals, x_grid) / len(g_vals)), label=g, lw=1.2)
    
    ax.set_xlim(-6, 6)
    if i % 2 != 0: ax.tick_params(labelleft=False)
    ax.axhline(0, color='black', ls=':', alpha=0.5, lw=0.5)
    ax.set_ylim(y_min_int, y_max_int)
    ax.set_title(sp, fontweight='bold', fontsize=6, pad=2)
    ax.tick_params(axis='both', labelsize=5, pad=1)
    if i >= 2: ax.set_xlabel("Age Score", fontsize=6, labelpad=2)
    sns.despine(ax=ax)

# ---------------------------------------------------------
# Panel E: Intra-Species Gene Shifts
# ---------------------------------------------------------
ax_e = fig.add_subplot(gs[:, 2])
gene_stats = df_genes.groupby(['species', 'gene'])[metric_col].agg(['median', 'count']).reset_index()

for i, sp in enumerate(reversed(target_species)):
    sp_stats = gene_stats[gene_stats['species'] == sp].sort_values('median')
    ax_e.plot([sp_stats['median'].min(), sp_stats['median'].max()], [i, i], color='#BBBBBB', alpha=0.3, lw=1.0)

    for j, (_, row) in enumerate(sp_stats.iterrows()):
        current_y = i + np.linspace(-0.25, 0.25, len(sp_stats))[j]
        marker_size = row['count'] * 0.4 
        
        ax_e.scatter(row['median'], current_y, s=marker_size, color=SPECIES_COLORS_E[sp], 
                     edgecolor='black', lw=0.5, zorder=10, alpha=0.9)
        
        text_offset_y = 0.05 + (np.sqrt(marker_size) * 0.004)
        
        ax_e.text(row['median'], current_y + text_offset_y, row['gene'],
                  fontsize=5, ha='center', va='bottom', fontweight='bold', zorder=11)

ax_e.set_yticks(np.arange(len(target_species))); ax_e.set_yticklabels([])
ax_e.tick_params(axis='y', length=0); ax_e.set_xlim(-1.0, 1.0); ax_e.set_ylim(-0.8, 3.8)

for y, label in zip(np.arange(len(target_species)), reversed(target_species)):
    ax_e.text(-1.0, y, label, ha='center', va='center', fontsize=5.5, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9), zorder=20)

ax_e.axvline(0, color='black', ls='--', alpha=0.4, lw=0.8)
ax_e.set_title('Intra-Species Gene Shifts', loc='center', pad=10)
add_panel_letter(ax_e, 'e', x=-0.05, y=1.05)
ax_e.set_xlabel("Median Age Bias Score", fontsize=6)
sns.despine(ax=ax_e)

# ---------------------------------------------------------
# Final Formatting and Export
# ---------------------------------------------------------
for ax in fig.axes:
    # Add minimal padding to tick labels except for completely invisible axes
    if not ax.axis() == (0.0, 1.0, 0.0, 1.0) and ax != ax_a:
        ax.tick_params(pad=2)

plt.savefig('Figure4_Landscape_Final_Nature_V2.pdf', dpi=300, transparent=True)
plt.show()