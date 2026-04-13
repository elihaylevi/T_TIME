import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.gridspec as gridspec

# ==========================================
# 1. Nature Style Settings (Vector & 180mm)
# ==========================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['lines.linewidth'] = 1.0

# Colors from previous figures for consistency
colors = {'Male': '#D55E00', 'Female': '#0072B2'}

def add_panel_letter(ax, letter, x=-0.1, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, 
            fontsize=8, fontweight='bold', va='bottom', ha='right')

# ==========================================
# 2. Data Loading & Robust Cleaning
# ==========================================
# (Note: Using the logic from your provided snippet)
df = pd.read_csv('test_preds_covid.csv', low_memory=False)

sex_col = next((c for c in df.columns if c.lower() in ['biological sex', 'sex', 'biological_sex']), 'Sex')
df.rename(columns={sex_col: 'Sex', 'Age': 'y_true'}, inplace=True)
df['Sex'] = df['Sex'].astype(str).str.strip().str.capitalize().replace({'1': 'Male', '1.0': 'Male', '0': 'Female', '0.0': 'Female'})
df = df[df['Sex'].isin(['Male', 'Female'])]

df['residual'] = df['y_pred'] - df['y_true']
df['abs_residual'] = df['residual'].abs()

# 90th Percentile Filter (Robust Analysis)
cut90 = df['abs_residual'].quantile(0.90)
df_clean = df[df['abs_residual'] <= cut90].copy()

# ==========================================
# 3. Statistical Computations
# ==========================================
females_abs = df_clean[df_clean['Sex'] == 'Female']['abs_residual']
males_abs = df_clean[df_clean['Sex'] == 'Male']['abs_residual']

stat_ks, p_ks = stats.ks_2samp(females_abs, males_abs)
stat_levene, p_levene = stats.levene(females_abs, males_abs)

# Interaction Model (Age x Sex)
model = smf.ols('residual ~ y_true * C(Sex)', data=df_clean).fit()
interaction_pval = model.pvalues.get('y_true:C(Sex)[T.Male]', np.nan)

# ==========================================
# 4. Figure Construction
# ==========================================
# Width: 180mm (7.08 inches), Height: 3 inches (compact for supplement)
fig = plt.figure(figsize=(7.08, 3), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.18)

# --- Panel A: ECDF of Absolute Residuals ---
ax_a = fig.add_subplot(gs[0])
sns.ecdfplot(data=df_clean, x='abs_residual', hue='Sex', palette=colors, ax=ax_a, lw=1.2)
add_panel_letter(ax_a, 'a')

ax_a.set_title('Error Distribution Symmetry', loc='center', pad=10, fontsize=7, fontweight='bold')
ax_a.set_xlabel('Absolute Prediction Error (Years)')
ax_a.set_ylabel('Cumulative Probability')

# Annotation box for stats
stats_text = f"KS test $P$ = {p_ks:.2f}\nLevene $P$ = {p_levene:.2f}"
ax_a.text(0.95, 0.05, stats_text, transform=ax_a.transAxes, fontsize=6, 
          ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax_a.legend(frameon=False, fontsize=6, loc='center right')

# --- Panel B: Bias Stability across Decades ---
ax_b = fig.add_subplot(gs[0, 1])

bins = np.arange(20, 90, 10) # 20s to 80s
df_clean['Age_Decade'] = pd.cut(df_clean['y_true'], bins=bins, right=False)
decades = sorted(df_clean['Age_Decade'].dropna().unique())

diffs, cis, labels = [], [], []

for d in decades:
    sub = df_clean[df_clean['Age_Decade'] == d]
    f_sub = sub[sub['Sex'] == 'Female']['residual']
    m_sub = sub[sub['Sex'] == 'Male']['residual']
    
    if len(f_sub) >= 5 and len(m_sub) >= 5:
        diff = m_sub.mean() - f_sub.mean()
        se = np.sqrt(f_sub.var()/len(f_sub) + m_sub.var()/len(m_sub))
        diffs.append(diff)
        cis.append(1.96 * se)
        labels.append(f"{int(d.left)}s")

x_pos = np.arange(len(labels))
ax_b.errorbar(x_pos, diffs, yerr=cis, fmt='o', color='black', markersize=3, 
              capsize=2, elinewidth=0.8, markeredgewidth=0.8)

ax_b.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# Global mean bias line
mean_m = df_clean[df_clean['Sex'] == 'Male']['residual'].mean()
mean_f = df_clean[df_clean['Sex'] == 'Female']['residual'].mean()
ax_b.axhline(mean_m - mean_f, color='#27AE60', linestyle=':', linewidth=0.8, alpha=0.8, label='Global Mean Bias')

add_panel_letter(ax_b, 'b')
ax_b.set_title('Bias Stability Across Lifespan', loc='center', pad=10, fontsize=7, fontweight='bold')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(labels)
ax_b.set_xlabel('Age Decade')
ax_b.set_ylabel('Bias (Male - Female Residuals) [yr]')

# Interaction stat text
interaction_text = f"Age $\\times$ Sex Interaction\n$P$ = {interaction_pval:.2f}"
ax_b.text(0.05, 0.95, interaction_text, transform=ax_b.transAxes, fontsize=6, 
          ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

sns.despine(fig)

# ==========================================
# 5. Save Final Figure
# ==========================================
plt.savefig('Supplementary_Fig1_Fairness_Stability.pdf', format='pdf', transparent=True)
plt.show()

print("✅ Supplementary Figure S1 generated as vector PDF.")