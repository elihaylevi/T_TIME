"""
Figure 2: Age Prediction Performance, Sex-Specific Bias, and Clinical Validation
------------------------------------------------------------------------------
This script generates a Nature-compliant multi-panel figure demonstrating:
(A-B) Model performance (Chronological vs. Predicted Age) on Primary and External Validation cohorts.
(C) Systematic offset (Raincloud plot) between male and female predictions.
(D-E) Mean Age Residual (AAR) differences across biological sex and COVID-19 impact.
(F-H) Accelerated aging analysis for specific clinical cohorts (Hypertension, Immune System, Neuro-Psych).

Outputs: 
- Figure2_Master_Complete.pdf (180mm standard width)
- AAR_Clinical_Categories_Export.csv (Data export for clinical patients)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import matplotlib.gridspec as gridspec
import re
import os

# ==========================================
# 1. Nature Style & Settings
# ==========================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

# Color Palette
c_main = '#E69F00'
c_male = '#D55E00'
c_female = '#0072B2'

# ==========================================
# 2. Auxiliary Plotting Functions
# ==========================================

def format_pval(p):
    """Formats p-values for Nature-compliant display."""
    if p < 0.001:
        return r"$\mathit{P} < 0.001$"
    else:
        return r"$\mathit{P}$ = " + f"{p:.3f}"

def add_stat_bracket(ax, x1, x2, y, p_val, height=0.5):
    """Adds a statistical significance bracket between two groups on a bar plot."""
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=0.8, c='k')
    text = "ns" if p_val > 0.05 else format_pval(p_val)
    ax.text((x1+x2)*.5, y+height+0.1, text, ha='center', va='bottom', color='k', fontsize=6)

def plot_raincloud_with_stats(df, ax):
    """
    Creates a custom raincloud plot (half-violin + boxplot + scatter) 
    to visualize the systematic offset between male and female residuals.
    """
    male = df[df['Sex']=='Male']['residual'].values
    female = df[df['Sex']=='Female']['residual'].values

    # Half-violins
    parts = ax.violinplot([female, male], positions=[0, 1], vert=False, showextrema=False, widths=0.8)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(c_female if i==0 else c_male)
        pc.set_alpha(0.5)

    # Boxplots
    bp = ax.boxplot([female, male], positions=[0, 1], vert=False, widths=0.15, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], [c_female, c_male]):
        patch.set_facecolor(color)
        patch.set_linewidth(0.5)
    plt.setp(bp['medians'], color='black')

    # Scatter points (jittered)
    np.random.seed(42)
    ax.scatter(female, np.random.normal(0, 0.04, size=len(female)) + 0.25, s=1, color=c_female, alpha=0.1, rasterized=True)
    ax.scatter(male, np.random.normal(1, 0.04, size=len(male)) + 0.25, s=1, color=c_male, alpha=0.1, rasterized=True)

    # Statistics
    diff = male.mean() - female.mean()
    p_val = stats.ttest_ind(male, female, equal_var=False).pvalue

    ax.text(0.05, 0.95, f"Gap = {diff:.1f} yr\n{format_pval(p_val)}", 
            transform=ax.transAxes, ha='left', va='top', fontsize=6)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Female', 'Male'])
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

# ==========================================
# 3. Data Processing & Clinical Categorization
# ==========================================

def load_and_prep_all_data():
    """
    Loads primary predictions, external validation sets, and clinical tables.
    Calculates residuals, Accelerated Aging Rates (AAR), and flags clinical 
    conditions using regex patterns on medical text fields.
    """
    # Load Primary Cohort
    df_primary_full = pd.read_csv('test_preds_covid.csv', low_memory=False)

    sex_col_cov = next((c for c in df_primary_full.columns if c.lower() in ['biological sex', 'sex', 'biological_sex']), 'Sex')
    df_primary_full.rename(columns={sex_col_cov: 'Sex', 'Age': 'y_true'}, inplace=True)
    df_primary_full['residual'] = df_primary_full['y_pred'] - df_primary_full['y_true']
    df_primary_full['Sex'] = df_primary_full['Sex'].astype(str).str.strip().str.capitalize().replace({'1': 'Male', '1.0': 'Male', '0': 'Female', '0.0': 'Female'})
    df_primary_full = df_primary_full[df_primary_full['Sex'].isin(['Male', 'Female'])]

    # Map COVID Impact
    df_elihay = pd.read_csv('TCR_shared_ELIHAY_full.csv')
    id_to_infected = dict(zip(df_elihay['Sample_ID'].astype(str), df_elihay['Infected']))

    def categorize_covid(name):
        name_str = str(name)
        for e_id, inf in id_to_infected.items():
            if e_id in name_str:
                return 'COVID Impacted' if pd.notna(inf) and str(inf).strip().split('.')[0] == '1' else 'Ref. Healthy'
        return 'COVID Impacted'

    df_primary_full['Group'] = df_primary_full['sample name'].apply(categorize_covid)

    # Load External Validation Cohort
    df_ext_full = pd.read_csv('test_preds_emerson_CLEAN_final_124.csv')
    df_ext_full.rename(columns={'biological_sex': 'Sex'}, inplace=True)
    df_ext_full['residual'] = df_ext_full['y_pred'] - df_ext_full['y_true']
    df_ext_full['Sex'] = df_ext_full['Sex'].astype(str).str.strip().str.capitalize()
    df_ext_full = df_ext_full[df_ext_full['Sex'].isin(['Male', 'Female'])]
    df_ext_full['Group'] = 'Healthy'

    # Filter 90th percentile to clean extreme outliers for statistical comparisons
    cut90_cov = df_primary_full['residual'].abs().quantile(0.90)
    df_primary_clean = df_primary_full[df_primary_full['residual'].abs() <= cut90_cov].copy()

    cut90_ext = df_ext_full['residual'].abs().quantile(0.90)
    df_ext_clean = df_ext_full[df_ext_full['residual'].abs() <= cut90_ext].copy()

    df_stats_clean = pd.concat([df_primary_clean, df_ext_clean], ignore_index=True)

    # Load Clinical Cohort and Calculate AAR Baseline
    df_sick = pd.read_csv('FINAL_Clinical_Case_Study_Table.csv', low_memory=False)
    sick_id_col = 'sample name' if 'sample name' in df_sick.columns else df_sick.columns[0]
    sick_samples = df_sick[sick_id_col].astype(str).unique()

    df_healthy_raw = df_primary_full[~df_primary_full['sample name'].astype(str).isin(sick_samples)].copy()
    slope_init, intercept_init, _, _, _ = stats.linregress(df_healthy_raw['y_true'], df_healthy_raw['y_pred'])
    df_healthy_raw['AAR_abs'] = (df_healthy_raw['y_pred'] - (slope_init * df_healthy_raw['y_true'] + intercept_init)).abs()

    threshold_95 = df_healthy_raw['AAR_abs'].quantile(0.95)
    df_healthy_clin = df_healthy_raw[df_healthy_raw['AAR_abs'] <= threshold_95].copy()

    # Final clinical slope based on healthy reference
    slope_clin, intercept_clin, _, _, _ = stats.linregress(df_healthy_clin['y_true'], df_healthy_clin['y_pred'])
    df_healthy_clin['AAR'] = df_healthy_clin['y_pred'] - (slope_clin * df_healthy_clin['y_true'] + intercept_clin)

    age_col_sick = 'Age' if 'Age' in df_sick.columns else 'y_true'
    df_sick['AAR'] = df_sick['y_pred'] - (slope_clin * df_sick[age_col_sick] + intercept_clin)

    # NLP Parsing for Clinical Categories
    text_cols = ['current_medications', 'diseases', 'selected_autoimmune_diagnoses', 'selected_other_diagnoses', 'describe_other_diagnoses', 'describe_immunosupressants', 'describe_cancers', 'describe_autoimmune_medications', 'describe_autoimmune_diagnoses', 'cancer_type', 'cancer_diagnosed', 'nsaid_type']

    mega_categories = {
        'Hypertension': {
            'words': ['lisinopril', 'amlodipine', 'losartan', 'candesartan', 'olmesartan', 'hctz', 'hydrochlorothiazide', 'atenolol', 'hypertension', 'high blood pressure', 'uses_ace_inhibitor', 'uses_arb'],
            'flags': ['has_chronic_hypertension']
        },
        'Immune System': {
            'words': ['hashimoto', 'rheumatoid', 'crohn', 'psoriasis', 'ulcerative colitis', 'lupus', 'arthritis', 'immunosuppressant', 'transplant', 'immunodeficiency', 'orencia', 'stelara', 'humira', 'enbrel', 'dupixent', 'plaquenil', 'uses_immunosuppressant', 'uses_autoimmune_medications'],
            'flags': ['is_immunocompromised', 'uses_immunosuppressant']
        },
        'Neuro-Psych': {
            'words': ['depression', 'anxiety', 'stress', 'escitalopram', 'citalopram', 'prozac', 'zoloft', 'cymbalta', 'wellbutrin', 'xanax', 'alprazolam', 'epilepsy', 'bipolar', 'seizure', 'lithium', 'lamictal', 'trileptal', 'ambien', 'provigil'],
            'flags': []
        }
    }

    for cat, config in mega_categories.items():
        pattern = r'\b(' + '|'.join(map(re.escape, config['words'])) + r')\b'
        text_hits = df_sick[text_cols].fillna('').astype(str).apply(lambda x: x.str.contains(pattern, case=False, na=False)).any(axis=1)

        flag_hits = pd.Series(False, index=df_sick.index)
        for f in config['flags']:
            if f in df_sick.columns:
                flag_hits = flag_hits | df_sick[f].apply(lambda x: str(x).lower() in ['1', '1.0', 'true', 'yes', 'positive']).fillna(False)
        df_sick[cat] = (text_hits | flag_hits).astype(int)

    return df_primary_full, df_ext_full, df_primary_clean, df_ext_clean, df_stats_clean, df_sick, df_healthy_clin, slope_clin, intercept_clin, age_col_sick, sick_id_col

# ==========================================
# 4. Master Figure Construction
# ==========================================

def create_final_figure():
    """Generates the master 3-row figure and exports clinical AAR data."""
    (df_primary_full, df_ext_full, df_primary_clean, df_ext_clean,
     df_stats_clean, df_sick, df_healthy_clin, slope_clin, intercept_clin, age_col_sick, sick_id_col) = load_and_prep_all_data()

    # 180mm width standard, compact height to bind rows together
    fig = plt.figure(figsize=(7.08, 7.2), dpi=300)
    
    gs = gridspec.GridSpec(3, 12, height_ratios=[1, 1, 1], hspace=0.6, wspace=1.6, 
                           left=0.08, right=0.98, top=0.95, bottom=0.08)

    def add_panel_letter(ax, letter):
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, 
                fontsize=7, fontweight='bold', va='bottom', ha='right')

    # ------------------------------------------
    # Row 1: Regression Models (Panels a, b)
    # ------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0:6])
    mae = mean_absolute_error(df_primary_full['y_true'], df_primary_full['y_pred'])
    ax1.scatter(df_primary_full['y_true'], df_primary_full['y_pred'], color=c_main, alpha=0.6, s=8, edgecolors='none', rasterized=True)
    ax1.plot([0, 100], [0, 100], 'k--', lw=0.8)
    ax1.set_title('Primary Cohort (Test)', loc='center')
    add_panel_letter(ax1, 'a')
    ax1.set_xlabel('Chronological Age (yr)')
    ax1.set_ylabel('Predicted Age (yr)', labelpad=6)
    ax1.text(0.05, 0.85, r"$\mathit{n}$" + f" = {len(df_primary_full)}\nMAE = {mae:.2f} yr", transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[0, 6:12])
    mae_h = mean_absolute_error(df_ext_full['y_true'], df_ext_full['y_pred'])
    ax2.scatter(df_ext_full['y_true'], df_ext_full['y_pred'], color=c_main, alpha=0.6, s=8, edgecolors='none', rasterized=True)
    ax2.plot([0, 100], [0, 100], 'k--', lw=0.8)
    ax2.set_title('External Validation Cohort (Test)', loc='center')
    add_panel_letter(ax2, 'b')
    ax2.set_xlabel('Chronological Age (yr)')
    ax2.set_ylabel('Predicted Age (yr)', labelpad=6)
    ax2.text(0.05, 0.85, r"$\mathit{n}$" + f" = {len(df_ext_full)}\nMAE = {mae_h:.2f} yr", transform=ax2.transAxes)

    # ------------------------------------------
    # Row 2: Sex-Specific Biases (Panels c, d, e)
    # ------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0:4])
    plot_raincloud_with_stats(df_stats_clean, ax3)
    ax3.set_title('Systematic Offset', loc='center')
    add_panel_letter(ax3, 'c')
    ax3.set_xlabel('Residual (Predicted - True)')

    ax4 = fig.add_subplot(gs[1, 4:9])
    sns.barplot(data=df_primary_clean, x='Group', y='residual', hue='Sex',
                palette={'Female': c_female, 'Male': c_male},
                order=['Ref. Healthy', 'COVID Impacted'], capsize=0.1, errwidth=1.5, ax=ax4)
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_title('Primary Cohort', loc='center')
    add_panel_letter(ax4, 'd')
    ax4.set_ylabel('Mean Age Residual (yr)', labelpad=-1)
    ax4.set_xlabel('')
    ax4.legend(loc='lower right', frameon=False)

    n_ref = len(df_primary_clean[df_primary_clean['Group'] == 'Ref. Healthy'])
    n_imp = len(df_primary_clean[df_primary_clean['Group'] == 'COVID Impacted'])
    ax4.set_xticklabels([f'Ref. Healthy\n(' + r"$\mathit{n}$" + f'={n_ref})', f'COVID Impacted\n(' + r"$\mathit{n}$" + f'={n_imp})'])

    ax5 = fig.add_subplot(gs[1, 9:12], sharey=ax4)
    ax5.set_facecolor('#F2F2F2')
    sns.barplot(data=df_ext_clean, x='Group', y='residual', hue='Sex',
                palette={'Female': c_female, 'Male': c_male},
                dodge=True, capsize=0.1, errwidth=1.5, ax=ax5)
    ax5.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.set_title('External Val. Cohort', loc='center')
    add_panel_letter(ax5, 'e')
    ax5.set_ylabel('')
    ax5.set_xlabel('')

    n_ext = len(df_ext_clean)
    ax5.set_xticklabels([f'Healthy\n(' + r"$\mathit{n}$" + f'={n_ext})'])
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.tick_params(axis='y', which='both', length=0)
    ax5.spines['left'].set_visible(False)
    if ax5.get_legend() is not None: ax5.get_legend().remove()

    # Dynamic scaling for aligned bar plots
    y_min_val = min(ax4.get_ylim()[0], ax5.get_ylim()[0]) - 0.8
    y_max_val = max(ax4.get_ylim()[1], ax5.get_ylim()[1]) + 2.0
    ax4.set_ylim(bottom=y_min_val, top=y_max_val)

    # Statistical Annotations
    m1 = df_primary_clean[(df_primary_clean['Group']=='Ref. Healthy') & (df_primary_clean['Sex']=='Male')]['residual']
    f1 = df_primary_clean[(df_primary_clean['Group']=='Ref. Healthy') & (df_primary_clean['Sex']=='Female')]['residual']
    add_stat_bracket(ax4, -0.2, 0.2, max(m1.mean() + m1.sem()*1.96, f1.mean() + f1.sem()*1.96) + 0.3, stats.ttest_ind(m1, f1, equal_var=False).pvalue, 0.3)

    m2 = df_primary_clean[(df_primary_clean['Group']=='COVID Impacted') & (df_primary_clean['Sex']=='Male')]['residual']
    f2 = df_primary_clean[(df_primary_clean['Group']=='COVID Impacted') & (df_primary_clean['Sex']=='Female')]['residual']
    add_stat_bracket(ax4, 0.8, 1.2, max(m2.mean() + m2.sem()*1.96, f2.mean() + f2.sem()*1.96) + 0.3, stats.ttest_ind(m2, f2, equal_var=False).pvalue, 0.3)

    m3 = df_ext_clean[df_ext_clean['Sex']=='Male']['residual']
    f3 = df_ext_clean[df_ext_clean['Sex']=='Female']['residual']
    add_stat_bracket(ax5, -0.2, 0.2, max(m3.mean() + m3.sem()*1.96, f3.mean() + f3.sem()*1.96) + 0.3, stats.ttest_ind(m3, f3, equal_var=False).pvalue, 0.3)

    # ------------------------------------------
    # Row 3: Clinical Cohorts (Panels f, g, h)
    # ------------------------------------------
    categories = ['Hypertension', 'Immune System', 'Neuro-Psych']
    letters_fgh = ['f', 'g', 'h']
    
    limit_min = min(df_sick[age_col_sick].min(), df_sick['y_pred'].min(), df_healthy_clin['y_true'].min()) - 5
    limit_max = max(df_sick[age_col_sick].max(), df_sick['y_pred'].max(), df_healthy_clin['y_true'].max()) + 5

    export_data = []

    for i, cat in enumerate(categories):
        ax = fig.add_subplot(gs[2, i*4:(i+1)*4])
        ax.set_facecolor('#FAFAFA')

        # Background healthy points
        ax.scatter(df_healthy_clin['y_true'], df_healthy_clin['y_pred'], c='silver', alpha=0.3, s=6, edgecolors='none', zorder=1, rasterized=True)
        x_vals = np.array([limit_min, limit_max])
        ax.plot(x_vals, slope_clin * x_vals + intercept_clin, color='k', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)

        subset = df_sick[df_sick[cat] == 1]

        if subset.empty:
            ax.set_title(f"{cat} (No Data)", loc='center')
            add_panel_letter(ax, letters_fgh[i])
            continue

        point_color = 'royalblue' if cat == 'Hypertension' else 'crimson'
        ax.scatter(subset[age_col_sick], subset['y_pred'], c=point_color, alpha=0.7, s=15, edgecolors='black', linewidth=0.5, zorder=3, rasterized=True)

        for _, row in subset.iterrows():
            expected_y = slope_clin * row[age_col_sick] + intercept_clin
            ax.plot([row[age_col_sick], row[age_col_sick]], [expected_y, row['y_pred']], color=point_color, alpha=0.4, linewidth=1, zorder=2)

            export_data.append({
                'Category': cat,
                'Sample_ID': row.get(sick_id_col, 'N/A'),
                'Chronological_Age': row[age_col_sick],
                'Predicted_Age': row['y_pred'],
                'AAR_Value': row['AAR'],
            })

        _, p_val = stats.ttest_ind(subset['AAR'].dropna(), df_healthy_clin['AAR'].dropna(), equal_var=False)
        mean_aar = subset['AAR'].mean()

        ax.set_title(f"{cat}", loc='center')
        add_panel_letter(ax, letters_fgh[i])
        ax.set_xlabel('Chronological Age (yr)')
        
        if i == 0: 
            ax.set_ylabel('Predicted Age (yr)', labelpad=6)

        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)

        ax.text(0.05, 0.76, r"$\mathit{n}$" + f" = {len(subset)}\nMean AAR = {mean_aar:+.1f} yr\n{format_pval(p_val)}", transform=ax.transAxes, fontsize=6)

    # General cleanup
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', pad=1, labelsize=6)

    plt.savefig('Figure2_Master_Complete.pdf', dpi=300, transparent=True)
    plt.show()

    if export_data:
        df_export = pd.DataFrame(export_data)
        df_export.to_csv('AAR_Clinical_Categories_Export.csv', index=False)
        print("✅ Exported clinical AAR values to AAR_Clinical_Categories_Export.csv")

if __name__ == "__main__":
    create_final_figure()