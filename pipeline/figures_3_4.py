import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent)); import config as C
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'sans-serif',
 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.linewidth':0.5,
 'xtick.major.width':0.5,'ytick.major.width':0.5})
c_main='#E69F00'; c_recal='#009E73'; FIG=str(C.FIGURES)
def clean(ax): ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=6,pad=1)

# ===== Figure 4b: corrected antigen-species ordering =====
# C15 (2026-09-08): antigen_age_bias.csv is a SHIPPED input under results/tables/, but
# C.TABLES points at $TTIME_WORK/results/tables which is empty in a fresh workspace.
# Prefer the workspace copy, fall back to the repo's.
_tab=C.TABLES/'antigen_age_bias.csv'
if not _tab.exists(): _tab=Path(__file__).resolve().parents[1]/'results'/'tables'/'antigen_age_bias.csv'
sp=pd.read_csv(_tab).sort_values('median')
fig,ax=plt.subplots(figsize=(3.4,2.6),dpi=300)
colors=['#0072B2' if m<0 else '#D55E00' for m in sp['median']]
ax.hlines(range(len(sp)),0,sp['median'],color=colors,lw=1)
ax.scatter(sp['median'],range(len(sp)),color=colors,s=[18+c*0.05 for c in sp['count']],zorder=3)
ax.set_yticks(range(len(sp))); ax.set_yticklabels([f"{s} (n={c})" for s,c in zip(sp['cat'],sp['count'])],fontsize=6)
ax.axvline(0,color='grey',ls='--',lw=0.8); ax.set_xlabel('Median signed-Wasserstein (youth < 0)')
ax.set_title('Antigen-category age bias',fontsize=7)
ax.text(0.02,0.06,'all viral species youth-skewed;\nCMV/EBV least so, not old-biased',transform=ax.transAxes,fontsize=5.5,style='italic')
clean(ax); fig.tight_layout()
fig.savefig(f'{FIG}/Figure4b.pdf',dpi=300,transparent=True); fig.savefig(f'{FIG}/Figure4b.png',dpi=150,bbox_inches='tight')
print('Figure 4b saved')

# ===== Figure 3 additions: baselines + MAIT depletion =====
B=json.load(open(C.OUTPUTS/'baselines.json'))
order=[('diversity_ridge','Diversity\n(Ridge)'),('diversity_gbm','Diversity\n(GBM)'),
       ('ridge_features','Ridge\n(features)'),('elasticnet_features','ElasticNet\n(features)'),('_mlp','T-Time\nMLP')]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.6,2.7),dpi=300,gridspec_kw={'width_ratios':[2.2,1]})
# C3: diversity_ridge / diversity_gbm need the raw repertoire TSVs (config.TRAIN_DIR /
# TEST_DIR) and are absent from baselines.json in a data-only checkout. Drop whatever the
# file does not carry instead of dying with a KeyError.
_missing=[k for k,_ in order if k not in B]
if _missing: print(f'[!] baselines.json lacks {_missing} - omitting those bars')
order=[(k,n) for k,n in order if k in B]
names=[n for k,n in order]; maes=[B[k]['MAE'] for k,_ in order]; r2s=[B[k]['R2'] for k,_ in order]
cols=[c_main if k=='_mlp' else '#BBBBBB' for k,_ in order]
x=np.arange(len(names))
ax1.bar(x,maes,color=cols,width=0.62)
for i,(k,_) in enumerate(order):
    ci=B[k].get('MAE_CI')
    if ci: ax1.plot([i,i],ci,color='k',lw=1)
ax1.set_xticks(x); ax1.set_xticklabels(names,fontsize=6); ax1.set_ylabel('Test MAE (yr)')
ax1.set_title('Benchmark vs interpretable baselines (R1.2, R2)',fontsize=6.5)
for i,v in enumerate(maes): ax1.text(i,v+0.3,f'{v:.1f}',ha='center',fontsize=5.5)
ax1.text(-0.14,1.06,'a',transform=ax1.transAxes,fontsize=8,fontweight='bold'); clean(ax1)
# MAIT depletion
M=json.load(open(C.OUTPUTS/'mait_classifier.json'))
ax2.bar([0,1],[M['full']['AUC'],M['mait_depleted']['AUC']],color=[c_main,c_recal],width=0.6)
ax2.set_ylim(0.5,0.95); ax2.set_xticks([0,1]); ax2.set_xticklabels(['All\nsignificant','MAIT-\ndepleted'],fontsize=6)
ax2.set_ylabel('Young-vs-old AUC')
for i,v in enumerate([M['full']['AUC'],M['mait_depleted']['AUC']]): ax2.text(i,v+0.005,f'{v:.3f}',ha='center',fontsize=6)
ax2.set_title('MAIT not driving\nseparability (R2)',fontsize=6.5)
ax2.text(-0.28,1.06,'b',transform=ax2.transAxes,fontsize=8,fontweight='bold'); clean(ax2)
fig.tight_layout()
fig.savefig(f'{FIG}/Figure3_additions.pdf',dpi=300,transparent=True); fig.savefig(f'{FIG}/Figure3_additions.png',dpi=150,bbox_inches='tight')
print('Figure 3 additions saved')
