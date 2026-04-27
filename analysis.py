"""
Retail Marketing Analytics
Customer Segmentation & Growth Strategy
========================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────
PALETTE = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#264653','#A8DADC','#6D6875']
sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150})

# ════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("RETAIL MARKETING ANALYTICS — ANALYSIS PIPELINE")
print("=" * 60)

sales = pd.read_csv('data/retail_sales.csv', parse_dates=['date'])
mktg  = pd.read_csv('data/marketing_trends.csv', parse_dates=['month'])

print(f"\n📦 Sales records  : {len(sales):,}")
print(f"📊 Marketing rows : {len(mktg):,}")
print(f"📅 Date range     : {sales['date'].min().date()} → {sales['date'].max().date()}")

# ════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("\n[1/5] Running EDA …")

# --- Revenue by Category ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Retail Sales — Exploratory Data Analysis', fontsize=18, fontweight='bold', y=1.01)

cat_rev = sales.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
axes[0,0].bar(cat_rev.index, cat_rev.values/1e6, color=PALETTE)
axes[0,0].set_title('Total Revenue by Category (₹M)', fontweight='bold')
axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('Revenue (₹M)')
axes[0,0].tick_params(axis='x', rotation=45)

# --- Revenue by Channel ---
ch_rev = sales.groupby('channel')['revenue'].sum()
axes[0,1].pie(ch_rev, labels=ch_rev.index, autopct='%1.1f%%', colors=PALETTE,
              startangle=140, wedgeprops={'edgecolor':'white','linewidth':1.5})
axes[0,1].set_title('Revenue Share by Channel', fontweight='bold')

# --- Monthly Revenue Trend ---
monthly = sales.set_index('date').resample('ME')['revenue'].sum().reset_index()
axes[0,2].plot(monthly['date'], monthly['revenue']/1e3, color=PALETTE[0], linewidth=2)
axes[0,2].fill_between(monthly['date'], monthly['revenue']/1e3, alpha=0.15, color=PALETTE[0])
axes[0,2].set_title('Monthly Revenue Trend (₹K)', fontweight='bold')
axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('Revenue (₹K)')
axes[0,2].tick_params(axis='x', rotation=30)

# --- Revenue by Region ---
reg_rev = sales.groupby('region')['revenue'].sum().sort_values(ascending=True)
axes[1,0].barh(reg_rev.index, reg_rev.values/1e6, color=PALETTE[2])
axes[1,0].set_title('Revenue by Region (₹M)', fontweight='bold')
axes[1,0].set_xlabel('Revenue (₹M)')

# --- Discount vs Revenue ---
axes[1,1].scatter(sales['discount_pct'], sales['revenue'], alpha=0.3, color=PALETTE[3], s=8)
axes[1,1].set_title('Discount % vs Revenue', fontweight='bold')
axes[1,1].set_xlabel('Discount (%)'); axes[1,1].set_ylabel('Revenue (₹)')

# --- Revenue by Age Group ---
age_rev = sales.groupby('age_group')['revenue'].mean().sort_index()
axes[1,2].bar(age_rev.index, age_rev.values, color=PALETTE[4])
axes[1,2].set_title('Avg Revenue by Age Group', fontweight='bold')
axes[1,2].set_xlabel('Age Group'); axes[1,2].set_ylabel('Avg Revenue (₹)')

plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', bbox_inches='tight')
plt.close()
print("   ✓ EDA chart saved")

# ════════════════════════════════════════════════════════════════════
# 3. RFM ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("[2/5] Building RFM model …")

snapshot = sales['date'].max() + pd.Timedelta(days=1)
rfm = sales.groupby('customer_id').agg(
    recency   = ('date', lambda x: (snapshot - x.max()).days),
    frequency = ('date', 'count'),
    monetary  = ('revenue', 'sum')
).reset_index()

# Score 1–5
for col, asc in [('recency',False),('frequency',True),('monetary',True)]:
    lbl = col[0].upper()
    bins = pd.qcut(rfm[col], 5, duplicates='drop', retbins=True)[1]
    n_bins = len(bins) - 1
    labels_asc   = list(range(1, n_bins+1))
    labels_desc  = list(range(n_bins, 0, -1))
    rfm[f'{lbl}_score'] = pd.cut(rfm[col], bins=bins, labels=labels_asc if asc else labels_desc,
                                  include_lowest=True).astype(int)

rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

def segment(s):
    if s >= 13: return 'Champions'
    if s >= 10: return 'Loyal'
    if s >= 7:  return 'Potential'
    if s >= 5:  return 'At Risk'
    return 'Lost'

rfm['segment'] = rfm['RFM_score'].apply(segment)
seg_counts = rfm['segment'].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('RFM Customer Analysis', fontsize=16, fontweight='bold')

axes[0].bar(seg_counts.index, seg_counts.values, color=PALETTE)
axes[0].set_title('Customer Count by Segment', fontweight='bold')
axes[0].set_ylabel('Customers')

seg_val = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=False)
axes[1].bar(seg_val.index, seg_val.values/1e6, color=PALETTE)
axes[1].set_title('Total Revenue by Segment (₹M)', fontweight='bold')
axes[1].set_ylabel('Revenue (₹M)')

axes[2].scatter(rfm['recency'], rfm['monetary'], c=rfm['RFM_score'],
                cmap='RdYlGn', alpha=0.5, s=15)
axes[2].set_title('Recency vs Monetary (colored by RFM)', fontweight='bold')
axes[2].set_xlabel('Recency (days)'); axes[2].set_ylabel('Total Spend (₹)')

plt.tight_layout()
plt.savefig('outputs/02_rfm_analysis.png', bbox_inches='tight')
plt.close()
print(f"   ✓ RFM segments: {rfm['segment'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════
# 4. K-MEANS CUSTOMER SEGMENTATION
# ════════════════════════════════════════════════════════════════════
print("[3/5] Running K-Means clustering …")

feats = rfm[['recency','frequency','monetary']].copy()
scaler = StandardScaler()
X = scaler.fit_transform(feats)

# Elbow + Silhouette
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

best_k = K_range[np.argmax(silhouettes)]
print(f"   Best K = {best_k} (silhouette = {max(silhouettes):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['cluster'] = km_final.fit_predict(X)

# PCA for 2-D viz
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('K-Means Customer Segmentation', fontsize=16, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'o-', color=PALETTE[0])
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')

axes[1].plot(list(K_range), silhouettes, 's-', color=PALETTE[1])
axes[1].axvline(best_k, color='red', linestyle='--', alpha=0.6, label=f'Best K={best_k}')
axes[1].legend()
axes[1].set_title('Silhouette Scores', fontweight='bold')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Score')

for cl in range(best_k):
    mask = rfm['cluster'] == cl
    axes[2].scatter(pca_coords[mask,0], pca_coords[mask,1],
                    label=f'Cluster {cl}', alpha=0.5, s=12, color=PALETTE[cl])
axes[2].set_title(f'PCA View — {best_k} Clusters', fontweight='bold')
axes[2].legend(); axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('outputs/03_kmeans_segmentation.png', bbox_inches='tight')
plt.close()

cluster_profile = rfm.groupby('cluster')[['recency','frequency','monetary']].mean().round(1)
print("   Cluster Profiles:\n", cluster_profile.to_string())

# ════════════════════════════════════════════════════════════════════
# 5. MARKETING CHANNEL ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("[4/5] Marketing channel analysis …")

camp_perf = mktg.groupby('campaign').agg(
    total_spend   = ('spend','sum'),
    total_rev     = ('revenue_generated','sum'),
    avg_roi       = ('roi','mean'),
    total_conv    = ('conversions','sum'),
    avg_cac       = ('cac','mean')
).round(2).sort_values('avg_roi', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Marketing Channel Performance', fontsize=16, fontweight='bold')

axes[0,0].barh(camp_perf.index, camp_perf['avg_roi'], color=PALETTE)
axes[0,0].set_title('Avg ROI by Campaign (%)', fontweight='bold')
axes[0,0].set_xlabel('ROI (%)')

axes[0,1].scatter(camp_perf['total_spend']/1e6, camp_perf['avg_roi'],
                  s=[c/50 for c in camp_perf['total_conv']],
                  color=PALETTE[:len(camp_perf)], alpha=0.8, zorder=3)
for i, row in camp_perf.iterrows():
    axes[0,1].annotate(i, (row['total_spend']/1e6, row['avg_roi']),
                       textcoords='offset points', xytext=(5,3), fontsize=8)
axes[0,1].set_title('Spend vs ROI (bubble=conversions)', fontweight='bold')
axes[0,1].set_xlabel('Total Spend (₹M)'); axes[0,1].set_ylabel('Avg ROI (%)')

monthly_camp = mktg.pivot_table(index='month', columns='campaign', values='revenue_generated', aggfunc='sum')
for col, color in zip(monthly_camp.columns, PALETTE):
    axes[1,0].plot(monthly_camp.index, monthly_camp[col]/1e3, label=col, color=color, linewidth=1.5)
axes[1,0].set_title('Monthly Revenue by Campaign (₹K)', fontweight='bold')
axes[1,0].legend(fontsize=7); axes[1,0].set_xlabel('')
axes[1,0].tick_params(axis='x', rotation=30)

axes[1,1].bar(camp_perf.index, camp_perf['avg_cac'], color=PALETTE)
axes[1,1].set_title('Avg Customer Acquisition Cost (₹)', fontweight='bold')
axes[1,1].set_ylabel('CAC (₹)'); axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('outputs/04_marketing_analysis.png', bbox_inches='tight')
plt.close()
print("   ✓ Marketing analysis complete")

# ════════════════════════════════════════════════════════════════════
# 6. GROWTH STRATEGY DASHBOARD
# ════════════════════════════════════════════════════════════════════
print("[5/5] Generating growth strategy summary …")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('RETAIL GROWTH STRATEGY DASHBOARD', fontsize=22,
             fontweight='bold', color='white', y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

def dark_ax(ax, title=''):
    ax.set_facecolor('#161B22')
    ax.spines['bottom'].set_color('#30363D')
    ax.spines['left'].set_color('#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#8B949E')
    ax.xaxis.label.set_color('#8B949E')
    ax.yaxis.label.set_color('#8B949E')
    if title: ax.set_title(title, color='white', fontweight='bold', pad=8)
    return ax

# --- KPIs ---
kpis = [
    ('Total Revenue', f"₹{sales['revenue'].sum()/1e6:.1f}M", '#2A9D8F'),
    ('Customers',     f"{sales['customer_id'].nunique():,}", '#457B9D'),
    ('Avg Order',     f"₹{sales['revenue'].mean():.0f}",    '#E9C46A'),
    ('Avg ROI',       f"{mktg['roi'].mean():.1f}%",          '#E63946'),
]
for i,(label,val,col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(col+'22')
    ax.spines[:].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5,0.62,val,ha='center',va='center',fontsize=26,
            fontweight='bold',color=col,transform=ax.transAxes)
    ax.text(0.5,0.28,label,ha='center',va='center',fontsize=11,
            color='#8B949E',transform=ax.transAxes)

# --- Segment distribution ---
ax2 = dark_ax(fig.add_subplot(gs[1,:2]), 'RFM Segment Distribution')
seg_pct = (rfm['segment'].value_counts()/len(rfm)*100).sort_values(ascending=True)
bars = ax2.barh(seg_pct.index, seg_pct.values, color=PALETTE[:len(seg_pct)])
for bar, pct in zip(bars, seg_pct.values):
    ax2.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             f'{pct:.1f}%', va='center', color='white', fontsize=9)
ax2.set_xlabel('% of Customers')

# --- Channel revenue ---
ax3 = dark_ax(fig.add_subplot(gs[1,2:]), 'Revenue by Sales Channel')
ch = sales.groupby('channel')['revenue'].sum().sort_values(ascending=False)
ax3.bar(ch.index, ch.values/1e6, color=PALETTE[2:])
ax3.set_ylabel('Revenue (₹M)')

# --- ROI by campaign ---
ax4 = dark_ax(fig.add_subplot(gs[2,:2]), 'Marketing ROI by Campaign')
roi_sort = camp_perf['avg_roi'].sort_values(ascending=True)
colors_roi = [PALETTE[0] if v < 0 else PALETTE[2] for v in roi_sort.values]
ax4.barh(roi_sort.index, roi_sort.values, color=colors_roi)
ax4.axvline(0, color='white', linewidth=0.8, alpha=0.4)
ax4.set_xlabel('Avg ROI (%)')

# --- Top categories ---
ax5 = dark_ax(fig.add_subplot(gs[2,2:]), 'Top Categories by Profit')
cat_profit = sales.groupby('product_category')['profit'].sum().sort_values(ascending=False)
ax5.bar(cat_profit.index, cat_profit.values/1e6, color=PALETTE)
ax5.set_ylabel('Profit (₹M)')
ax5.tick_params(axis='x', rotation=45)

plt.savefig('outputs/05_growth_dashboard.png', bbox_inches='tight',
            facecolor='#0D1117')
plt.close()

print("\n" + "="*60)
print("✅ ALL OUTPUTS SAVED TO outputs/")
print("="*60)

# Save cluster profiles and RFM for reporting
rfm.to_csv('outputs/rfm_segments.csv', index=False)
camp_perf.to_csv('outputs/campaign_performance.csv')
print("✓ rfm_segments.csv & campaign_performance.csv exported")
