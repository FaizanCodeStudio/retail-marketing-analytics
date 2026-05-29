# 🛒 Retail Marketing Analytics
### Customer Segmentation & Growth Strategy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end retail analytics project that combines **RFM analysis**, **K-Means customer clustering**, and **marketing channel ROI evaluation** to derive actionable growth strategies for a retail business

---

## 📌 Project Overview

| Dimension | Detail |
|-----------|--------|
| **Domain** | Retail / E-commerce |
| **Goal** | Segment customers, measure marketing effectiveness, recommend growth levers |
| **Dataset** | Synthetic data modelled on Kaggle retail + marketing datasets |
| **Records** | 5,000 sales transactions · 294 monthly marketing records |
| **Date Range** | Jan 2020 – Dec 2023 |

---

## 📁 Project Structure

```
retail-marketing-analytics/
│
├── data/
│   ├── retail_sales.csv          # 5,000 transaction records
│   └── marketing_trends.csv      # 294 monthly campaign records
│
├── src/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── analysis.py               # Full analysis pipeline
│
├── outputs/
│   ├── 01_eda_overview.png       # Exploratory data analysis
│   ├── 02_rfm_analysis.png       # RFM segment breakdown
│   ├── 03_kmeans_segmentation.png # K-Means elbow, silhouette & PCA
│   ├── 04_marketing_analysis.png  # Channel ROI & performance
│   ├── 05_growth_dashboard.png    # Executive growth dashboard
│   ├── rfm_segments.csv          # Customer-level RFM scores
│   └── campaign_performance.csv  # Aggregated campaign metrics
│
├── reports/
│   └── GROWTH_STRATEGY.md        # Detailed strategy recommendations
│
├── requirements.txt
└── README.md
```

---

## 🔍 Analysis Modules

### 1. Exploratory Data Analysis
- Revenue breakdown by product category, region, sales channel
- Monthly revenue trend (2020–2023)
- Discount sensitivity analysis
- Demographics (age group, gender)

### 2. RFM Segmentation
Customers scored on **Recency · Frequency · Monetary** and grouped into:

| Segment | Description | Strategy |
|---------|-------------|----------|
| 🏆 Champions | High R, F & M | Reward & upsell |
| 💚 Loyal | Regular buyers | Cross-sell & retain |
| 🌱 Potential | Promising recent buyers | Nurture & activate |
| ⚠️ At Risk | Declining engagement | Win-back campaigns |
| 💔 Lost | Long inactive | Re-engagement or drop |

### 3. K-Means Clustering
- Optimal **K selected via silhouette score**
- Elbow method for validation
- **PCA** for 2-D cluster visualization
- Cluster profiling by mean R, F, M values

### 4. Marketing Channel Analysis
- ROI, CAC, and conversion rate per campaign type
- Monthly revenue trend by channel
- Spend vs. return scatter analysis

### 5. Growth Strategy Dashboard
- Dark-mode executive summary with KPI tiles
- Segment, channel, and campaign insights in a single view

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total Revenue | ₹7.2M+ |
| Unique Customers | ~5,000 |
| Best ROI Channel | Search Ads |
| Lowest CAC Channel | Email |
| Optimal K-Means K | 7 |
| Silhouette Score | 0.386 |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/retail-marketing-analytics.git
cd retail-marketing-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate datasets
python src/generate_data.py

# 4. Run full analysis
python src/analysis.py

# 5. View outputs/
```

---

## 📈 Sample Outputs

| Chart | Description |
|-------|-------------|
| `01_eda_overview.png` | 6-panel EDA dashboard |
| `02_rfm_analysis.png` | RFM segment counts, revenue, scatter |
| `03_kmeans_segmentation.png` | Elbow + Silhouette + PCA cluster plot |
| `04_marketing_analysis.png` | ROI, spend, monthly trends |
| `05_growth_dashboard.png` | Dark-mode executive summary |

---

## 📋 Growth Recommendations

See [`reports/GROWTH_STRATEGY.md`](reports/GROWTH_STRATEGY.md) for the full strategy document.

**TL;DR:**
1. **Champions & Loyals** → Premium loyalty program + early access
2. **Potentials** → Automated nurture email sequences
3. **At Risk** → Time-limited win-back discounts
4. **Shift budget** → Email & Search Ads (highest ROI, lowest CAC)
5. **Invest** → Mobile App channel (fastest growing, under-spent)

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data wrangling
- **Matplotlib / Seaborn** — visualisation
- **scikit-learn** — KMeans, PCA, StandardScaler, silhouette_score
- **Plotly** — interactive charts (optional extension)

---

## 📄 License

MIT © 2024
