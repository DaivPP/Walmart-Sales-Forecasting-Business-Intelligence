# Walmart Sales Forecasting & Business Intelligence

> Time series forecasting on 143 weeks of real Walmart sales data — with ARIMA vs Prophet model comparison, holiday impact analysis, and store-level business insights.

---

## The Problem

Retail demand forecasting is one of the most high-value applications of data science. Accurate weekly sales forecasts allow retailers to optimize inventory, staffing, and promotions. This project builds and compares two forecasting models on 2.8 years of real Walmart sales data across 45 stores and 81 departments, then extracts actionable business insights from the patterns found.

---

## Results at a Glance

### Forecasting (Store 20, Dept 92 — Highest Revenue Department)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive Baseline (mean) | — | — | 8.09% |
| ARIMA(2,1,2) ✓ | $13,435 | $15,352 | 7.66% |
| Prophet | $14,157 | $16,153 | 8.23% |

**ARIMA outperformed Prophet on the June–October test window by 0.57 MAPE points.** This period contains no major US retail holidays. Prophet's primary advantage is holiday and seasonality modelling — on a full-year test window including November–December, Prophet is expected to outperform due to Thanksgiving's 39.7% sales lift effect.

### Business Insights

| Finding | Value |
|---|---|
| Total revenue (2010–2012) | $6.74 billion |
| Thanksgiving sales lift | +39.7% vs average week |
| Christmas week sales | -8.5% (post-holiday slump) |
| Super Bowl lift | +3.0% |
| Type A vs Type C stores | 2.1x higher average weekly sales |
| Best performing store | Store 20 (Type A, $29,508/week avg) |

---

## Why This Problem is Hard

Forecasting retail sales has three compounding challenges:

**1. Multiple seasonality layers** — weekly patterns, monthly patterns, annual holiday spikes, and year-over-year trends all overlap. A model that captures one may miss another.

**2. External factors** — fuel prices, unemployment, temperature, and promotional markdowns all affect sales independently of historical patterns.

**3. 3,331 unique time series** — 45 stores × 81 departments, each with its own demand pattern. A model that works for a grocery department won't work for an electronics department.

---

## Project Structure

```
walmart-sales-forecasting/
├── walmart_forecasting.ipynb   # Full analysis notebook
├── train.csv                   # Weekly sales per store/dept
├── stores.csv                  # Store type and size
├── features.csv                # External factors (temp, fuel, CPI etc.)
├── plots/
│   ├── overall_sales_trend.html
│   ├── sales_by_store_type.html
│   ├── store_performance.html
│   ├── monthly_seasonality.html
│   ├── top_departments.html
│   ├── external_factors.html
│   ├── dept92_raw_series.html
│   ├── arima_forecast.html
│   ├── prophet_vs_arima.html
│   ├── prophet_components.png
│   ├── holiday_lift.html
│   └── dept_month_heatmap.png
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis

Three data sources merged into one master dataset (421,570 rows):
- **train.csv** — weekly sales per store/department
- **stores.csv** — store type (A/B/C) and size in sq ft
- **features.csv** — external factors: temperature, fuel price, CPI, unemployment, markdowns

Key findings from EDA:
- **Store type drives revenue:** Type A stores average $20,100/week vs $9,520 for Type C — a 2.1x gap strongly correlated with store size
- **Thanksgiving dominates:** +39.7% sales lift, the single largest demand event in the dataset
- **Christmas is a trap:** The dataset marks the post-Christmas week as the holiday — this week shows -8.5% sales vs average, because the buying already happened in November/early December
- **Departments 92, 95, 38** are the top 3 revenue drivers across all stores ($484M, $449M, $393M respectively over 3 years)

### 2. Feature Engineering

Five categories of features created:

- **Time features:** Year, Month, Week, Quarter — tells the model where in the calendar each row sits
- **Cyclical encoding:** Week converted to sine/cosine to fix the "week 52 is close to week 1" problem that naive numeric encoding misses
- **Lag features:** Sales from 1 week ago, 4 weeks ago, and 52 weeks ago (same week last year)
- **Rolling averages:** 4-week and 12-week rolling means to capture short and medium-term trend direction
- **Holiday season flag:** Binary indicator for November/December weeks

### 3. Temporal Train/Test Split

**Critical decision:** Time series data must be split chronologically, never randomly. Random splitting allows the model to "see the future" during training — for example, training on December 2012 data while testing on June 2011. This inflates performance metrics artificially.

Split: Train on Feb 2010 – May 2012 (121 weeks), test on Jun–Oct 2012 (22 weeks).

### 4. ARIMA Baseline

ARIMA(2,1,2) fitted on Store 20, Dept 92:
- **p=2:** Uses last 2 weeks of sales as autoregressive inputs
- **d=1:** One round of differencing applied — ADF test confirmed series was non-stationary (p=0.51)
- **q=2:** Moving average over last 2 forecast errors

Achieved **MAPE of 7.66%** — meaning average weekly forecast error of ~$13,400 on a department averaging ~$175,000/week in sales.

### 5. Prophet Model

Facebook Prophet configured with:
- **Yearly seasonality:** Captures the Nov/Dec annual peak
- **US retail holidays:** Thanksgiving (7-day pre-window), Christmas (7-day pre-window), Super Bowl, Labour Day explicitly injected
- **Multiplicative seasonality:** Holiday lifts scale with trend rather than being fixed dollar amounts — more realistic for growing retail sales

Prophet achieved **MAPE of 8.23%** on the June–October test window — slightly behind ARIMA. **This is an honest result with a clear explanation:** the test window contains no major holidays, so Prophet's primary advantage never activates. On a full-year test including Thanksgiving and Christmas, Prophet's holiday modelling is expected to produce lower MAPE.

### 6. Business Impact Analysis

Beyond model metrics, the project answers questions a retail manager actually cares about:

- Which stores are growing fastest year-over-year?
- Which holiday drives the most incremental revenue?
- Which departments are most sensitive to seasonal demand?
- How much revenue is at stake during Thanksgiving week specifically?

---

## Key Takeaways

**On the modelling side:**
- Always use temporal splits for time series — random splits cause data leakage
- ARIMA's d parameter handles non-stationarity; always run an ADF test before fitting
- Prophet's advantage is seasonal/holiday modelling — it needs a full-year test window to demonstrate this
- A model that wins on one test window may lose on another — evaluate on multiple periods before drawing conclusions

**On the business side:**
- Thanksgiving is 6x more impactful than any other holiday (+39.7% vs Super Bowl's +3%)
- The post-Christmas week is a sales slump, not a peak — brands that overstock for "Christmas week" based on the holiday flag are making a data interpretation error
- Type A store performance is driven by size, not just location — the correlation between sq ft and weekly sales is strong

---

## Dataset

[Walmart Store Sales Forecasting — Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

Weekly sales data for 45 Walmart stores across the US, covering Feb 2010 – Oct 2012. Includes external economic indicators and promotional markdown data.

---

## Requirements

```
pandas numpy matplotlib seaborn plotly prophet statsmodels scikit-learn
```

Install: `pip install pandas numpy matplotlib seaborn plotly prophet statsmodels scikit-learn`

---

## Author

**Daivansh Pushkarna**
[LinkedIn](https://www.linkedin.com/in/daivanshp4) • [GitHub](https://github.com/DaivPP)
