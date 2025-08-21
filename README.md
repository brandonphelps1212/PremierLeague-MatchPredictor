# Premier League Match Predictor ⚽️

A machine learning project that predicts the outcome of Premier League matches using historical data, rolling statistics, and advanced features such as xG (expected goals) and shot metrics.

This project demonstrates:

- **Web scraping** with `requests` + `BeautifulSoup`
- **Data wrangling** with `pandas`
- **Feature engineering** (form, rolling averages, xG/xGA, shot quality)
- **Machine learning pipelines** with scikit-learn (Random Forests & Gradient Boosting)
- **Time-series validation** to avoid look-ahead bias
- End-to-end workflow: **data collection → feature engineering → model training → evaluation**

---

## 🚀 Project Overview

The workflow is split into two main parts:

1. **Data Collection**

   - `scrape.py`: Scrapes match data from [FBref](https://fbref.com/) (with polite headers & delays).
   - If FBref blocks requests, it automatically falls back to [football-data.co.uk](https://www.football-data.co.uk/) CSV feeds.
   - Produces a `matches.csv` file with one row per team per match.

2. **Prediction Model**
   - `predict.py`: Trains and evaluates models to predict whether a team will **win** a given match.
   - Supports:
     - Random Forest (`--model rf`)
     - Gradient Boosting (`--model gb`)
   - Includes **rolling features** like team form (last 3/5 matches), goals for/against, xG, shots, and on-target rates.
   - Uses **time-based train/test split** to simulate realistic forecasting.
   - Configurable decision threshold (`--threshold`) to trade precision vs recall.

---

## 📊 Results

With historical Premier League data (2017–2025 seasons):

- **Baseline accuracy**: ~66% on test data
- **Precision (predicting wins)**: ~0.55–0.60
- **Recall**: ~0.30–0.40 (can be increased by lowering decision threshold)
- Shows realistic performance given the unpredictable nature of football matches.

> ⚠️ Note: Football outcomes are highly variable — this project is more about **demonstrating ML techniques** than guaranteeing betting accuracy.

---

## 🛠️ Tech Stack

- **Python**: 3.9+
- **Libraries**:
  - `pandas`, `numpy` — data handling
  - `scikit-learn` — ML models & pipelines
  - `requests`, `beautifulsoup4` — web scraping
- **Version control**: Git + GitHub

---
