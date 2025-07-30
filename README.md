# 📊 Market Sentiment and Trading Behavior Analysis

This project explores how trader performance and strategy align (or diverge) from market sentiment using two real-world datasets — a **Fear and Greed Index** and **trading data from Hyperliquid**. The primary goal is to build a classification model to predict whether a trade results in profit or loss, while also uncovering patterns through unsupervised learning.

---

## 📁 Datasets

### 1. **Fear and Greed Index**
- Captures public market sentiment over time.
- Fields:
  - `Timestamp`, `Value` (0–100 scale), `Classification` (e.g., Fear, Greed), `Date`.

### 2. **Historical Trade Data (Hyperliquid)**
- Contains detailed transactional logs of user trades.
- Fields include:
  - `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Closed PnL`, `Direction`, and more.

---

## 🎯 Objective

- Analyze the relationship between **market sentiment** and **trading behavior**.
- Discover clusters or latent patterns in trader profiles.
- Build a **binary classification model** to predict trade outcomes:
  - `1`: Profit  
  - `0`: Loss

---

## 🔧 Methodology

### 🧼 Data Preprocessing
- Cleaned inconsistent timestamps.
- Merged datasets using normalized date formats.
- Handled missing values and engineered features like:
  - `Trade Return %`
  - `Sentiment Value`
  - `Status` (Profit or Loss)

### ⚖️ Class Imbalance Handling
- Dataset was skewed toward one class (Profitable trades).
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.

### 📊 Exploratory Data Analysis
- Time-series trends of sentiment vs trading activity.
- Correlation analysis using **Pearson** and **Spearman** metrics.
- Clustering with **KMeans** after **Standard Scaling** and **PCA**.

### 📉 Dimensionality Reduction (PCA)
- Reduced feature space to key principal components.
- Visualized variance and explained feature contributions.

### 🔍 Clustering (KMeans)
- Identified distinct trader behavior clusters.
- Used **Elbow method** to choose optimal cluster count (k).

### 🧠 Classification Modeling
- Models: Logistic Regression, Random Forest, and XGBoost.
- Target: `Status` (0 = Loss, 1 = Profit)
- **Bayesian Optimization** used for hyperparameter tuning.
- Evaluation metrics:
  - Accuracy, F1-Score, ROC-AUC, Precision/Recall

---

## 🏁 Results

- Weak correlation found between market sentiment and PnL, suggesting traders often act contrary to emotional indicators.
- PCA showed strong dimensional compression (few features hold most variance).
- KMeans revealed strategic clusters of traders (risk-averse vs aggressive).
- Final classification model showed improved recall for the minority class (Losses) after applying SMOTE and Bayesian tuning.
