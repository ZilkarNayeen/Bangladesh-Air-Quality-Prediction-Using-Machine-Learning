# Bangladesh Air Quality Forecasting Using Machine Learning

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)
![Framework](https://img.shields.io/badge/scikit--learn-latest-orange.svg?style=flat-square)
![Academic Level](https://img.shields.io/badge/Academic%20Research-BRAC%20University-red.svg?style=flat-square)

An advanced statistical and ensemble-based machine learning pipeline engineered to forecast the Air Quality Index (AQI) across Bangladesh utilizing high-volume historical environmental observations.

## Project Overview
Air pollution represents a severe systemic public health crisis in densely populated urban and industrial zones within Bangladesh. This repository contains the data preprocessing infrastructure, feature engineering workflows, and predictive regression model architectures optimized to forecast continuous AQI values. The models ingest complex particulate material metrics, gaseous pollutant patterns, and localized temporal markers to provide stable monitoring capabilities.

---

## Authors & Researchers
* **Hossain Md Nayeen Zilkar** – Department of Computer Science and Engineering, BRAC University — [GitHub Profile](https://github.com/ZilkarNayeen)
* **Ahnaf Shafique** – Department of Computer Science and Engineering, BRAC University
* **Salsabin Haque Simi** – Department of Computer Science and Engineering, BRAC University

---

## Architectural Workflow
The system processes large-scale data points through a strict multi-tier data science pipeline:
1. **Exploratory Data Analysis (EDA):** Deep analysis of pollutant distributions, feature correlations, and missing value trends.
2. **Data Cleaning & Preprocessing:** Targeting missing labels, performing strategic feature dropping, and executing dimensional encoding.
3. **Outlier Mitigation & Scaling:** Implementing Interquartile Range (IQR) bounding alongside statistical standard scaling.
4. **Ensemble Modeling & Evaluation:** Running parallel execution loops across baseline linear estimators and specialized tree-based regressors.

---

## Dataset & Preprocessing Pipeline
The pipeline operates on the *Bangladesh Air Quality Index (AQI) Dataset (2000-2025)*, analyzing historical hourly metrics spanning **103 cities**.

### Data Optimization Metrics
- **Initial Dataset Volume:** 1,048,551 rows × 13 attributes with 775,023 missing values.
- **Post-Cleaning Volume:** 893,517 rows × 12 clean attributes (0 missing values, 0 duplicates).
- **Target Variable Range Tuning:** Outlier treatments using the IQR bounds shifted maximum target values from 299.60 down to 268.00, reducing systematic noise while preserving the true statistical variance of moderate-to-unhealthy ranges.

### Engineering Steps Applied
* **Target Isolation:** Dropped all rows featuring missing `aqi` target labels.
* **Feature Pruning:** Stripped out `carbon_dioxide` due to an unacceptable missing rate exceeding 50% (774,327 missing points). Removed low-variance position bounds (`lat`, `lon`, `city_id`).
* **Temporal Discretization:** Extracted high-resolution time indicators (`year`, `month`, `day`, `hour`) from raw string timestamps.
* **Categorical Mapping:** Enforced deterministic integer mapping over spatial markers (`city_name`) via `LabelEncoder`.
* **Feature Uniformity:** Applied `StandardScaler` to align structural variance across features.
- **Validation Splitting:** Distributed processed pools into an 80% Training Set (714,813 records) and a 20% Testing Set (178,704 records).

---

## Model Evaluation & Performance Results
The predictive task was framed as a continuous regression problem. Five advanced tree-based ensemble variants were trained and thoroughly benchmarked against a baseline Linear Regression module using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).

### Ranked Performance Benchmark (Test Data)
| Model Architecture | Rank | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Extra Trees Regressor** | **1** | **6.83** | **10.74** | **0.948** |
| Random Forest Regressor | 2 | 7.12 | 11.21 | 0.943 |
| Histogram Gradient Boosting | 3 | 9.50 | 13.71 | 0.915 |
| Decision Tree Regressor | 4 | 9.25 | 14.67 | 0.903 |
| Gradient Boosting Regressor | 5 | 10.90 | 15.62 | 0.890 |
| Linear Regression (Baseline) | 6 | 15.43 | 19.91 | 0.821 |

### Key Experimental Discoveries
- **Ensemble Dominance:** The **Extra Trees Regressor** captured highly non-linear environmental interactions best, yielding an **RMSE of 10.74** and accounting for **94.8%** of total AQI variance.
- **Error Reduction:** Moving from the Linear Regression baseline to the Extra Trees model **decreased overall RMSE by 46.06%** and enhanced overall fit ($R^2$) by **15.42%**.
- **Feature Influence:** Structural analysis showed that particulate matter features were the strongest indicators, with `PM2.5` leading at an importance score of 0.459, followed by `PM10` at 0.253.

---

## Local Installation & Usage

### Prerequisites
Ensure your local Python ecosystem matches these target environments:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
