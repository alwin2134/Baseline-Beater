# Baseline Beater: Marketing Campaign Prediction

## Overview
This project focuses on predicting customer responses to marketing campaigns. We started with a baseline machine learning script that had mediocre performance (F1-Score: 0.28) and completely overhauled the feature engineering, data imputation, and modeling approach to dramatically improve its predictive power.

## Our Results
- **Baseline F1-Score:** 0.28283
- **Optimized F1-Score:** 0.61871 (a **118.7% improvement!**)

## The Three Key Changes

1. **Most Impactful Change (Feature Engineering):** 
   We engineered new features that reflect true human behavior instead of just looking at raw numbers. The absolute most impactful change was creating a `Total_Campaigns_Accepted` feature.
   
2. **Why it Worked Logically:** 
   Human behavior is highly repetitive. If a customer accepted previous marketing campaigns, they are significantly more likely to accept a new one. The original model completely ignored this historical loyalty data. By aggregating past campaign acceptances into a single "loyalty score", we gave the model a powerful signal that immediately improved its ability to accurately identify the "Yes" customers.

3. **Other Major Fixes:** 
    - **Data Leakage & Outliers:** Fixed a critical bug where missing `Income` values were originally filled with `$0`. We replaced this with Median Imputation to prevent skewing the data.
    - **Model Upgrade:** Swapped the simple `LogisticRegression` for a tree-based `HistGradientBoostingClassifier`, which automatically handles non-linear patterns, diverse feature scales, and missing values gracefully.
    - **Handling Class Imbalance:** Used `class_weight='balanced'` and optimized the decision threshold to `0.42` to combat the severe class imbalance (85% 'No' vs 15% 'Yes'). This forced the model to actively look for the rare positive cases instead of lazily guessing 'No' for everyone.

## How to Run
1. Ensure you have the required libraries installed (`pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`).
2. Run the `baseline_model.ipynb` Jupyter Notebook from top to bottom.
