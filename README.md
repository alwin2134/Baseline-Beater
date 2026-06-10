# Baseline Beater: Marketing Campaign Prediction

## Overview
This project focuses on predicting customer responses to marketing campaigns. We started with a basic Logistic Regression baseline and completely overhauled it with a **Stacking Ensemble** of LightGBM + XGBoost + HistGradientBoosting, powered by Optuna hyperparameter tuning and 46 engineered features — achieving a **113.2% improvement** in F1-Score.

## Our Results

| | Score |
|---|---|
| 🔴 **Intern Baseline F1** (LogReg, raw features) | `0.30108` |
| 🟢 **Best Model F1** (Stacking Ensemble) | **`0.64198`** |
| 📈 **Improvement** | **+113.2%** |
| 🎯 **ROC-AUC** | `0.92179` |

## The Four Key Changes

1. **Most Impactful Change — Feature Engineering:**
   We created 46 features from the original raw columns. The single most powerful feature was `Total_Cmp_Accepted` — the sum of all past campaign acceptances. Human behavior is repetitive: customers who said "Yes" before are far more likely to say "Yes" again. The intern's model completely ignored this signal.

2. **Why It Worked Logically:**
   Raw columns like `Income` and `Year_Birth` tell the model very little on their own. But combining them into `Income_per_Member` (purchasing power per household person), `Spending_x_Loyalty` (total spend × campaign loyalty), and `Web_Purchase_Rate` (conversion efficiency) gives the model a true picture of customer behaviour.

3. **Stacking Ensemble + Optuna Tuning:**
   Instead of a single model, we combined three expert models (LightGBM, XGBoost, HistGradientBoosting) into a **Stacking Ensemble**, where a meta-learner learns how to best combine their predictions. LightGBM's hyperparameters were auto-tuned using **Optuna Bayesian optimisation** over 50 trials to maximise CV F1-Score.

4. **Class Imbalance + OOF Threshold Tuning:**
   85% of customers say "No" — a lazy model just guesses "No" every time and looks 85% accurate but finds zero real buyers. We used `class_weight='balanced'` and tuned the decision threshold using **Out-of-Fold (OOF) probabilities** — the most statistically robust method — to maximise F1 without touching the test set.

## How to Run
1. Install dependencies: `pip install pandas scikit-learn numpy matplotlib seaborn xgboost lightgbm optuna`
2. Run `baseline_model.ipynb` from top to bottom — all scores are computed live, nothing is hardcoded.
