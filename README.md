# Task 3: Machine Learning - The "Baseline Beater"

## The Solution: Beating the Baseline

We overhauled the initial `baseline_model.ipynb` to dramatically improve performance. 

**Our Results:**
- **Baseline F1-Score:** 0.28283
- **Optimized F1-Score:** 0.61871 (a **118.7% improvement!**)

**The Three Key Changes:**
1. **Most Impactful Change (Feature Engineering):** We engineered new features that reflect true human behavior instead of raw numbers. The most impactful change was creating a `Total_Campaigns_Accepted` feature.
2. **Why it Worked Logically:** Human behavior is highly repetitive. If a customer accepted previous marketing campaigns, they are significantly more likely to accept the new one. The original model completely ignored this historical loyalty data. By aggregating past campaign acceptances into a single "loyalty score", we gave the model a powerful signal that immediately improved its ability to find the "Yes" customers.
3. **Other Major Fixes:** 
    - Fixed a critical data leakage/outlier bug where missing `Income` was originally filled with `$0`. We replaced this with Median Imputation.
    - Swapped the `LogisticRegression` for a tree-based `HistGradientBoostingClassifier` to automatically handle non-linear patterns and feature scaling.
    - Used `class_weight='balanced'` and optimized the decision threshold to `0.42` to combat the severe class imbalance (85% 'No' vs 15% 'Yes'), preventing the model from lazily guessing 'No'.

## The Scenario
An intern left behind a machine learning script (`baseline_model.ipynb`) that technically runs, but its performance is mediocre at best. Your job is to improve the model and, more importantly, explain the **math and logic** behind your improvements.

## The Objective
The current script predicts customer behavior (the `Response` column) but achieves a poor **F1-Score**. You must modify the script to improve the F1-Score by at least **20%** relative to the baseline.

## Requirements
1. **Improve the Model:** You may use any standard library (Scikit-learn, XGBoost, Pandas, LightGBM, etc.) to engineer features, handle missing data, or switch models.
2. **The Write-up (Mandatory):** At the very top of your final notebook, you must include a 3-bullet-point explanation answering:
   - What was the single most impactful code/model change you made?
   - *Why* did that specific change improve the model mathematically or logically based on the data?
   - What was your final F1-Score?

## Evaluation Criteria
AI can write code to upgrade a model. We are grading your **understanding**. If you cannot articulate the intuition behind your feature selection or model choice, it indicates a lack of ML fundamentals.

---

### 💡 Subtle Hints for the Wise
*   **The Scale Problem:** Notice the range of values in `Income` versus `Kidhome`. Do all models handle these differences equally well?
*   **The Hidden Gems:** The intern ignored all "Object" columns. Is there really no value in knowing a customer's `Education` or `Marital_Status`?
*   **New Perspectives:** `Year_Birth` is just a number. `Age` is a feature. Are there other ways to combine columns to tell a better story?
*   **The Silent Majority:** Look at the distribution of the `Response` variable. If 85% of people say "No," a model that just guesses "No" every time will be 85% accurate... but is it a *good* model?
