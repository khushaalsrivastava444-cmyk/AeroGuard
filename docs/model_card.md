# AeroGuard Model Card: Champion XGBoost Regressor

## Model Overview
- **Model Name:** AeroGuard XGBoost Turbofan RUL Regressor
- **Architecture:** Gradient Boosted Decision Trees (`XGBRegressor`)
- **Version:** 1.0
- **Framework:** XGBoost 3.x, Scikit-learn 1.9, Python 3.12
- **Objective:** Supervised regression on Remaining Useful Life (RUL) with $L_2$ squared error loss and piecewise-linear target ceiling.

---

## Hyperparameter Specification
- `n_estimators`: 120
- `max_depth`: 5
- `learning_rate`: 0.05
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `reg_alpha`: 1.0 ($L_1$ regularization)
- `reg_lambda`: 5.0 ($L_2$ regularization)
- `random_state`: 42

---

## Input Features (32 Total)
- **Flight Condition Controls (2):** `setting_1`, `setting_2`
- **Dynamic Sensors (14):** `s_2`, `s_3`, `s_4`, `s_7`, `s_8`, `s_9`, `s_11`, `s_12`, `s_13`, `s_14`, `s_15`, `s_17`, `s_20`, `s_21`
- **Causal Rolling Means (window=5, 15):** e.g., `s_4_roll_mean_15`, `s_11_roll_mean_15`, `s_12_roll_mean_15`
- **Causal First Differences (window=5, 15):** e.g., `s_4_diff_15`, `s_11_diff_15`

Excluded: 8 invariant / near-zero variance channels (`setting_3`, $s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$).

---

## Quantitative Performance

### 5-Fold GroupKFold Cross-Validation (Across 100 Training Engines)
- **CV MAE:** $12.78 \pm 0.85$ cycles
- **CV RMSE:** $17.61 \pm 1.20$ cycles
- **CV $R^2$:** $0.821$

### Official NASA Test Fleet (100 Truncated Engines at Terminal Observed Cycle)
- **Official Test RMSE:** **17.86 cycles**
- **Official Test MAE:** **13.28 cycles**
- **Official Test $R^2$:** **0.815**
- **Official NASA Prognostics Score:** **678.7**

---

## Uncertainty & Conformal Calibration
- **Method:** Split Conformal Prediction
- **Nominal Confidence Level:** 90%
- **Empirical Test Fleet Coverage:** **85.0%**
- **Conformal Margin $\pm q$:** $\pm 26.21$ cycles
- **Mean Interval Width:** $50.5$ cycles
