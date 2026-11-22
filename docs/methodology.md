# AeroGuard: Mathematical & Experimental Methodology

**Document Version:** 1.0  
**Scope:** NASA C-MAPSS FD001 Turbofan Engine Degradation Simulation  

---

## 1. Remaining Useful Life (RUL) Target Formulation

### 1.1 Complete Training Trajectories
Let engine $i \in \{1, \dots, N_{\text{fleet}}\}$ operate over discrete cycles $t \in \{1, \dots, T_i\}$, where $T_i$ denotes the terminal cycle at functional failure.
The raw uncapped Remaining Useful Life at cycle $t$ is formulated as:
$$RUL(i, t) = T_i - t$$

At terminal cycle $t = T_i$, $RUL = 0$. At inception $t = 1$, $RUL = T_i - 1$.

### 1.2 Piecewise-Linear Target Capping
Empirical inspection of turbofan degradation demonstrates that engines exhibit stationary, mean-reverting sensor profiles during early operating life ($\tau < 65\%$ of lifecycle). In this early phase, an engine that will survive 300 cycles is indistinguishable in its physical degradation state from an engine that will survive 150 cycles. 

Attempting to fit an uncapped linear target forces models to learn arbitrary noise patterns in healthy regimes, drastically destabilizing gradient updates. We therefore apply a piecewise-linear target ceiling:
$$RUL_{\text{capped}}(i, t) = \min(T_i - t, RUL_{\text{cap}})$$

Through sensitivity ablation across $RUL_{\text{cap}} \in [100, 115, 125, 140, \infty]$, we observed that $RUL_{\text{cap}} = 115-125$ cycles optimizes the NASA Asymmetric Score on held-out test data.

---

## 2. Temporal Feature Engineering

All transformations are computed strictly within each engine group and rely exclusively on historical observations (no future information leakage).

Let $x_{t,j}$ denote the measurement of sensor $j$ at operational cycle $t$ for engine $i$:

### 2.1 Causal Rolling Mean
Captures the underlying smoothed degradation trend by attenuating high-frequency sensor noise:
$$\mu_{t,w}(x_j) = \frac{1}{\min(t, w)} \sum_{k=0}^{\min(t, w) - 1} x_{t-k, j}$$
Using `min_periods=1` prevents missing values at the beginning of an engine trajectory while maintaining strict causal boundaries.

### 2.2 Causal Rolling Standard Deviation
Quantifies the increasing variance and dynamic instability associated with mechanical blade flutter and clearance wear:
$$\sigma_{t,w}(x_j) = \sqrt{\frac{1}{\min(t, w) - 1} \sum_{k=0}^{\min(t, w) - 1} (x_{t-k, j} - \mu_{t,w}(x_j))^2} \quad (\text{for } t \ge 2)$$
For $t = 1$, $\sigma_{1,w} = 0.0$.

### 2.3 First Difference / Degradation Velocity
Approximates the instantaneous time-derivative ($\frac{dx}{dt}$) to capture the velocity of failure progression:
$$\Delta_w x_{t,j} = \begin{cases} x_{t,j} - x_{t-w, j} & \text{if } t > w \\ x_{t,j} - x_{1,j} & \text{if } t \le w \end{cases}$$

---

## 3. Data Leakage Prevention & Validation Protocol

A critical flaw in naive time-series ML implementations is random $K$-fold cross-validation, which splits adjacent cycles of the same engine between training and validation folds. Because $x_t$ and $x_{t+1}$ have high autocorrelation, random splitting produces artificially inflated validation performance that fails to generalize.

### 3.1 Engine-Level GroupKFold
We enforce complete fleet isolation using `GroupKFold` ($K = 5$) on `engine_id`:
$$\mathcal{E}_{\text{val}}^{(k)} \cap \mathcal{E}_{\text{train}}^{(k)} = \emptyset \quad \forall k \in \{1, \dots, 5\}$$
All 20 validation engines in fold $k$ are completely withheld from model fitting.

### 3.2 Preprocessing Hygiene
Standardization ($\mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$) is fitted strictly on the training partition of each fold and applied downstream to validation/test folds using learned training statistics.

### 3.3 Test Protocol Simulation
The official NASA test set (`test_FD001.txt`) consists of truncated sequences where the evaluation objective is to predict RUL at the **terminal observed cycle**. When evaluating validation folds, we extract the terminal cycle of held-out engines to benchmark against the exact operational task.

---

## 4. Evaluation Mathematics

### 4.1 Standard Metrics
- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### 4.2 NASA Asymmetric Prognostics Metric
In aerospace maintenance, late predictions ($\hat{y} > y$) are far more hazardous than early predictions ($\hat{y} < y$). Predicting that an engine has 30 cycles remaining when it has only 10 risks catastrophic in-flight failure, whereas predicting 10 cycles when it has 30 merely causes an early ground inspection.

Let signed residual $d_i = \hat{y}_i - y_i$. The total NASA score is:
$$S = \sum_{i=1}^N s_i, \quad s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \quad (\text{Early}) \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \quad (\text{Late}) \end{cases}$$

---

## 5. Conformal Uncertainty Estimation

We implement **Split Conformal Prediction** to construct valid $1 - \alpha$ prediction intervals without distributional assumptions:
1. Partition training fleet into proper training engines $\mathcal{E}_{\text{proper}}$ (80%) and calibration engines $\mathcal{E}_{\text{cal}}$ (20%).
2. Train model on $\mathcal{E}_{\text{proper}}$ and compute non-conformity residuals on calibration engines:
   $$R_i = |y_i - \hat{y}_i| \quad \forall i \in \mathcal{E}_{\text{cal}}$$
3. For significance level $\alpha = 0.10$ (90% nominal coverage), compute the finite-sample corrected empirical quantile:
   $$\hat{q} = \text{Quantile}\left(\{R_i\}_{i=1}^n, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$
4. The prediction interval for test instance $x$ is:
   $$C(x) = [\max(0, \hat{y} - \hat{q}), \hat{y} + \hat{q}]$$
