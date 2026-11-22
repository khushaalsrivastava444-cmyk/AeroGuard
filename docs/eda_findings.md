# Exploratory Data Analysis & Sensor Degradation Report

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)  
**Author:** AeroGuard Research & Applied ML Team  

---

## 1. Fleet Lifecycle & Operational Longevity

In the training set ($\mathcal{D}_{\text{train}}$), all 100 engines run continuously until functional failure ($RUL = 0$). In the test set ($\mathcal{D}_{\text{test}}$), operating trajectories are truncated at arbitrary time steps prior to failure.

### Summary Statistics of Trajectory Horizons ($T_i$)

| Statistic | Training Fleet ($T_i^{\text{fail}}$) | Test Fleet ($T_i^{\text{observed}}$) | Ground-Truth Test Target ($RUL_i^{\text{final}}$) |
| :--- | :--- | :--- | :--- |
| **Engine Count** | 100 | 100 | 100 |
| **Total Cycles** | 20,631 | 13,096 | — |
| **Minimum** | 128 cycles (Engine 39) | 31 cycles | 7 cycles |
| **25th Percentile ($Q_1$)** | 177 cycles | 74 cycles | 31 cycles |
| **Median** | 199 cycles (Engine 69) | 133 cycles | 86 cycles |
| **Mean ($\mu$)** | 206.31 cycles | 130.96 cycles | 75.08 cycles |
| **75th Percentile ($Q_3$)** | 229 cycles | 178 cycles | 114 cycles |
| **Maximum** | 362 cycles (Engine 69) | 303 cycles | 145 cycles |
| **Std Dev ($\sigma$)** | 46.34 cycles | 60.77 cycles | 41.76 cycles |

### Analytical Takeaways
1. **Variance in Lifespans:** Initial manufacturing tolerances and operating wear vary widely: the most resilient engine (Engine 69, 362 cycles) operates nearly $3\times$ longer than the weakest engine (Engine 39, 128 cycles).
2. **Failure Cycle as Input is Leakage:** Because $T_i$ is unknown at inference time, models must never receive $T_i$ or normalized life percentage $\tau = t / T_i$ as input features. They must infer degradation state purely from historical sensor measurements up to cycle $t$.

---

## 2. Sensor Invariance & Channel Pruning

FD001 operates under a single condition (Sea Level). Across all 20,631 training snapshots, $8$ of the $24$ operational/sensor channels exhibit zero or near-zero variance ($\sigma < 10^{-4}$):

| Channel Name | Description | Empirical Std ($\sigma$) | Status | Action in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `setting_3` | Altitude / Flight Condition | $0.0000$ | Invariant | **Drop** |
| `s_1` | Fan inlet temperature ($T_2$) | $0.0000$ | Invariant | **Drop** |
| `s_5` | Pressure at fan inlet ($P_2$) | $5.33 \times 10^{-15}$ | Invariant | **Drop** |
| `s_6` | Pressure in bypass-duct ($P_{15}$) | $0.0449$ | Quasi-constant (98% 21.61) | **Drop** |
| `s_10` | Engine pressure ratio ($P_{50}/P_2$) | $0.0000$ | Invariant | **Drop** |
| `s_16` | Static bypass pressure ($Ps_{15}$) | $3.47 \times 10^{-18}$ | Invariant | **Drop** |
| `s_18` | Demanded fan speed ($Nf_{\text{dmd}}$) | $0.0000$ | Invariant | **Drop** |
| `s_19` | Demanded corrected fan speed | $0.0000$ | Invariant | **Drop** |

### Mathematical Justification for Channel Exclusion
Retaining invariant channels creates singular or near-singular covariance matrices ($\det(\mathbf{X}^T\mathbf{X}) \approx 0$), causing instability in Ordinary Least Squares (OLS) and Ridge regression inverses ($(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}$). In tree-based models (Random Forest, XGBoost), invariant features consume split candidate evaluations without adding predictive information.

---

## 3. High-Pressure Compressor (HPC) Degradation Mechanics

In FD001, fault propagation is driven by HPC blade erosion and tip clearance growth. The 14 dynamic sensors respond in two distinct physical clusters:

### Cluster A: Thermal Degradation Indicators ($r_{\text{cycle}} > +0.64$)
As compressor efficiency degrades ($\eta_{\text{HPC}} \downarrow$), the closed-loop engine control unit injects additional fuel ($s_{13} \uparrow$) to maintain required thrust. This elevates gas-path temperatures and bleed flows:
- `s_2` ($T_{24}$, LPC outlet temp): $\bar{r} = +0.68$
- `s_3` ($T_{30}$, HPC outlet temp): $\bar{r} = +0.64$
- `s_4` ($T_{50}$, LPT outlet temp): $\bar{r} = +0.78$
- `s_11` (Bypass ratio, BPR): $\bar{r} = +0.81$
- `s_15` (LPT coolant bleed): $\bar{r} = +0.72$
- `s_17` (HPT coolant bleed): $\bar{r} = +0.67$

### Cluster B: Pneumatic & Dynamic Pressure Depletion ($r_{\text{cycle}} < -0.71$)
Lower compressor efficiency reduces pressure ratios across the core:
- `s_7` ($P_{30}$, HPC outlet total pressure): $\bar{r} = -0.76$
- `s_12` ($Ps_{30}$, HPC outlet static pressure): $\bar{r} = -0.79$
- `s_20` (HPT coolant bleed flow): $\bar{r} = -0.71$
- `s_21` (LPT coolant bleed flow): $\bar{r} = -0.72$

---

## 4. Non-Linear Degradation Phases & Target Capping

When sensor trajectories are normalized by lifetime fraction $\tau = t / T_i$, two distinct operational phases emerge:
1. **Nominal Stable Phase ($\tau \in [0\%, 65\%]$):** Sensors fluctuate around a healthy steady-state baseline governed by sensor noise. Degradation is negligible.
2. **Accelerated Failure Phase ($\tau > 65\%$):** Exponential deterioration toward failure.

### Implication for Supervised RUL Targets:
In a linear uncapped target formulation, an engine at cycle 10 with a 300-cycle lifespan has $RUL = 290$, while an engine at cycle 10 with a 150-cycle lifespan has $RUL = 140$. Yet physically, both engines are in identical brand-new, healthy states.

**Recommendation:** Implement **Piecewise Linear RUL Target Capping**:
$$RUL_{\text{capped}}(i, t) = \min(T_i - t, RUL_{\text{cap}})$$
In Milestone 3, we will systematically test candidate thresholds $RUL_{\text{cap}} \in [100, 115, 125, 140, \infty]$.
