# AeroGuard: System Limitations & Operational Scope

## 1. Simulated vs. Physical Reality
AeroGuard is trained and evaluated on the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. While C-MAPSS relies on high-fidelity thermodynamic modeling developed by NASA Ames, it remains a synthetic simulation:
- Sensor noise is modeled as independent Gaussian perturbations. Real aircraft sensors frequently experience drift, sensor bias, intermittent connectivity, and non-Gaussian spikes.
- Component degradation is driven by pre-programmed damage propagation equations. Real turbofans encounter unpredictable external foreign object damage (FOD), volcanic ash exposure, and varying maintenance overhauls.

---

## 2. Single Operating Condition Restriction (FD001)
FD001 simulates engines operating strictly at **Sea Level nominal flight** with a single failure mode: High-Pressure Compressor (HPC) degradation.
- **Limitation:** The current models do not account for variable throttle profiles, climbing/descending flight envelopes, or multi-condition operations present in FD002 and FD004.
- Invariant sensor channels (`setting_3`, $s_1, s_5, s_{10}$, etc.) are specific to FD001. In multi-condition datasets, these channels become dynamic and require condition-based normalization.

---

## 3. Decision-Support Heuristics vs. Certified Probabilities
- Risk classifications (e.g., **Critical**: $\text{RUL} \le 30$, **Warning**: $31 \le \text{RUL} \le 90$, **Healthy**: $\text{RUL} > 90$) are rule-based engineering heuristics intended for maintenance work-order scheduling.
- They must not be construed as FAA/EASA certified safety-critical airworthiness clearances.
- Point predictions carry uncertainty; maintenance decisions must incorporate the calibrated Conformal Prediction interval bounds rather than relying solely on point estimates.

---

## 4. Anomaly Detection Interpretation
Isolation Forest anomaly scores measure geometric distance and isolation in the 32-dimensional feature space. 
- While anomalous scores correlate with late-stage degradation, an anomaly score is **not** a calibrated failure probability.
- Anomalies may be triggered by harmless transient sensor noise or unusual operational settings without impending structural failure.
