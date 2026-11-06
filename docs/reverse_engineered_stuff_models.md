# Granular Reverse-Engineering of the 5 Public Stuff+ Architectures

This technical reference documents the mathematical formulations, feature matrices, and training targets reverse-engineered from the 5 leading public and open-source `Stuff+` systems and adapted for **Project Apex (v3.5)** (`High School TrackMan B1 -> NCAA Division 1`).

## 1. FanGraphs Stuff+ (Eno Sarris & Max Bay)
* **Algorithm**: LightGBM / XGBoost Gradient Boosted Decision Trees trained on pitch-level run value delta (`delta_run_exp`) and Called Strikes + Whiffs (`CSW%`).
* **Primary Feature Vector**: `RelSpeed`, `InducedVertBreak`, `HorzBreak`, `Extension`, `SpinRate`, `SpinAxis`, `RelHeight`, `RelSide`, `VertApprAngle (VAA)`, and Fastball-Relative Deltas (`dVelo`, `dIVB`, `dHB`).
* **Secondary Baseline Strategy**: Identifies each pitcher's primary fastball (`FF` or `SI` by maximum usage) and computes all offspeed features as differentials from that fastball anchor.

## 2. PitchingBot / botStf (Cameron Grove)
* **Algorithm**: Multi-class CatBoost / GBDT predicting contextual outcome probabilities (`P(Take)`, `P(CalledStrike | Take)`, `P(SwStr | Swing)`, `P(Foul | Swing)`, `E[xwOBAcon | BIP]`).
* **Primary Feature Vector**: Uses raw 3D acceleration components (`ax`, `az`) rather than raw inches of break so slower curveballs are not artificially rewarded purely for staying in the air longer.

## 3. Baseball Prospectus StuffPro
* **Algorithm**: Hierarchical two-stage partitioned GBDT (Stage 1: Take Run Value; Stage 2: Swing Run Value).
* **Location-Adjusted Approach Angles (`PVAA` / `PHAA`)**: Residuals of Vertical and Horizontal Approach Angles regressed against `PlateLocHeight`, `PlateLocSide`, `RelHeight`, and `RelSide`:
  $$\text{PVAA} = \text{VAA} - \mathbb{E}[\text{VAA} \mid \text{PlateLocHeight}, \text{RelHeight}]$$

## 4. Tread Athletics & Driveline College TrackMan Stuff+
* **Algorithm**: NCAA Division 1 TrackMan-calibrated XGBoost + Empirical Bayes bullpen shrinkage.
* **Key High-School-to-D1 Calibration Laws**:
  - **Non-Linear Sub-88 mph Fastball Velocity Cliff**:
    $$\text{Penalty}_{\text{cliff}} = \begin{cases} -0.45 \cdot (88.0 - v_{\text{FB}})^{1.6} & \text{if } v_{\text{FB}} < 88.0\text{ mph} \\ 0 & \text{otherwise} \end{cases}$$
  - **Fastball Anchor Viability Gate ($\gamma_{\text{anchor}}$)**:
    $$\gamma_{\text{anchor}} = \text{clamp}\left(\frac{v_{\text{FB}} - 78.0}{12.0},\; 0.15,\; 1.00\right)$$

## 5. tjStuff+ Open-Source Pipeline (Thomas Nestico)
* **Algorithm**: Open-source Python LightGBM regressor (`LightGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=63)`) trained on `delta_run_exp`.
* **Canonical Handedness Normalization**: Mirrors all LHP horizontal coordinates (`RelSide`, `HorzBreak`, `vx0`, `ax`, `SpinAxis`) into a canonical RHP frame prior to feature engineering.
