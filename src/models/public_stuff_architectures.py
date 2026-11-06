"""Reverse-Engineered Reference Implementations of the 5 Public Stuff+ Architectures."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PublicModelArchitectureSpec:
    model_id: str
    name: str
    authors_or_org: str
    ml_algorithm: str
    target_variable: str
    primary_features: List[str]
    secondary_baseline_strategy: str
    key_engineering_takeaway: str


PUBLIC_STUFF_ARCHITECTURES: Dict[str, PublicModelArchitectureSpec] = {
    "fangraphs_stuff_plus": PublicModelArchitectureSpec(
        model_id="fangraphs_stuff_plus",
        name="FanGraphs Stuff+ (Eno Sarris & Max Bay)",
        authors_or_org="FanGraphs / Max Bay & Eno Sarris",
        ml_algorithm="LightGBM / XGBoost Gradient Boosted Decision Trees",
        target_variable="Pitch-level expected run value delta (delta_run_exp) & CSW%",
        primary_features=[
            "RelSpeed", "InducedVertBreak", "HorzBreak", "Extension",
            "SpinRate", "SpinAxis", "RelHeight", "RelSide", "VertApprAngle",
            "velo_diff_from_fb", "ivb_diff_from_fb", "hb_diff_from_fb"
        ],
        secondary_baseline_strategy="Primary fastball (FF or SI by max usage) weighted average sets reference frame for all offspeed deltas.",
        key_engineering_takeaway="VAA adjusted for plate height and fastball-relative movement differentials drive >30% of secondary tree splits.",
    ),
    "pitchingbot_botstf": PublicModelArchitectureSpec(
        model_id="pitchingbot_botstf",
        name="PitchingBot / botStf (Cameron Grove)",
        authors_or_org="Cameron Grove / FanGraphs",
        ml_algorithm="Multi-Class CatBoost / Gradient Boosted Trees over Take/Swing Outcome Probabilities",
        target_variable="Expected Run Value synthesized from P(SwStr), P(Foul), P(BIP -> xwOBAcon), P(CalledStrike), P(Ball)",
        primary_features=[
            "RelSpeed", "AccelerationX", "AccelerationZ", "ReleaseExtension",
            "RelHeight", "RelSide", "ArmAngleProxy", "CountContext", "PlatoonHandedness"
        ],
        secondary_baseline_strategy="Acceleration vectors (ax, az) normalized by flight time squared; explicit batter-handedness interaction trees.",
        key_engineering_takeaway="Using acceleration vectors (or flight-time-normalized break) prevents slow curveballs from receiving artificial break credit.",
    ),
    "bp_stuffpro": PublicModelArchitectureSpec(
        model_id="bp_stuffpro",
        name="Baseball Prospectus StuffPro",
        authors_or_org="Baseball Prospectus Research Team",
        ml_algorithm="Hierarchical Two-Stage Partitioned GBDT (Take Sub-Model + Swing Sub-Model)",
        target_variable="Deserved Run Value per 100 pitches partitioned by batter swing decision",
        primary_features=[
            "RelSpeed", "PVAA (Location-Adjusted VAA)", "PHAA (Location-Adjusted HAA)",
            "InducedVertBreak", "HorzBreak", "SpinEfficiency", "SlotDeviationFromShoulderArc"
        ],
        secondary_baseline_strategy="De-trends VAA and HAA against PlateLocHeight and PlateLocSide so pitches aren't rewarded merely for high/outside location.",
        key_engineering_takeaway="PVAA = VAA - E[VAA | PlateLocHeight, RelHeight] isolates true plane deception from target height.",
    ),
    "tread_driveline_college": PublicModelArchitectureSpec(
        model_id="tread_driveline_college",
        name="Tread Athletics & Driveline College TrackMan Stuff+",
        authors_or_org="Tread Athletics / Driveline Baseball R&D",
        ml_algorithm="XGBoost + Empirical Bayes Shrinkage with Stadium Altitude & B1 Tilt Correction",
        target_variable="NCAA D1 In-Zone Whiff% and Chase% mapped to 100-mean / 10-SD scale",
        primary_features=[
            "RelSpeed", "AltitudeNormalizedIVB", "AltitudeNormalizedHB",
            "SpinEfficiency", "GyroDegree", "EffectiveVelocity", "FastballAnchorViability"
        ],
        secondary_baseline_strategy="Applies a non-linear sub-88 mph fastball velocity cliff and discounts secondary grades when fastball lacks D1 anchor viability.",
        key_engineering_takeaway="High school bullpens require sea-level air density normalization and a fastball anchor gate to prevent 'batting practice fastball' inflation.",
    ),
    "tjstuff_plus_open": PublicModelArchitectureSpec(
        model_id="tjstuff_plus_open",
        name="tjStuff+ Open-Source Pipeline (Thomas Nestico)",
        authors_or_org="Thomas Nestico (@tjstats)",
        ml_algorithm="LightGBM Regressor on Statcast / AAA TrackMan delta_run_exp with Leave-One-Season-Out CV",
        target_variable="Expected Run Value (xRV) -> Z-Score -> 100 + 10 * Z",
        primary_features=[
            "release_speed", "pfx_z", "pfx_x", "release_extension",
            "release_pos_x", "release_pos_z", "spin_axis", "release_spin_rate",
            "vy0", "vx0", "vz0", "ax", "ay", "az", "delta_velo_primary", "delta_ivb_primary", "delta_hb_primary"
        ],
        secondary_baseline_strategy=" mirror left-handed horizontal coordinates (rel_side, hb, spin_axis) to RHP canonical frame before feature engineering.",
        key_engineering_takeaway="Mirroring LHP horizontal signs into a canonical RHP coordinate frame doubles training sample efficiency and stabilizes tree splits.",
    ),
}
