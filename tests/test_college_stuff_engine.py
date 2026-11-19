"""
Project Apex — Phase 2: End-to-End Pytest & CLI Verification Suite (`--synthetic-d1-test`).

Generates 25,000 realistic synthetic NCAA Division 1 TrackMan pitches across 200 college
pitchers with ground-truth swing, whiff, called-strike, and batted-ball xwOBAcon probabilities,
and verifies:
  1. TrackMan ingestion, 100-pitch pitcher filtering, and `delta_RV` target engineering
  2. Kinematic & fastball-differential feature engineering (`PerceivedSpeed`, `BauerUnits`,
     `VAA_diff`, `HAA_diff`, `ArmSlotAngle`, `VeloDiff_FB`, `IVBDiff_FB`, `HBDiff_FB`)
  3. 4-Head Multi-Task XGBoost Cascade training with strict `GroupKFold(n_splits=5)` by `PitcherId`
     and monotonic velocity constraints on fastballs
  4. 10x10 synthetic strike-zone grid location neutralization (`Overall`, `vs. RHB`, `vs. LHB`)
  5. NCAA D1 100-baseline Stuff+ sanity checks:
       - 83.0 mph RHP FF (15.4" IVB, 13.0" HB) scores in [65.0, 76.0] Stuff+
       - 95.0 mph RHP FF (18.5" IVB, 6.5' ext) scores >= 115.0 Stuff+
       - Monotonic velocity check: increasing FF velocity strictly increases Stuff+
  6. Browser JSON export (`data/processed/ncaa_d1_stuff_model.json`) & pure-JS inference
     parity (`web/stuff_model_inference.js`) within 0.01 Stuff+.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from src.models.college_stuff_engine import (
    CANONICAL_PITCH_FAMILIES,
    CollegeStuffEngine,
    EMPIRICAL_LOCATION_WEIGHTS_10X10,
    MODEL_FEATURE_COLUMNS,
    build_monotonic_constraints,
    compute_expected_vaa_and_haa,
    engineer_pitch_features,
    ingest_trackman_dataframe,
    ingest_trackman_directory,
    neutralize_location_xrv,
)
from src.models.export_for_web import (
    export_model_for_web,
    verify_python_vs_js_parity,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--synthetic-d1-test` CLI flag for pytest."""
    try:
        parser.addoption(
            "--synthetic-d1-test",
            action="store_true",
            default=False,
            help="Run the full 25,000-pitch synthetic NCAA D1 training and JS parity suite.",
        )
    except ValueError:
        pass


# =============================================================================
# 1. REALISTIC 25,000-PITCH SYNTHETIC NCAA D1 TRACKMAN GENERATOR
# =============================================================================

def generate_synthetic_ncaa_d1_dataset(
    n_pitches: int = 25_000,
    n_pitchers: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate `n_pitches` (default 25,000) realistic NCAA Division 1 in-game TrackMan pitches
    across `n_pitchers` (200 pitchers, ~125 pitches/pitcher) with ground-truth swing,
    called-strike, whiff, and batted-ball `ExitSpeed`/`Angle` outcomes governed by physical
    pitch kinematics, plate location, count leverage, and platoon splits.
    """
    rng = np.random.default_rng(seed)
    pitches_per_pitcher = max(105, int(np.ceil(n_pitches / n_pitchers)))
    total_rows = pitches_per_pitcher * n_pitchers

    pitcher_ids = np.repeat([f"NCAA_D1_P{i:03d}" for i in range(1, n_pitchers + 1)], pitches_per_pitcher)[:n_pitches]
    pitcher_is_lhp_map = rng.random(n_pitchers) < 0.28
    pitcher_fb_velo_map = np.clip(rng.normal(91.2, 2.85, size=n_pitchers), 82.0, 98.0)
    pitcher_fb_ivb_map = np.clip(rng.normal(15.8, 1.85, size=n_pitchers), 10.5, 21.0)
    pitcher_fb_hb_map = np.clip(rng.normal(8.5, 1.85, size=n_pitchers), 3.0, 14.0)
    pitcher_ext_map = np.clip(rng.normal(6.15, 0.22, size=n_pitchers), 5.4, 6.9)
    pitcher_rel_z_map = np.clip(rng.normal(5.85, 0.25, size=n_pitchers), 5.1, 6.6)
    pitcher_rel_x_map = np.clip(rng.normal(1.85, 0.25, size=n_pitchers), 1.1, 2.6)

    p_idx = np.repeat(np.arange(n_pitchers), pitches_per_pitcher)[:n_pitches]
    is_lhp = pitcher_is_lhp_map[p_idx]
    sign_hand = np.where(is_lhp, -1.0, 1.0)
    pitcher_throws = np.where(is_lhp, "Left", "Right")

    is_lhb = rng.random(n_pitches) < 0.42
    batter_side = np.where(is_lhb, "Left", "Right")
    same_hand = (is_lhp == is_lhb).astype(np.float64)

    # Pitch family usage distribution across NCAA D1
    family_choices = ["FF", "SI", "FC", "SL", "SW", "CU", "CH", "FS"]
    family_probs = [0.44, 0.10, 0.06, 0.16, 0.06, 0.07, 0.08, 0.03]
    pitch_families = rng.choice(family_choices, size=n_pitches, p=family_probs)

    fb_velo_base = pitcher_fb_velo_map[p_idx]
    fb_ivb_base = pitcher_fb_ivb_map[p_idx]
    fb_hb_base = pitcher_fb_hb_map[p_idx]

    rel_speed = np.empty(n_pitches, dtype=np.float64)
    spin_rate = np.empty(n_pitches, dtype=np.float64)
    spin_axis_norm = np.empty(n_pitches, dtype=np.float64)
    ivb = np.empty(n_pitches, dtype=np.float64)
    hb_norm = np.empty(n_pitches, dtype=np.float64)

    for fam in family_choices:
        mask = pitch_families == fam
        m_cnt = int(np.sum(mask))
        if m_cnt == 0:
            continue
        if fam == "FF":
            rel_speed[mask] = np.clip(fb_velo_base[mask] + rng.normal(0.0, 1.10, m_cnt), 81.0, 99.0)
            spin_rate[mask] = np.clip(rng.normal(2260.0, 165.0, m_cnt), 1750.0, 2850.0)
            spin_axis_norm[mask] = np.clip(rng.normal(208.0, 12.0, m_cnt), 175.0, 245.0)
            ivb[mask] = np.clip(fb_ivb_base[mask] + rng.normal(0.0, 1.05, m_cnt), 9.0, 22.5)
            hb_norm[mask] = np.clip(fb_hb_base[mask] + rng.normal(0.0, 1.05, m_cnt), 2.0, 15.5)
        elif fam == "SI":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 0.8 + rng.normal(0.0, 1.05, m_cnt), 80.0, 98.0)
            spin_rate[mask] = np.clip(rng.normal(2140.0, 160.0, m_cnt), 1650.0, 2650.0)
            spin_axis_norm[mask] = np.clip(rng.normal(232.0, 12.0, m_cnt), 200.0, 260.0)
            ivb[mask] = np.clip(rng.normal(8.5, 2.2, m_cnt), 1.0, 15.0)
            hb_norm[mask] = np.clip(rng.normal(15.2, 2.1, m_cnt), 8.5, 21.0)
        elif fam == "FC":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 4.8 + rng.normal(0.0, 1.1, m_cnt), 78.0, 93.5)
            spin_rate[mask] = np.clip(rng.normal(2380.0, 175.0, m_cnt), 1850.0, 2900.0)
            spin_axis_norm[mask] = np.clip(rng.normal(155.0, 15.0, m_cnt), 115.0, 185.0)
            ivb[mask] = np.clip(rng.normal(7.5, 2.1, m_cnt), 1.0, 13.5)
            hb_norm[mask] = np.clip(rng.normal(-3.2, 1.9, m_cnt), -9.0, 2.0)
        elif fam == "SL":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 9.8 + rng.normal(0.0, 1.3, m_cnt), 72.0, 89.0)
            spin_rate[mask] = np.clip(rng.normal(2460.0, 195.0, m_cnt), 1900.0, 3050.0)
            spin_axis_norm[mask] = np.clip(rng.normal(112.0, 18.0, m_cnt), 65.0, 155.0)
            ivb[mask] = np.clip(rng.normal(1.2, 2.4, m_cnt), -6.5, 7.5)
            hb_norm[mask] = np.clip(rng.normal(-6.5, 2.2, m_cnt), -11.8, -1.0)
        elif fam == "SW":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 11.2 + rng.normal(0.0, 1.3, m_cnt), 71.0, 86.5)
            spin_rate[mask] = np.clip(rng.normal(2610.0, 200.0, m_cnt), 2050.0, 3200.0)
            spin_axis_norm[mask] = np.clip(rng.normal(92.0, 14.0, m_cnt), 55.0, 125.0)
            ivb[mask] = np.clip(rng.normal(1.5, 2.3, m_cnt), -4.0, 7.5)
            hb_norm[mask] = np.clip(rng.normal(-14.8, 2.1, m_cnt), -21.0, -12.1)
        elif fam == "CU":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 14.0 + rng.normal(0.0, 1.5, m_cnt), 68.0, 84.5)
            spin_rate[mask] = np.clip(rng.normal(2540.0, 210.0, m_cnt), 1900.0, 3150.0)
            spin_axis_norm[mask] = np.clip(rng.normal(42.0, 16.0, m_cnt), 5.0, 85.0)
            ivb[mask] = np.clip(rng.normal(-11.5, 2.7, m_cnt), -19.5, -4.5)
            hb_norm[mask] = np.clip(rng.normal(-8.5, 2.4, m_cnt), -15.5, -2.0)
        elif fam == "CH":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 9.0 + rng.normal(0.0, 1.25, m_cnt), 72.0, 88.5)
            spin_rate[mask] = np.clip(rng.normal(1740.0, 180.0, m_cnt), 1200.0, 2300.0)
            spin_axis_norm[mask] = np.clip(rng.normal(238.0, 14.0, m_cnt), 200.0, 270.0)
            ivb[mask] = np.clip(rng.normal(6.2, 2.4, m_cnt), -1.0, 13.0)
            hb_norm[mask] = np.clip(rng.normal(14.2, 2.2, m_cnt), 7.5, 20.0)
        elif fam == "FS":
            rel_speed[mask] = np.clip(fb_velo_base[mask] - 7.8 + rng.normal(0.0, 1.25, m_cnt), 74.0, 90.0)
            spin_rate[mask] = np.clip(rng.normal(1380.0, 170.0, m_cnt), 850.0, 1900.0)
            spin_axis_norm[mask] = np.clip(rng.normal(225.0, 18.0, m_cnt), 180.0, 265.0)
            ivb[mask] = np.clip(rng.normal(2.8, 2.5, m_cnt), -4.5, 9.5)
            hb_norm[mask] = np.clip(rng.normal(11.8, 2.3, m_cnt), 5.0, 18.0)

    ext = np.clip(pitcher_ext_map[p_idx] + rng.normal(0.0, 0.08, n_pitches), 4.8, 7.2)
    rel_height = np.clip(pitcher_rel_z_map[p_idx] + rng.normal(0.0, 0.07, n_pitches), 4.5, 6.9)
    rel_side_norm = np.clip(pitcher_rel_x_map[p_idx] + rng.normal(0.0, 0.07, n_pitches), 0.8, 3.2)
    rel_side = rel_side_norm * sign_hand
    hb = hb_norm * sign_hand
    spin_axis = np.where(is_lhp, (360.0 - spin_axis_norm) % 360.0, spin_axis_norm)

    # Plate locations & count distribution
    plate_z = np.clip(rng.normal(2.48, 0.72, n_pitches), 0.4, 4.6)
    plate_x_norm = np.clip(rng.normal(0.0, 0.62, n_pitches), -2.1, 2.1)
    plate_x = plate_x_norm * sign_hand

    exp_vaa, exp_haa_norm = compute_expected_vaa_and_haa(
        rel_speed_mph=rel_speed,
        rel_height_ft=rel_height,
        rel_side_norm_ft=rel_side_norm,
        plate_loc_height_ft=plate_z,
        plate_loc_side_norm_ft=plate_x_norm,
    )
    # VAA residual driven by IVB above baseline
    vaa_diff = 0.085 * (ivb - 10.0) + rng.normal(0.0, 0.18, n_pitches)
    haa_diff = -0.04 * hb_norm + rng.normal(0.0, 0.15, n_pitches)
    vaa = exp_vaa + vaa_diff
    haa_norm = exp_haa_norm + haa_diff
    haa = haa_norm * sign_hand

    balls = rng.choice([0, 1, 2, 3], size=n_pitches, p=[0.38, 0.30, 0.20, 0.12])
    strikes = rng.choice([0, 1, 2], size=n_pitches, p=[0.42, 0.33, 0.25])
    count_lev = (strikes - balls) / 3.0

    zone_x_scaled = plate_x_norm / (17.0 / 24.0)
    zone_z_scaled = (plate_z - 2.50) / 1.05
    zone_dist = np.sqrt(zone_x_scaled ** 2 + zone_z_scaled ** 2)
    in_zone = ((np.abs(plate_x_norm) <= 0.83) & (plate_z >= 1.50) & (plate_z <= 3.50)).astype(np.float64)

    # Compute dimensionless physical pitch quality S_phys per pitch family
    s_phys = np.zeros(n_pitches, dtype=np.float64)
    ff_mask = pitch_families == "FF"
    s_phys[ff_mask] = (
        0.78 * ((rel_speed[ff_mask] - 91.2) / 3.05)
        + 0.30 * ((ivb[ff_mask] - 15.8) / 2.10)
        - 0.22 * ((hb_norm[ff_mask] - 8.5) / 2.10)
        + 0.20 * ((ext[ff_mask] - 6.15) / 0.25)
        + 0.22 * vaa_diff[ff_mask]
        + np.where((rel_speed[ff_mask] >= 93.5) & (ivb[ff_mask] >= 17.5), 0.45, 0.0)
    )
    other_mask = ~ff_mask
    s_phys[other_mask] = (
        0.55 * ((rel_speed[other_mask] - 82.5) / 4.5)
        + 0.35 * (np.abs(hb_norm[other_mask]) - 8.0) / 4.0
        + 0.25 * ((spin_rate[other_mask] - 2200.0) / 350.0)
        + 0.18 * (same_hand[other_mask] - 0.5)
    )

    # 1. Ground-Truth P(Swing)
    logit_swing = 1.15 - 1.95 * zone_dist + 0.38 * strikes - 0.22 * balls + 0.12 * s_phys
    p_swing_true = 1.0 / (1.0 + np.exp(-logit_swing))
    is_swing = rng.random(n_pitches) < p_swing_true

    # 2. Ground-Truth P(CalledStrike | Take)
    logit_cs = 3.20 - 4.10 * zone_dist + 0.15 * in_zone
    p_cs_true = 1.0 / (1.0 + np.exp(-logit_cs))
    is_cs_given_take = rng.random(n_pitches) < p_cs_true

    # 3. Ground-Truth P(Whiff | Swing) — strongly driven by s_phys (monotonically increasing with velocity)
    logit_whiff = -0.95 + 1.15 * s_phys + 0.35 * zone_dist + 0.15 * count_lev
    p_whiff_true = 1.0 / (1.0 + np.exp(-logit_whiff))
    is_whiff_given_swing = rng.random(n_pitches) < p_whiff_true

    # 4. Ground-Truth Contact ExitSpeed & Launch Angle — high s_phys suppresses ExitSpeed and barrels
    is_foul_given_contact = rng.random(n_pitches) < 0.38
    exit_speed = np.clip(91.5 - 6.2 * s_phys + rng.normal(0.0, 2.2, n_pitches), 58.0, 114.0)
    launch_angle = np.clip(20.0 - 7.5 * s_phys + rng.normal(0.0, 6.0, n_pitches), -25.0, 65.0)

    pitch_call = np.empty(n_pitches, dtype=object)
    play_result = np.full(n_pitches, "Undefined", dtype=object)
    kor_bb = np.full(n_pitches, "Undefined", dtype=object)

    take_cs_mask = (~is_swing) & is_cs_given_take
    take_ball_mask = (~is_swing) & (~is_cs_given_take)
    swing_whiff_mask = is_swing & is_whiff_given_swing
    swing_foul_mask = is_swing & (~is_whiff_given_swing) & is_foul_given_contact
    swing_bip_mask = is_swing & (~is_whiff_given_swing) & (~is_foul_given_contact)

    pitch_call[take_cs_mask] = "StrikeCalled"
    pitch_call[take_ball_mask] = "BallCalled"
    pitch_call[swing_whiff_mask] = "StrikeSwinging"
    pitch_call[swing_foul_mask] = "FoulBall"
    pitch_call[swing_bip_mask] = "InPlay"

    kor_bb[(take_cs_mask | swing_whiff_mask) & (strikes == 2)] = "Strikeout"
    kor_bb[take_ball_mask & (balls == 3)] = "Walk"
    play_result[swing_bip_mask] = np.where(exit_speed[swing_bip_mask] > 93.0, "Single", "Out")

    exit_speed_col = np.where(swing_bip_mask | swing_foul_mask, np.round(exit_speed, 2), np.nan)
    angle_col = np.where(swing_bip_mask | swing_foul_mask, np.round(launch_angle, 2), np.nan)

    return pd.DataFrame({
        "PitcherId": pitcher_ids,
        "PitcherThrows": pitcher_throws,
        "BatterSide": batter_side,
        "TaggedPitchType": pitch_families,
        "AutoPitchType": pitch_families,
        "PitchCall": pitch_call,
        "PlayResult": play_result,
        "KorBB": kor_bb,
        "Balls": balls,
        "Strikes": strikes,
        "RelSpeed": np.round(rel_speed, 2),
        "SpinRate": np.round(spin_rate, 1),
        "SpinAxis": np.round(spin_axis, 2),
        "RelHeight": np.round(rel_height, 3),
        "RelSide": np.round(rel_side, 3),
        "Extension": np.round(ext, 3),
        "InducedVertBreak": np.round(ivb, 2),
        "HorzBreak": np.round(hb, 2),
        "PlateLocHeight": np.round(plate_z, 3),
        "PlateLocSide": np.round(plate_x, 3),
        "VertApprAngle": np.round(vaa, 3),
        "HorzApprAngle": np.round(haa, 3),
        "ExitSpeed": exit_speed_col,
        "Angle": angle_col,
    })


# =============================================================================
# 2. PYTEST FIXTURES & END-TO-END VERIFICATION SUITE
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_d1_df() -> pd.DataFrame:
    """Generate the 25,000-pitch synthetic NCAA D1 dataset once per test module."""
    return generate_synthetic_ncaa_d1_dataset(n_pitches=25_000, n_pitchers=200, seed=42)


@pytest.fixture(scope="module")
def trained_d1_engine(synthetic_d1_df: pd.DataFrame) -> CollegeStuffEngine:
    """Train the 4-head XGBoost cascade with GroupKFold(n_splits=5) on the 25,000-pitch dataset."""
    engine = CollegeStuffEngine(
        n_estimators=85,
        max_depth=4,
        learning_rate=0.07,
        n_splits=5,
        random_state=42,
    )
    engine.fit(synthetic_d1_df, min_pitches_per_pitcher=100)
    return engine


def test_ingestion_and_feature_engineering(synthetic_d1_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify CSV directory ingestion, <100 pitch pitcher filtering, and engineered features."""
    # Add a low-sample scrub pitcher with only 15 pitches and verify it gets filtered out
    scrub_df = synthetic_d1_df.iloc[:15].copy()
    scrub_df["PitcherId"] = "SCRUB_LOW_SAMPLE_P999"
    combined_raw = pd.concat([synthetic_d1_df, scrub_df], ignore_index=True)

    csv_path = tmp_path / "ncaa_d1_sample.csv"
    combined_raw.to_csv(csv_path, index=False)

    df_ingested = ingest_trackman_directory(tmp_path, min_pitches_per_pitcher=100)
    assert "SCRUB_LOW_SAMPLE_P999" not in set(df_ingested["PitcherId"])
    assert len(df_ingested) == 25_000
    assert set(df_ingested["PitchFamily"]).issubset(set(CANONICAL_PITCH_FAMILIES))
    assert "delta_RV" in df_ingested.columns

    df_feat = engineer_pitch_features(df_ingested)
    for col in MODEL_FEATURE_COLUMNS:
        assert col in df_feat.columns, f"Missing engineered feature column: {col}"


def test_groupkfold_metrics_and_monotonic_constraints(trained_d1_engine: CollegeStuffEngine) -> None:
    """Verify GroupKFold(5) validation AUCs/RMSEs and fastball velocity monotonic constraints."""
    assert "FF" in trained_d1_engine.family_models
    ff_model = trained_d1_engine.family_models["FF"]

    # Out-of-fold discrimination checks across heads 1-3 and RMSE/R2 on head 4
    assert ff_model.metrics.swing_roc_auc > 0.72
    assert ff_model.metrics.whiff_roc_auc > 0.65
    assert ff_model.metrics.called_strike_roc_auc > 0.85
    assert ff_model.metrics.xwobacon_rmse < 0.25

    # Verify monotonic constraints on RelSpeed (index 0) and PerceivedSpeed (index 1)
    mc_whiff = build_monotonic_constraints("FF", "whiff")
    mc_xw = build_monotonic_constraints("FF", "xwobacon")
    assert mc_whiff[0] == 1 and mc_whiff[1] == 1
    assert mc_xw[0] == -1 and mc_xw[1] == -1


def test_10x10_location_neutralization_and_d1_stuff_sanity_gates(
    trained_d1_engine: CollegeStuffEngine,
) -> None:
    """
    Assert the two mandatory NCAA D1 scouting sanity checks:
      1. 83.0 mph RHP four-seamer (15.4" IVB, 13.0" HB) scores in [65.0, 76.0] Stuff+
      2. 95.0 mph RHP four-seamer (18.5" IVB, 6.5' ext) scores >= 115.0 Stuff+
      3. Monotonicity: 95.0 mph > 91.5 mph > 83.0 mph
    """
    assert np.isclose(np.sum(EMPIRICAL_LOCATION_WEIGHTS_10X10), 1.0)

    hs_83_ff = {
        "PitcherThrows": "RHP",
        "TaggedPitchType": "FF",
        "RelSpeed": 83.0,
        "SpinRate": 2120.0,
        "SpinAxis": 215.0,
        "RelHeight": 5.80,
        "RelSide": 1.85,
        "Extension": 6.00,
        "InducedVertBreak": 15.4,
        "HorzBreak": 13.0,
        "PlateLocHeight": 2.50,
        "PlateLocSide": 0.0,
        "VertApprAngle": -5.35,
        "HorzApprAngle": -1.45,
    }
    grade_83 = neutralize_location_xrv(trained_d1_engine, hs_83_ff)
    assert 65.0 <= grade_83.stuff_plus_overall <= 76.0, (
        f"Expected 83.0 mph RHP FF (15.4\" IVB, 13.0\" HB) in [65, 76] Stuff+, got {grade_83.stuff_plus_overall}"
    )

    d1_avg_ff = {
        "PitcherThrows": "RHP",
        "TaggedPitchType": "FF",
        "RelSpeed": 91.2,
        "SpinRate": 2260.0,
        "SpinAxis": 208.0,
        "RelHeight": 5.85,
        "RelSide": 1.85,
        "Extension": 6.15,
        "InducedVertBreak": 15.8,
        "HorzBreak": 8.5,
        "PlateLocHeight": 2.50,
        "PlateLocSide": 0.0,
        "VertApprAngle": -4.85,
        "HorzApprAngle": -1.25,
    }
    grade_avg = neutralize_location_xrv(trained_d1_engine, d1_avg_ff)
    assert 95.0 <= grade_avg.stuff_plus_overall <= 105.0

    elite_95_ff = {
        "PitcherThrows": "RHP",
        "TaggedPitchType": "FF",
        "RelSpeed": 95.0,
        "SpinRate": 2440.0,
        "SpinAxis": 202.0,
        "RelHeight": 5.80,
        "RelSide": 1.80,
        "Extension": 6.50,
        "InducedVertBreak": 18.5,
        "HorzBreak": 8.5,
        "PlateLocHeight": 2.50,
        "PlateLocSide": 0.0,
        "VertApprAngle": -4.15,
        "HorzApprAngle": -1.15,
    }
    grade_95 = neutralize_location_xrv(trained_d1_engine, elite_95_ff)
    assert grade_95.stuff_plus_overall >= 115.0, (
        f"Expected 95.0 mph RHP FF (18.5\" IVB, 6.5' ext) >= 115.0 Stuff+, got {grade_95.stuff_plus_overall}"
    )
    hs_77_ff = dict(hs_83_ff, RelSpeed=77.0, VertApprAngle=-5.85)
    grade_77 = neutralize_location_xrv(trained_d1_engine, hs_77_ff)
    assert 0.0 <= grade_77.grid_expected_xwobacon <= 1.0
    assert 0.0 <= grade_83.grid_expected_xwobacon <= 1.0
    assert 0.0 <= grade_95.grid_expected_xwobacon <= 1.0
    assert grade_77.stuff_plus_overall < grade_83.stuff_plus_overall - 15.0
    assert grade_77.stuff_plus_overall < grade_83.stuff_plus_overall < grade_avg.stuff_plus_overall < grade_95.stuff_plus_overall


def test_web_json_export_and_js_inference_parity(
    trained_d1_engine: CollegeStuffEngine,
    tmp_path: Path,
) -> None:
    """Verify JSON export and < 0.01 Stuff+ parity between Python xgboost and pure-JS tree walker."""
    json_out = tmp_path / "ncaa_d1_stuff_model.json"
    js_out = tmp_path / "stuff_model_inference.js"

    payload = export_model_for_web(trained_d1_engine, json_output_path=json_out, js_output_path=js_out)
    assert json_out.exists() and js_out.exists()
    assert payload["metadata"]["training_sample_size"] == 25_000
    assert "FF" in payload["college_baselines"]
    assert "FF" in payload["xgboost_trees"]

    test_pitches: List[Dict[str, object]] = [
        {
            "PitcherThrows": "RHP",
            "TaggedPitchType": "FF",
            "RelSpeed": 83.0,
            "SpinRate": 2120.0,
            "SpinAxis": 215.0,
            "RelHeight": 5.80,
            "RelSide": 1.85,
            "Extension": 6.00,
            "InducedVertBreak": 15.4,
            "HorzBreak": 13.0,
            "PlateLocHeight": 2.50,
            "PlateLocSide": 0.0,
            "VertApprAngle": -5.35,
            "HorzApprAngle": -1.45,
        },
        {
            "PitcherThrows": "RHP",
            "TaggedPitchType": "FF",
            "RelSpeed": 95.0,
            "SpinRate": 2440.0,
            "SpinAxis": 202.0,
            "RelHeight": 5.80,
            "RelSide": 1.80,
            "Extension": 6.50,
            "InducedVertBreak": 18.5,
            "HorzBreak": 8.5,
            "PlateLocHeight": 2.50,
            "PlateLocSide": 0.0,
            "VertApprAngle": -4.15,
            "HorzApprAngle": -1.15,
        },
        {
            "PitcherThrows": "LHP",
            "TaggedPitchType": "SL",
            "RelSpeed": 82.5,
            "SpinRate": 2580.0,
            "SpinAxis": 245.0,
            "RelHeight": 5.75,
            "RelSide": -1.90,
            "Extension": 6.10,
            "InducedVertBreak": 1.5,
            "HorzBreak": 7.8,
            "PlateLocHeight": 2.20,
            "PlateLocSide": 0.35,
            "VertApprAngle": -6.75,
            "HorzApprAngle": 1.60,
        },
    ]

    parity_report = verify_python_vs_js_parity(
        engine=trained_d1_engine,
        json_model_path=json_out,
        js_helper_path=js_out,
        test_pitches=test_pitches,
        tolerance_stuff_plus=0.01,
    )
    for item in parity_report:
        assert float(item["diff"]) <= 0.01
        if "python_vs_js_diff" in item:
            assert float(item["python_vs_js_diff"]) <= 0.01


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project Apex Phase 2 Synthetic NCAA D1 Verification")
    parser.add_argument("--synthetic-d1-test", action="store_true", help="Run 25,000-pitch synthetic D1 verification")
    args, unknown = parser.parse_known_args()

    sys.exit(pytest.main([__file__, "-v", "-s"] + (["--synthetic-d1-test"] if args.synthetic_d1_test else []) + unknown))
