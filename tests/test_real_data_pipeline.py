"""
Unit tests for Phase 2B real-data ingestion and college-anchored calibration pipeline.

Verifies:
1. Baseball Savant (Catcher's feet) -> TrackMan (Pitcher's inches) coordinate conversion
   for both RHP and LHP (`InducedVertBreak = pfx_z * 12`, `HorzBreak = -pfx_x * 12` for
   both RHP and LHP, so LHP FF `pfx_x = +0.85` ft -> `HorzBreak = -10.2` in and
   `HB_norm = +10.2` in after `engineer_pitch_features()`).
2. TrackMan V3 team whitelist filtering (`wcl_teams` including Marion Berries drops
   NWAC/CCBC exhibition opponents on both `PitcherTeam` and `BatterTeam`), optional
   `divisions` whitelist, and `min_pitches_per_pitcher = 30`.
3. Statcast MiLB FSL filtering by `college_draftee_ids` (MLBAM IDs) and Aug 1 - Sept 30 window.
4. MLB Stats API Draft helper (`fetch_mlb_draft_college_ids`) keeps 4-year college and JUCO
   picks while excluding high school / prep picks.
5. College-only `mu_college` / `sigma_college` anchoring (`cohort == 'college'`) when merged
   with an extreme `fsl_draftee` cohort.
6. Candidate JSON export (`ncaa_d1_stuff_model_candidate.json`) schema parity with live
   `ncaa_d1_stuff_model.json` and guardrail blocking accidental live model overwrites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.real_data_pipeline import (
    DEFAULT_WCL_TEAMS,
    convert_statcast_movement_to_trackman,
    fetch_mlb_draft_college_ids,
    load_statcast_milb_dataset,
    load_trackman_v3_dataset,
    train_college_anchored_candidate_model,
)
from src.models.college_stuff_engine import (
    batch_neutralize_location_xrv,
    engineer_pitch_features,
    ingest_trackman_dataframe,
)


def _make_synthetic_trackman_df(
    pitcher_ids: List[str],
    pitches_per_pitcher: int = 35,
    team_name: str = "Victoria HarbourCats",
    batter_team: str = "Bellingham Bells",
    division: str = "D1",
    velo_base: float = 90.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    calls = ["StrikeCalled", "StrikeSwinging", "FoulBall", "InPlay", "BallCalled"]
    for idx, pid in enumerate(pitcher_ids):
        throws = "Left" if idx % 2 == 1 else "Right"
        sign = -1.0 if throws == "Left" else 1.0
        for p in range(pitches_per_pitcher):
            is_ff = p % 3 != 0
            pt = "Fastball" if is_ff else "Slider"
            velo = (velo_base if is_ff else velo_base - 9.0) + rng.normal(0, 1.2)
            ivb = (16.5 if is_ff else 1.0) + rng.normal(0, 1.5)
            hb = (9.5 if is_ff else -6.5) * sign + rng.normal(0, 1.5)
            call = calls[p % len(calls)]
            rows.append(
                {
                    "PitcherId": pid,
                    "Pitcher": f"Pitcher {pid}",
                    "PitcherTeam": team_name,
                    "BatterTeam": batter_team,
                    "Division": division,
                    "Date": "2026-07-12",
                    "PitcherThrows": throws,
                    "BatterSide": "Left" if p % 2 == 0 else "Right",
                    "TaggedPitchType": pt,
                    "AutoPitchType": pt,
                    "PitchCall": call,
                    "PlayResult": "Out" if call == "InPlay" else "Undefined",
                    "KorBB": "Undefined",
                    "Balls": p % 4,
                    "Strikes": p % 3,
                    "RelSpeed": velo,
                    "SpinRate": (2280 if is_ff else 2480) + rng.normal(0, 80),
                    "SpinAxis": 210.0 if is_ff else 90.0,
                    "InducedVertBreak": ivb,
                    "HorzBreak": hb,
                    "RelHeight": 5.85 + rng.normal(0, 0.12),
                    "RelSide": 1.85 * sign + rng.normal(0, 0.1),
                    "Extension": 6.20 + rng.normal(0, 0.15),
                    "PlateLocHeight": 2.50 + rng.normal(0, 0.65),
                    "PlateLocSide": rng.normal(0, 0.55),
                    "VertApprAngle": -4.8 + rng.normal(0, 0.6),
                    "HorzApprAngle": -1.2 * sign + rng.normal(0, 0.4),
                    "ExitSpeed": 88.0 + rng.normal(0, 6.0) if call == "InPlay" else np.nan,
                    "Angle": 14.0 + rng.normal(0, 12.0) if call == "InPlay" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _make_synthetic_statcast_df(
    pitcher_ids: List[int],
    pitches_per_pitcher: int = 35,
    game_date: str = "2026-08-22",
    velo_base: float = 96.5,
    seed: int = 99,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    descs = ["called_strike", "swinging_strike", "foul", "hit_into_play", "ball"]
    for idx, pid in enumerate(pitcher_ids):
        p_throws = "L" if idx % 2 == 1 else "R"
        for p in range(pitches_per_pitcher):
            is_ff = p % 3 != 0
            desc = descs[p % len(descs)]
            rows.append(
                {
                    "pitcher": pid,
                    "player_name": f"Draftee {pid}",
                    "game_date": game_date,
                    "p_throws": p_throws,
                    "stand": "L" if p % 2 == 0 else "R",
                    "pitch_type": "FF" if is_ff else "SL",
                    "description": desc,
                    "events": "field_out" if desc == "hit_into_play" else "",
                    "balls": p % 4,
                    "strikes": p % 3,
                    "release_speed": (velo_base if is_ff else velo_base - 9.5) + rng.normal(0, 1.0),
                    "release_spin_rate": (2450 if is_ff else 2650) + rng.normal(0, 70),
                    "spin_axis": 215.0 if is_ff else 85.0,
                    "pfx_x": (-0.85 if p_throws == "R" else 0.85) if is_ff else (0.55 if p_throws == "R" else -0.55),
                    "pfx_z": 1.50 if is_ff else 0.10,
                    "release_pos_z": 5.95 + rng.normal(0, 0.1),
                    "release_pos_x": -1.85 if p_throws == "R" else 1.85,
                    "release_extension": 6.45 + rng.normal(0, 0.12),
                    "plate_z": 2.55 + rng.normal(0, 0.55),
                    "plate_x": rng.normal(0, 0.50),
                    "launch_speed": 84.0 if desc == "hit_into_play" else np.nan,
                    "launch_angle": 12.0 if desc == "hit_into_play" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_statcast_to_trackman_coordinate_conversion() -> None:
    """
    Verify Savant Catcher's feet -> TrackMan Pitcher's inches sign convention:
      - HorzBreak = -pfx_x * 12.0 for BOTH RHP and LHP
      - LHP FF (pfx_x = +0.85 ft) -> HorzBreak = -10.2 in -> HB_norm = +10.2 in
    """
    ivb, hb = convert_statcast_movement_to_trackman(
        pfx_x_ft=np.array([-0.85, 0.85, 0.50, -0.50]),
        pfx_z_ft=np.array([1.50, 1.25, -0.50, 0.0]),
        p_throws=np.array(["R", "L", "R", "L"]),
    )
    np.testing.assert_allclose(ivb, [18.0, 15.0, -6.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(hb, [10.2, -10.2, -6.0, 6.0], atol=1e-6)

    # End-to-end check on a synthetic LHP Savant FF: HorzBreak = -10.2 in, HB_norm = +10.2 in
    lhp_raw = _make_synthetic_statcast_df([710001, 710002], pitches_per_pitcher=6, game_date="2026-08-20")
    lhp_only = lhp_raw.loc[lhp_raw["p_throws"] == "L"].copy()
    loaded = load_statcast_milb_dataset(lhp_only, college_draftee_ids={710002}, min_pitches_per_pitcher=1)
    ff_loaded = loaded.loc[loaded["TaggedPitchType"] == "FF"]
    np.testing.assert_allclose(ff_loaded["HorzBreak"].to_numpy(), -10.2, atol=1e-6)

    clean = ingest_trackman_dataframe(loaded, min_pitches_per_pitcher=1)
    feat = engineer_pitch_features(clean)
    ff_feat = feat.loc[feat["PitchFamily"] == "FF"]
    np.testing.assert_allclose(ff_feat["HB_norm"].to_numpy(), 10.2, atol=1e-6)


def test_trackman_v3_loader_filters_teams_divisions_and_min_pitches() -> None:
    """
    Verify wcl_teams whitelist includes Marion Berries, drops NWAC/CCBC opponents on
    both PitcherTeam and BatterTeam, and enforces min_pitches_per_pitcher = 30.
    """
    df_wcl_keep = _make_synthetic_trackman_df(
        ["WCL_1"], pitches_per_pitcher=35, team_name="Marion Berries", batter_team="Victoria HarbourCats", division="D1"
    )
    df_wcl_vs_exhibition = _make_synthetic_trackman_df(
        ["WCL_VS_EXH"], pitches_per_pitcher=35, team_name="Victoria HarbourCats", batter_team="NWAC Exhibition All-Stars", division="D1"
    )
    df_wcl_small = _make_synthetic_trackman_df(
        ["WCL_SMALL"], pitches_per_pitcher=15, team_name="Bellingham Bells", batter_team="Marion Berries", division="D1"
    )
    df_nwac_drop = _make_synthetic_trackman_df(
        ["NWAC_1"], pitches_per_pitcher=40, team_name="NWAC Exhibition All-Stars", batter_team="Victoria HarbourCats", division="NWAC"
    )
    df_ccbc_drop = _make_synthetic_trackman_df(
        ["CCBC_1"], pitches_per_pitcher=40, team_name="CCBC Selects", batter_team="Victoria HarbourCats", division="CCBC"
    )
    combined = pd.concat([df_wcl_keep, df_wcl_vs_exhibition, df_wcl_small, df_nwac_drop, df_ccbc_drop], ignore_index=True)

    # Default filter_batter_team=True drops exhibition batting opponent too
    loaded = load_trackman_v3_dataset(
        combined,
        wcl_teams=DEFAULT_WCL_TEAMS,
        filter_batter_team=True,
        divisions=["D1"],
        min_pitches_per_pitcher=30,
    )
    assert set(loaded["PitcherId"].unique()) == {"WCL_1"}
    assert (loaded["cohort"] == "college").all()

    # With filter_batter_team=False, WCL pitcher facing exhibition batters is retained
    loaded_no_bat = load_trackman_v3_dataset(
        combined,
        wcl_teams=DEFAULT_WCL_TEAMS,
        filter_batter_team=False,
        divisions=["D1"],
        min_pitches_per_pitcher=30,
    )
    assert set(loaded_no_bat["PitcherId"].unique()) == {"WCL_1", "WCL_VS_EXH"}


def test_fetch_mlb_draft_college_ids_excludes_hs_picks() -> None:
    """Verify fetch_mlb_draft_college_ids keeps 4YR/JC picks and excludes HS picks."""
    mock_payload = {
        "drafts": {
            "rounds": [
                {
                    "round": "1",
                    "picks": [
                        {
                            "person": {"id": 801001},
                            "school": {"name": "Oregon State", "schoolClass": "4YR"},
                        },
                        {
                            "person": {"id": 801002},
                            "school": {"name": "Central Arizona College", "schoolClass": "JC"},
                        },
                        {
                            "person": {"id": 801003},
                            "school": {"name": "IMG Academy", "schoolClass": "HS"},
                        },
                        {
                            "person": {"id": 801004},
                            "school": {"name": "Lincoln High School", "schoolClass": ""},
                        },
                    ],
                }
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        ids = fetch_mlb_draft_college_ids([2025])
        assert mock_open.call_count == 1
        assert ids == {801001, 801002}


def test_statcast_milb_loader_filters_draftees_and_aug_sept_window() -> None:
    """Verify college_draftee_ids MLBAM filter and Aug 1 - Sept 30 date window."""
    df_valid = _make_synthetic_statcast_df([700001], pitches_per_pitcher=12, game_date="2026-08-15")
    df_valid_sep = _make_synthetic_statcast_df([700002], pitches_per_pitcher=12, game_date="2026-09-28")
    df_early_july = _make_synthetic_statcast_df([700001], pitches_per_pitcher=12, game_date="2026-07-10")
    df_non_draftee = _make_synthetic_statcast_df([999999], pitches_per_pitcher=12, game_date="2026-08-20")
    raw = pd.concat([df_valid, df_valid_sep, df_early_july, df_non_draftee], ignore_index=True)

    loaded = load_statcast_milb_dataset(raw, college_draftee_ids={700001, 700002}, min_pitches_per_pitcher=1)
    assert set(loaded["PitcherId"].unique()) == {"MLBAM_700001", "MLBAM_700002"}
    assert len(loaded) == 24
    assert (loaded["cohort"] == "fsl_draftee").all()
    # Check converted IVB = 1.50 * 12 = 18.0 on FF rows
    ff_rows = loaded.loc[loaded["TaggedPitchType"] == "FF"]
    np.testing.assert_allclose(ff_rows["InducedVertBreak"].to_numpy(), 18.0, atol=1e-6)


def test_college_anchored_training_and_candidate_json_schema_parity(tmp_path: Path) -> None:
    """
    Verify:
    1. Live model overwrite is blocked by guardrail.
    2. mu_college and sigma_college are computed strictly on cohort == 'college'.
    3. Exported candidate JSON matches live ncaa_d1_stuff_model.json top-level & metadata keys.
    """
    college_raw = _make_synthetic_trackman_df(
        ["COL_1", "COL_2", "COL_3", "COL_4"], pitches_per_pitcher=40, velo_base=89.0, seed=11
    )
    college_df = load_trackman_v3_dataset(college_raw, min_pitches_per_pitcher=30)

    fsl_raw = _make_synthetic_statcast_df(
        [800001, 800002], pitches_per_pitcher=40, game_date="2026-08-25", velo_base=98.0, seed=22
    )
    fsl_df = load_statcast_milb_dataset(fsl_raw, college_draftee_ids={800001, 800002}, min_pitches_per_pitcher=30)

    # Guardrail: refusing to overwrite live production model
    with pytest.raises(ValueError, match="Refusing to overwrite live production model"):
        train_college_anchored_candidate_model(
            college_df=college_df,
            fsl_draftee_df=fsl_df,
            candidate_json_path=tmp_path / "ncaa_d1_stuff_model.json",
            min_pitches_per_pitcher=30,
            n_estimators=6,
            max_depth=3,
            n_splits=2,
        )

    cand_path = tmp_path / "ncaa_d1_stuff_model_candidate.json"
    engine, payload = train_college_anchored_candidate_model(
        college_df=college_df,
        fsl_draftee_df=fsl_df,
        candidate_json_path=cand_path,
        min_pitches_per_pitcher=30,
        n_estimators=6,
        max_depth=3,
        n_splits=2,
    )
    assert cand_path.exists()

    # Verify mu_college / sigma_college match strictly college-only rows (not 98 mph FSL draftees)
    col_clean = ingest_trackman_dataframe(college_df, min_pitches_per_pitcher=30)
    col_feat = engineer_pitch_features(col_clean)
    ff_col = col_feat.loc[col_feat["PitchFamily"] == "FF"]
    ff_model = engine.family_models["FF"]
    expected_xrv = batch_neutralize_location_xrv(
        ff_model.head_swing, ff_model.head_whiff, ff_model.head_called_strike, ff_model.head_xwobacon, ff_col
    )
    assert ff_model.baseline_stats.count == len(ff_col)
    assert abs(ff_model.baseline_stats.mu_college_xrv - float(np.nanmean(expected_xrv))) < 1e-5
    assert ff_model.baseline_stats.velo_mean < 92.0  # College mean (~89 mph), NOT pulled up by 98 mph FSL

    # Verify schema parity with live data/processed/ncaa_d1_stuff_model.json
    live_path = Path("data/processed/ncaa_d1_stuff_model.json")
    if live_path.exists():
        live_payload = json.loads(live_path.read_text(encoding="utf-8"))
        assert set(payload.keys()) == set(live_payload.keys())
        assert set(payload["metadata"].keys()) == set(live_payload["metadata"].keys())
        first_fam = next(iter(payload["college_baselines"].keys()))
        assert set(payload["college_baselines"][first_fam].keys()) == set(live_payload["college_baselines"]["FF"].keys())
        assert set(payload["xgboost_trees"][first_fam].keys()) == set(live_payload["xgboost_trees"]["FF"].keys())
