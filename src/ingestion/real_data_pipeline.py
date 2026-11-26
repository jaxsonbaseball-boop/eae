"""
Phase 2B Real-Data Ingestion & College-Anchored Calibration Pipeline.

Prepares the 4-head XGBoost Stuff+ pipeline for real-world TrackMan V3 CSVs
(West Coast League + Mendeley Track_Combo.csv) and Aug-Sept Single-A Florida
State League (FSL) Baseball Savant Statcast CSVs.

Key Guarantees:
1. TrackMan V3 loader normalizes column aliases, filters by `wcl_teams` whitelist
   on both `PitcherTeam` and (optionally) `BatterTeam` (excluding non-league
   NWAC/CCBC exhibition opponents), optional `divisions` whitelist, and
   `min_pitches_per_pitcher = 30`.
2. Statcast MiLB loader converts Baseball Savant Catcher's-perspective feet
   (`pfx_x`, `pfx_z`) to TrackMan Pitcher's-perspective inches:
     - `InducedVertBreak = pfx_z * 12.0`
     - `HorzBreak = -pfx_x * 12.0` for BOTH RHP and LHP (since downstream
       `engineer_pitch_features()` multiplies LHP `HorzBreak * -1` into `HB_norm`)
   and filters strictly by `college_draftee_ids` (MLBAM IDs, from CSV or
   `fetch_mlb_draft_college_ids`) and Aug 1 - Sept 30.
3. College-anchored training merges `cohort == 'college'` and `cohort == 'fsl_draftee'`
   to fit the 4-head XGBoost cascade, computes `100 = College Avg` (`mu_college`,
   `sigma_college`) STRICTLY on `cohort == 'college'`, and exports to
   `data/processed/ncaa_d1_stuff_model_candidate.json` without overwriting the
   live production `data/processed/ncaa_d1_stuff_model.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.models.college_stuff_engine import (
    CANONICAL_PITCH_FAMILIES,
    CollegeStuffEngine,
    PitchFamilyBaselineStats,
    batch_neutralize_location_xrv,
    compute_expected_vaa_and_haa,
    engineer_pitch_features,
    ingest_trackman_dataframe,
)
from src.models.export_for_web import export_model_for_web

logger = logging.getLogger(__name__)

# Official West Coast League (WCL) member clubs (canonical names & common TrackMan codes)
DEFAULT_WCL_TEAMS: Set[str] = {
    "bellingham bells", "bel_bel", "bellingham",
    "bend elks", "ben_elk", "bend",
    "corvallis knights", "cor_kni", "corvallis",
    "cowlitz black bears", "cow_bla", "cowlitz",
    "edmonton riverhawks", "edm_riv", "edmonton",
    "kamloops northpaws", "kam_nor", "kamloops",
    "kelowna falcons", "kel_fal", "kelowna",
    "marion berries", "mar_ber", "marion",
    "nanaimo nightowls", "nan_nig", "nanaimo",
    "port angeles lefties", "por_lef", "port angeles",
    "portland pickles", "pdx_pic", "portland",
    "ridgefield raptors", "rid_rap", "ridgefield",
    "springfield drifters", "spr_dri", "springfield",
    "victoria harbourcats", "vic_har", "victoria",
    "walla walla sweets", "wal_swe", "walla walla",
    "wenatchee applesox", "wen_app", "wenatchee",
    "yakima valley pippins", "yak_pip", "yakima",
}

LIVE_MODEL_FILENAME = "ncaa_d1_stuff_model.json"
DEFAULT_CANDIDATE_MODEL_PATH = Path("data/processed/ncaa_d1_stuff_model_candidate.json")
MLB_DRAFT_API_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/draft/{year}"

TRACKMAN_V3_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "PitcherId": ("PitcherId", "Pitcher_Id", "pitcher_id", "PitcherID"),
    "Pitcher": ("Pitcher", "PitcherName", "pitcher_name"),
    "PitcherTeam": ("PitcherTeam", "Pitcher_Team", "Team", "team"),
    "BatterTeam": ("BatterTeam", "Batter_Team", "batter_team"),
    "Division": ("Division", "Level", "League", "Conference"),
    "Date": ("Date", "GameDate", "game_date", "SessionDate"),
    "PitcherThrows": ("PitcherThrows", "Throws", "p_throws", "PitcherHand"),
    "BatterSide": ("BatterSide", "BatterHand", "stand"),
    "TaggedPitchType": ("TaggedPitchType", "PitchType", "pitch_type"),
    "AutoPitchType": ("AutoPitchType", "auto_pitch_type"),
    "PitchCall": ("PitchCall", "pitch_call"),
    "PlayResult": ("PlayResult", "play_result"),
    "KorBB": ("KorBB", "kor_bb"),
    "Balls": ("Balls", "balls"),
    "Strikes": ("Strikes", "strikes"),
    "RelSpeed": ("RelSpeed", "ReleaseSpeed", "release_speed", "Velo"),
    "SpinRate": ("SpinRate", "release_spin_rate", "TotalSpin"),
    "SpinAxis": ("SpinAxis", "spin_axis"),
    "InducedVertBreak": ("InducedVertBreak", "IVB", "InducedVerticalBreak"),
    "HorzBreak": ("HorzBreak", "HB", "HorizontalBreak"),
    "RelHeight": ("RelHeight", "ReleaseHeight", "release_pos_z"),
    "RelSide": ("RelSide", "ReleaseSide", "release_pos_x"),
    "Extension": ("Extension", "release_extension", "RelExtension"),
    "PlateLocHeight": ("PlateLocHeight", "PlateZ", "plate_z"),
    "PlateLocSide": ("PlateLocSide", "PlateX", "plate_x"),
    "VertApprAngle": ("VertApprAngle", "VAA"),
    "HorzApprAngle": ("HorzApprAngle", "HAA"),
    "ExitSpeed": ("ExitSpeed", "launch_speed"),
    "Angle": ("Angle", "LaunchAngle", "launch_angle"),
}

STATCAST_DESCRIPTION_TO_PITCHCALL: Dict[str, str] = {
    "called_strike": "StrikeCalled",
    "swinging_strike": "StrikeSwinging",
    "swinging_strike_blocked": "StrikeSwinging",
    "foul_tip": "StrikeSwinging",
    "missed_bunt": "StrikeSwinging",
    "foul": "FoulBall",
    "foul_bunt": "FoulBall",
    "bunt_foul_tip": "FoulBall",
    "hit_into_play": "InPlay",
    "ball": "BallCalled",
    "blocked_ball": "BallCalled",
    "pitchout": "BallCalled",
    "hit_by_pitch": "HitByPitch",
}

STATCAST_EVENTS_TO_PLAYRESULT: Dict[str, str] = {
    "single": "Single",
    "double": "Double",
    "triple": "Triple",
    "home_run": "HomeRun",
    "field_out": "Out",
    "force_out": "Out",
    "grounded_into_double_play": "Out",
    "double_play": "Out",
    "fielders_choice_out": "Out",
    "sac_fly": "Sacrifice",
    "sac_bunt": "Sacrifice",
    "field_error": "Error",
}


def _norm_token(val: Any) -> str:
    return " ".join(str(val or "").strip().lower().split())


def _normalize_throws_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.upper()
    return pd.Series(
        np.where(cleaned.str.startswith("L"), "LHP", "RHP"),
        index=series.index,
        dtype="object",
    )


def _normalize_batter_side_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.upper()
    return pd.Series(
        np.where(cleaned.str.startswith("L"), "Left", "Right"),
        index=series.index,
        dtype="object",
    )


def _read_csv_inputs(source: Union[str, Path, Sequence[Union[str, Path]], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    paths: List[Path] = []
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            paths = sorted(p.rglob("*.csv"))
        elif p.is_file():
            paths = [p]
        else:
            raise FileNotFoundError(f"CSV source path does not exist: {p}")
    else:
        for item in source:
            p = Path(item)
            if p.is_dir():
                paths.extend(sorted(p.rglob("*.csv")))
            elif p.is_file():
                paths.append(p)
            else:
                raise FileNotFoundError(f"CSV source path does not exist: {p}")

    if not paths:
        raise FileNotFoundError("No CSV files found in the provided source path(s).")

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _canonicalize_trackman_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    existing_lower = {c.lower().strip(): c for c in out.columns}
    rename_map: Dict[str, str] = {}
    for canonical, aliases in TRACKMAN_V3_COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            match = existing_lower.get(alias.lower())
            if match is not None and match not in rename_map:
                rename_map[match] = canonical
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def fetch_mlb_draft_college_ids(
    years: Iterable[int],
    timeout_sec: float = 20.0,
) -> Set[int]:
    """
    Query the official MLB Stats API Draft endpoint (`statsapi.mlb.com/api/v1/draft/{year}`)
    and collect MLBAM `person.id` integers for drafted 4-year college and JUCO players,
    excluding high school / prep picks.
    """
    hs_classes = {"hs", "high school", "highschool", "prep"}
    college_ids: Set[int] = set()

    for year in years:
        url = MLB_DRAFT_API_URL_TEMPLATE.format(year=int(year))
        req = urllib.request.Request(url, headers={"User-Agent": "EAE-Phase2B-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        rounds = payload.get("drafts", {}).get("rounds", [])
        for round_obj in rounds:
            for pick in round_obj.get("picks", []):
                person = pick.get("person") or {}
                pid = person.get("id")
                if pid is None:
                    continue

                school = pick.get("school") or {}
                school_class = _norm_token(school.get("schoolClass") or school.get("class") or "")
                school_name = _norm_token(school.get("name") or "")

                if school_class in hs_classes:
                    continue
                if not school_class and (
                    "high school" in school_name
                    or school_name.endswith(" hs")
                    or " prep" in school_name
                ):
                    continue

                college_ids.add(int(pid))

    return college_ids


def load_trackman_v3_dataset(
    source: Union[str, Path, Sequence[Union[str, Path]], pd.DataFrame],
    wcl_teams: Optional[Collection[str]] = None,
    filter_batter_team: bool = True,
    divisions: Optional[Collection[str]] = None,
    min_pitches_per_pitcher: int = 30,
    cohort_label: str = "college",
    source_tag: str = "trackman_v3",
) -> pd.DataFrame:
    """
    Load and normalize TrackMan V3 CSV data (WCL or Mendeley Track_Combo.csv).

    Supports:
    - `wcl_teams` whitelist filtering on `PitcherTeam` and (when `filter_batter_team=True`
      and `BatterTeam` is present) on `BatterTeam`, dropping non-league NWAC/CCBC opponents
    - optional `divisions` whitelist filtering on `Division`
    - minimum pitcher sample threshold (`min_pitches_per_pitcher = 30`)
    """
    df = _canonicalize_trackman_columns(_read_csv_inputs(source))
    if df.empty:
        return df

    # Synthesize PitcherId from Pitcher name if PitcherId is absent
    if "PitcherId" not in df.columns:
        if "Pitcher" in df.columns:
            df["PitcherId"] = df["Pitcher"].astype(str).str.strip()
        else:
            raise ValueError("TrackMan dataset must contain either 'PitcherId' or 'Pitcher'.")

    # Filter by WCL team whitelist when provided
    if wcl_teams is not None and len(wcl_teams) > 0:
        allowed_teams = {_norm_token(t) for t in wcl_teams}
        if "PitcherTeam" not in df.columns:
            raise ValueError("wcl_teams whitelist provided, but 'PitcherTeam' column is missing.")
        keep_mask = df["PitcherTeam"].map(_norm_token).isin(allowed_teams)
        if filter_batter_team and "BatterTeam" in df.columns:
            keep_mask &= df["BatterTeam"].map(_norm_token).isin(allowed_teams)
        df = df.loc[keep_mask].copy()

    # Optional division whitelist filtering
    if divisions is not None and len(divisions) > 0:
        allowed_divs = {_norm_token(d) for d in divisions}
        if "Division" in df.columns:
            div_norm = df["Division"].map(_norm_token)
            df = df.loc[div_norm.isin(allowed_divs)].copy()

    if df.empty:
        return df

    # Normalize core categorical columns
    df["PitcherThrows"] = _normalize_throws_series(
        df["PitcherThrows"] if "PitcherThrows" in df.columns else pd.Series("RHP", index=df.index)
    )
    df["BatterSide"] = _normalize_batter_side_series(
        df["BatterSide"] if "BatterSide" in df.columns else pd.Series("Right", index=df.index)
    )

    if "AutoPitchType" not in df.columns:
        df["AutoPitchType"] = df.get("TaggedPitchType", "Fastball")
    if "TaggedPitchType" not in df.columns:
        df["TaggedPitchType"] = df["AutoPitchType"]

    for opt_col, default_val in (
        ("PitchCall", "BallCalled"),
        ("PlayResult", "Undefined"),
        ("KorBB", "Undefined"),
        ("Balls", 0),
        ("Strikes", 0),
        ("ExitSpeed", np.nan),
        ("Angle", np.nan),
        ("SpinAxis", 210.0),
    ):
        if opt_col not in df.columns:
            df[opt_col] = default_val

    numeric_cols = [
        "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
        "RelHeight", "RelSide", "Extension", "PlateLocHeight", "PlateLocSide",
        "VertApprAngle", "HorzApprAngle", "Balls", "Strikes", "ExitSpeed", "Angle",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing core kinematics
    core_required = ["PitcherId", "RelSpeed", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide"]
    df = df.dropna(subset=[c for c in core_required if c in df.columns]).copy()

    # Compute expected VAA/HAA when missing
    if "VertApprAngle" not in df.columns or df["VertApprAngle"].isna().any():
        is_lhp = df["PitcherThrows"].eq("LHP")
        sign_hand = np.where(is_lhp, -1.0, 1.0)
        exp_vaa, exp_haa = compute_expected_vaa_and_haa(
            rel_speed_mph=df["RelSpeed"].to_numpy(dtype=np.float64),
            rel_height_ft=df["RelHeight"].to_numpy(dtype=np.float64),
            rel_side_norm_ft=df["RelSide"].abs().to_numpy(dtype=np.float64),
            plate_loc_height_ft=df["PlateLocHeight"].fillna(2.50).to_numpy(dtype=np.float64),
            plate_loc_side_norm_ft=(df["PlateLocSide"].fillna(0.0).to_numpy(dtype=np.float64) * sign_hand),
        )
        if "VertApprAngle" not in df.columns:
            df["VertApprAngle"] = exp_vaa
        else:
            df["VertApprAngle"] = df["VertApprAngle"].fillna(pd.Series(exp_vaa, index=df.index))
        if "HorzApprAngle" not in df.columns:
            df["HorzApprAngle"] = exp_haa * sign_hand
        else:
            df["HorzApprAngle"] = df["HorzApprAngle"].fillna(pd.Series(exp_haa * sign_hand, index=df.index))

    # Enforce minimum pitches per pitcher
    if min_pitches_per_pitcher > 1:
        counts = df.groupby("PitcherId")["RelSpeed"].transform("size")
        df = df.loc[counts >= min_pitches_per_pitcher].copy()

    df["cohort"] = cohort_label
    df["data_source"] = source_tag
    return df.reset_index(drop=True)


def convert_statcast_movement_to_trackman(
    pfx_x_ft: Union[pd.Series, np.ndarray],
    pfx_z_ft: Union[pd.Series, np.ndarray],
    p_throws: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Baseball Savant movement (Catcher's perspective, feet; +pfx_x = 1B side)
    into TrackMan Pitcher's perspective (inches; +HorzBreak = 3B side / pitcher's right)
    for BOTH RHP and LHP:
      - InducedVertBreak = pfx_z * 12.0
      - HorzBreak = -pfx_x * 12.0 (both RHP and LHP; downstream `engineer_pitch_features()`
        multiplies LHP `HorzBreak * -1` into `HB_norm`)
    """
    del p_throws  # Perspective flip from Catcher to Pitcher is identical for RHP and LHP
    px = np.asarray(pfx_x_ft, dtype=np.float64)
    pz = np.asarray(pfx_z_ft, dtype=np.float64)

    ivb_in = pz * 12.0
    hb_in = -px * 12.0
    return ivb_in, hb_in


def load_statcast_milb_dataset(
    source: Union[str, Path, Sequence[Union[str, Path]], pd.DataFrame],
    college_draftee_ids: Collection[Union[int, str]],
    start_month_day: Tuple[int, int] = (8, 1),
    end_month_day: Tuple[int, int] = (9, 30),
    min_pitches_per_pitcher: int = 1,
    cohort_label: str = "fsl_draftee",
    source_tag: str = "statcast_milb_fsl",
) -> pd.DataFrame:
    """
    Load Aug-Sept Single-A Florida State League (FSL) Baseball Savant CSV data,
    filter to `college_draftee_ids` (MLBAM IDs) and the Aug 1 - Sept 30 date window,
    and convert Savant columns into TrackMan V3 Pitcher's Perspective schema.
    """
    raw = _read_csv_inputs(source)
    if raw.empty:
        return raw

    if "pitcher" not in raw.columns:
        raise ValueError("Statcast MiLB dataset must contain MLBAM 'pitcher' ID column.")
    if "game_date" not in raw.columns:
        raise ValueError("Statcast MiLB dataset must contain 'game_date' column.")

    allowed_ids = {str(int(x)) if str(x).strip().isdigit() else str(x).strip() for x in college_draftee_ids}
    pitcher_str = raw["pitcher"].apply(
        lambda v: str(int(v)) if pd.notna(v) and str(v).replace(".0", "").strip().isdigit() else str(v).strip()
    )
    df = raw.loc[pitcher_str.isin(allowed_ids)].copy()
    if df.empty:
        return df

    # Filter to Aug 1 - Sept 30 date window (month-day check works across any season)
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    valid_date_mask = dates.notna()
    month_day = list(zip(dates.dt.month.fillna(0).astype(int), dates.dt.day.fillna(0).astype(int)))
    in_window = pd.Series(
        [start_month_day <= md <= end_month_day for md in month_day],
        index=df.index,
    )
    df = df.loc[valid_date_mask & in_window].copy()
    if df.empty:
        return df

    # Convert pfx_x, pfx_z (Catcher's feet) -> TrackMan Pitcher's inches
    p_throws_raw = df["p_throws"] if "p_throws" in df.columns else pd.Series("R", index=df.index)
    ivb_in, hb_in = convert_statcast_movement_to_trackman(
        df["pfx_x"].astype(float),
        df["pfx_z"].astype(float),
        p_throws_raw,
    )

    pitcher_throws = _normalize_throws_series(p_throws_raw)
    batter_side = _normalize_batter_side_series(
        df["stand"] if "stand" in df.columns else pd.Series("R", index=df.index)
    )

    desc_lower = (
        df["description"].astype(str).str.strip().str.lower()
        if "description" in df.columns
        else pd.Series("ball", index=df.index)
    )
    events_lower = (
        df["events"].astype(str).str.strip().str.lower()
        if "events" in df.columns
        else pd.Series("", index=df.index)
    )
    pitch_call = desc_lower.map(STATCAST_DESCRIPTION_TO_PITCHCALL).fillna("BallCalled")
    play_result = events_lower.map(STATCAST_EVENTS_TO_PLAYRESULT).fillna("Undefined")
    kor_bb = np.where(
        events_lower.str.contains("strikeout", na=False),
        "Strikeout",
        np.where(events_lower.isin(["walk", "intent_walk"]), "Walk", "Undefined"),
    )

    rel_speed = pd.to_numeric(df["release_speed"], errors="coerce")
    rel_height = pd.to_numeric(df.get("release_pos_z", 5.85), errors="coerce").fillna(5.85)
    # Savant release_pos_x is Catcher's perspective (negative for RHP); negate to TrackMan Pitcher's perspective
    rel_side = -pd.to_numeric(df.get("release_pos_x", -1.85), errors="coerce").fillna(-1.85)
    ext = pd.to_numeric(df.get("release_extension", 6.15), errors="coerce").fillna(6.15)
    plate_z = pd.to_numeric(df.get("plate_z", 2.50), errors="coerce").fillna(2.50)
    plate_x = pd.to_numeric(df.get("plate_x", 0.0), errors="coerce").fillna(0.0)

    sign_hand = np.where(pitcher_throws.eq("LHP"), -1.0, 1.0)
    exp_vaa, exp_haa = compute_expected_vaa_and_haa(
        rel_speed_mph=rel_speed.fillna(91.5).to_numpy(dtype=np.float64),
        rel_height_ft=rel_height.to_numpy(dtype=np.float64),
        rel_side_norm_ft=rel_side.abs().to_numpy(dtype=np.float64),
        plate_loc_height_ft=plate_z.to_numpy(dtype=np.float64),
        plate_loc_side_norm_ft=(plate_x.to_numpy(dtype=np.float64) * sign_hand),
    )

    out = pd.DataFrame(
        {
            "PitcherId": "MLBAM_" + pitcher_str.loc[df.index].astype(str),
            "Pitcher": df["player_name"].astype(str) if "player_name" in df.columns else ("MLBAM_" + pitcher_str.loc[df.index].astype(str)),
            "Date": df["game_date"].astype(str),
            "PitcherThrows": pitcher_throws,
            "BatterSide": batter_side,
            "TaggedPitchType": df["pitch_type"].astype(str).str.upper(),
            "AutoPitchType": df["pitch_type"].astype(str).str.upper(),
            "PitchCall": pitch_call,
            "PlayResult": play_result,
            "KorBB": kor_bb,
            "Balls": pd.to_numeric(df.get("balls", 0), errors="coerce").fillna(0).astype(int),
            "Strikes": pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int),
            "RelSpeed": rel_speed,
            "SpinRate": pd.to_numeric(df.get("release_spin_rate", 2250.0), errors="coerce").fillna(2250.0),
            "SpinAxis": pd.to_numeric(df.get("spin_axis", 210.0), errors="coerce").fillna(210.0),
            "InducedVertBreak": ivb_in,
            "HorzBreak": hb_in,
            "RelHeight": rel_height,
            "RelSide": rel_side,
            "Extension": ext,
            "PlateLocHeight": plate_z,
            "PlateLocSide": plate_x,
            "VertApprAngle": exp_vaa,
            "HorzApprAngle": exp_haa * sign_hand,
            "ExitSpeed": pd.to_numeric(df.get("launch_speed", np.nan), errors="coerce"),
            "Angle": pd.to_numeric(df.get("launch_angle", np.nan), errors="coerce"),
            "cohort": cohort_label,
            "data_source": source_tag,
        }
    )

    out = out.dropna(subset=["RelSpeed", "InducedVertBreak", "HorzBreak"]).copy()
    if min_pitches_per_pitcher > 1:
        counts = out.groupby("PitcherId")["RelSpeed"].transform("size")
        out = out.loc[counts >= min_pitches_per_pitcher].copy()

    return out.reset_index(drop=True)


def anchor_baselines_strictly_to_college_cohort(
    engine: CollegeStuffEngine,
    df_features: pd.DataFrame,
    college_cohort_value: str = "college",
) -> Dict[str, PitchFamilyBaselineStats]:
    """
    Re-compute `mu_college_xrv`, `sigma_college_xrv`, and kinematic reference means
    STRICTLY on rows where `cohort == 'college'`, so `100 = College Avg` is anchored
    purely to collegiate pitching rather than pro draftees.
    """
    if "cohort" not in df_features.columns:
        raise ValueError("Merged training DataFrame must contain a 'cohort' column.")

    college_only = df_features.loc[df_features["cohort"] == college_cohort_value].copy()
    if college_only.empty:
        raise ValueError(f"No rows found with cohort == '{college_cohort_value}' for baseline anchoring.")

    anchored: Dict[str, PitchFamilyBaselineStats] = {}
    for family, fam_model in engine.family_models.items():
        sub_col = college_only.loc[college_only["PitchFamily"] == family]
        if len(sub_col) < 2:
            logger.warning(
                f"[{family}] Fewer than 2 college cohort rows ({len(sub_col)}); keeping existing baseline."
            )
            anchored[family] = fam_model.baseline_stats
            continue

        col_xrv = batch_neutralize_location_xrv(
            head_swing=fam_model.head_swing,
            head_whiff=fam_model.head_whiff,
            head_called_strike=fam_model.head_called_strike,
            head_xwobacon=fam_model.head_xwobacon,
            df_features=sub_col,
        )
        mu_col = float(np.nanmean(col_xrv))
        sigma_col = float(max(1e-5, np.nanstd(col_xrv, ddof=1)))

        updated_stats = PitchFamilyBaselineStats(
            pitch_family=family,
            count=int(len(sub_col)),
            mu_college_xrv=round(mu_col, 6),
            sigma_college_xrv=round(sigma_col, 6),
            velo_mean=round(float(sub_col["RelSpeed"].mean()), 2),
            velo_sd=round(float(max(0.5, sub_col["RelSpeed"].std(ddof=1))), 2),
            ivb_mean=round(float(sub_col["InducedVertBreak"].mean()), 2),
            ivb_sd=round(float(max(0.5, sub_col["InducedVertBreak"].std(ddof=1))), 2),
            hb_mean=round(float(sub_col["HB_norm"].mean()), 2),
            hb_sd=round(float(max(0.5, sub_col["HB_norm"].std(ddof=1))), 2),
            ext_mean=round(float(sub_col["Extension"].mean()), 2),
            ext_sd=round(float(max(0.1, sub_col["Extension"].std(ddof=1))), 2),
            vaa_mean=round(float(sub_col["VertApprAngle"].mean()), 2),
            vaa_sd=round(float(max(0.2, sub_col["VertApprAngle"].std(ddof=1))), 2),
            spin_mean=round(float(sub_col["SpinRate"].mean()), 1),
            spin_sd=round(float(max(20.0, sub_col["SpinRate"].std(ddof=1))), 1),
        )
        fam_model.baseline_stats = updated_stats
        fam_model.metrics.mu_college_xrv = round(mu_col, 6)
        fam_model.metrics.sigma_college_xrv = round(sigma_col, 6)
        anchored[family] = updated_stats

    return anchored


def train_college_anchored_candidate_model(
    college_df: pd.DataFrame,
    fsl_draftee_df: Optional[pd.DataFrame] = None,
    candidate_json_path: Union[str, Path] = DEFAULT_CANDIDATE_MODEL_PATH,
    min_pitches_per_pitcher: int = 30,
    n_estimators: int = 32,
    max_depth: int = 4,
    n_splits: int = 5,
    allow_overwrite_live: bool = False,
) -> Tuple[CollegeStuffEngine, Dict[str, Any]]:
    """
    Merge college TrackMan + optional FSL draftee Statcast DataFrames with a `cohort`
    tag, train the 4-head XGBoost cascade, anchor `100 = College Avg` (mu, sigma)
    strictly on `cohort == 'college'`, and export to `candidate_json_path`.

    Guards against accidentally overwriting the live `ncaa_d1_stuff_model.json`.
    """
    out_path = Path(candidate_json_path)
    if out_path.name == LIVE_MODEL_FILENAME and not allow_overwrite_live:
        raise ValueError(
            f"Refusing to overwrite live production model '{LIVE_MODEL_FILENAME}'. "
            f"Export to '{DEFAULT_CANDIDATE_MODEL_PATH}' instead."
        )

    parts = [college_df.assign(cohort="college")]
    if fsl_draftee_df is not None and not fsl_draftee_df.empty:
        parts.append(fsl_draftee_df.assign(cohort="fsl_draftee"))
    merged = pd.concat(parts, ignore_index=True)

    # Preserve cohort tag across ingestion & feature engineering
    cohort_by_pitcher = (
        merged.groupby("PitcherId")["cohort"]
        .agg(lambda s: "college" if (s == "college").any() else "fsl_draftee")
        .to_dict()
    )

    df_clean = ingest_trackman_dataframe(merged, min_pitches_per_pitcher=min_pitches_per_pitcher)
    df_feat = engineer_pitch_features(df_clean)
    df_feat["cohort"] = df_feat["PitcherId"].map(cohort_by_pitcher).fillna("college")

    engine = CollegeStuffEngine(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_splits=n_splits,
    )
    engine.training_sample_size = len(df_feat)

    for family in CANONICAL_PITCH_FAMILIES:
        sub = df_feat.loc[df_feat["PitchFamily"] == family]
        if len(sub) < 40 or sub["PitcherId"].nunique() < 2:
            continue
        engine.family_models[family] = engine.fit_pitch_family(sub, pitch_family=family)

    if not engine.family_models:
        raise RuntimeError("No pitch families met minimum sample threshold for training.")

    # Strictly anchor mu_college & sigma_college to cohort == 'college'
    anchor_baselines_strictly_to_college_cohort(engine, df_feat, college_cohort_value="college")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_model_for_web(engine=engine, json_output_path=out_path, js_output_path=None)
    return engine, payload


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2B Real-Data Ingestion & College-Anchored Candidate Model Calibration"
    )
    parser.add_argument("--wcl-dir", type=str, default=None, help="Directory or CSV of WCL TrackMan V3 files")
    parser.add_argument("--mendeley-csv", type=str, default=None, help="Path to Mendeley Track_Combo.csv")
    parser.add_argument("--fsl-csv", type=str, default=None, help="Directory or CSV of Aug-Sept FSL Statcast files")
    parser.add_argument("--draftee-ids-csv", type=str, default=None, help="CSV with 'mlbam_id' or 'pitcher' column")
    parser.add_argument(
        "--draft-years",
        nargs="*",
        type=int,
        default=None,
        help="MLB Draft years (e.g. 2024 2025 2026) to auto-fetch college/JUCO MLBAM IDs via Stats API",
    )
    parser.add_argument(
        "--no-filter-batter-team",
        action="store_true",
        help="Disable WCL BatterTeam whitelist filtering when loading --wcl-dir",
    )
    parser.add_argument("--divisions", nargs="*", default=None, help="Optional division whitelist (e.g. D1)")
    parser.add_argument("--min-pitches", type=int, default=30, help="Minimum pitches per pitcher (default: 30)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CANDIDATE_MODEL_PATH),
        help="Candidate JSON output path (must not overwrite live ncaa_d1_stuff_model.json)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_cli_args()
    college_parts: List[pd.DataFrame] = []

    if args.wcl_dir:
        college_parts.append(
            load_trackman_v3_dataset(
                args.wcl_dir,
                wcl_teams=DEFAULT_WCL_TEAMS,
                filter_batter_team=not args.no_filter_batter_team,
                divisions=args.divisions,
                min_pitches_per_pitcher=args.min_pitches,
                cohort_label="college",
                source_tag="wcl_trackman_v3",
            )
        )
    if args.mendeley_csv:
        college_parts.append(
            load_trackman_v3_dataset(
                args.mendeley_csv,
                wcl_teams=None,
                filter_batter_team=False,
                divisions=args.divisions,
                min_pitches_per_pitcher=args.min_pitches,
                cohort_label="college",
                source_tag="mendeley_track_combo",
            )
        )

    if not college_parts:
        raise SystemExit("Provide at least one college TrackMan source (--wcl-dir or --mendeley-csv).")

    college_df = pd.concat(college_parts, ignore_index=True)
    fsl_df = None
    if args.fsl_csv:
        draftee_ids: Set[Union[int, str]] = set()
        if args.draftee_ids_csv:
            id_df = pd.read_csv(args.draftee_ids_csv)
            id_col = "mlbam_id" if "mlbam_id" in id_df.columns else id_df.columns[0]
            draftee_ids.update(id_df[id_col].dropna().tolist())
        if args.draft_years:
            fetched_ids = fetch_mlb_draft_college_ids(args.draft_years)
            logger.info("Fetched %d college/JUCO draftee MLBAM IDs for years %s", len(fetched_ids), args.draft_years)
            draftee_ids.update(fetched_ids)

        if draftee_ids:
            fsl_df = load_statcast_milb_dataset(
                args.fsl_csv,
                college_draftee_ids=draftee_ids,
                min_pitches_per_pitcher=args.min_pitches,
            )
        else:
            logger.warning("--fsl-csv provided without --draftee-ids-csv or --draft-years; skipping FSL cohort.")

    train_college_anchored_candidate_model(
        college_df=college_df,
        fsl_draftee_df=fsl_df,
        candidate_json_path=args.output,
        min_pitches_per_pitcher=args.min_pitches,
    )
    logger.info("Candidate model exported cleanly to %s", args.output)
