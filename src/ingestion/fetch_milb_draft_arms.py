"""
EAE Phase 2B-1: MLB Stats API Post-Draft MiLB Pitch Telemetry Ingestion.

Fetches and caches post-draft tracked games across:
  - Complex / Rookie (sportId=16)
  - Single-A / Low-A (sportId=14)
  - High-A (sportId=13)
strictly restricted (100.0%) to 4-year college pitchers drafted in that same
calendar year (`school.schoolClass` starting with "4YR", excluding JC/HS),
aged 20.0 <= age_on_game_date < 23.0, with full optical tracking.

Outputs `data/processed/milb_postdraft_pitches.csv` with exact schema parity
to `index.html` (`scorePitch10x10`) and `src/models/college_stuff_engine.py`.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("eae.fetch_milb_draft_arms")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLB_API_BASE = "https://statsapi.mlb.com/api"
DEFAULT_SEASONS: Tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
DEFAULT_SPORT_IDS: Tuple[int, ...] = (16, 14, 13)  # 16=Complex/Rookie, 14=Single-A, 13=High-A
DEFAULT_CACHE_DIR = Path("data/raw/milb_statsapi_cache")
DEFAULT_OUTPUT_CSV = Path("data/processed/milb_postdraft_pitches.csv")

PITCH_FAMILY_MAP: Dict[str, str] = {
    "FF": "FF", "FA": "FF", "FASTBALL": "FF", "FOURSEAMFASTBALL": "FF", "4-SEAM": "FF", "FOUR-SEAM": "FF",
    "SI": "SI", "FT": "SI", "SINKER": "SI", "TWOSEAMFASTBALL": "SI", "2-SEAM": "SI", "TWO-SEAM": "SI",
    "FC": "FC", "CUTTER": "FC",
    "SL": "SL", "SLIDER": "SL",
    "SW": "SW", "ST": "SW", "SWEEPER": "SW", "SLURVE": "SW",
    "CU": "CU", "KC": "CU", "CS": "CU", "CURVEBALL": "CU", "KNUCKLECURVE": "CU",
    "CH": "CH", "CHANGEUP": "CH", "CHANGE": "CH",
    "FS": "FS", "FO": "FS", "SPLITTER": "FS", "SPLITFINGER": "FS",
}


def map_pitch_family(raw_tag: Any, ivb: float, hb_norm: float) -> str:
    """Exact parity with EAE `index.html` `mapPitchFamily(rawTag, ivb, hbNorm)`."""
    clean = str(raw_tag or "FF").strip().upper().replace(" ", "").replace("_", "")
    mapped = PITCH_FAMILY_MAP.get(clean, "FF")
    if mapped == "SL" and math.isfinite(hb_norm) and math.isfinite(ivb):
        if hb_norm <= -12.0 and ivb >= -4.5:
            return "SW"
    return mapped


def _throttled_get_json(
    session: requests.Session,
    url: str,
    cache_path: Optional[Path] = None,
    throttle_sec: float = 0.25,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """Fetch JSON with local disk cache, 0.25s throttling, and exponential backoff on 429/5xx."""
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for attempt in range(max_retries):
        time.sleep(throttle_sec)
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                return data
            if resp.status_code in (429, 500, 502, 503, 504):
                wait_s = (2 ** attempt) * 0.5
                logger.warning(f"HTTP {resp.status_code} for {url}; backing off {wait_s:.1f}s")
                time.sleep(wait_s)
                continue
            logger.warning(f"HTTP {resp.status_code} for {url}; skipping")
            return None
        except requests.RequestException as exc:
            wait_s = (2 ** attempt) * 0.5
            logger.warning(f"Request error ({exc}) for {url}; backing off {wait_s:.1f}s")
            time.sleep(wait_s)
    return None


def build_4yr_college_draft_whitelist(
    seasons: Sequence[int],
    cache_dir: Path,
    session: requests.Session,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Step A: Fetch `/api/v1/draft/{year}` and whitelist ONLY picks whose
    `school.schoolClass` (uppercased) starts with "4YR" (strictly excluding JC/HS).
    """
    whitelist: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for year in seasons:
        url = f"{MLB_API_BASE}/v1/draft/{year}"
        cache_path = cache_dir / f"draft_{year}.json"
        data = _throttled_get_json(session, url, cache_path=cache_path)
        if not data:
            continue

        rounds = (data.get("drafts") or {}).get("rounds") or []
        year_count = 0
        for rd in rounds:
            round_label = str(rd.get("round", ""))
            for pick in rd.get("picks") or []:
                school = pick.get("school") or {}
                school_class = str(school.get("schoolClass") or "").strip().upper()
                if not school_class.startswith("4YR"):
                    continue
                person = pick.get("person") or {}
                pid = person.get("id")
                if pid is None:
                    continue
                whitelist[(int(pid), int(year))] = {
                    "school_name": str(school.get("name") or "").strip(),
                    "school_class": school_class,
                    "pick_round": round_label,
                    "pick_number": pick.get("pickNumber"),
                    "birth_date": person.get("birthDate"),
                }
                year_count += 1
        logger.info(f"Draft {year}: whitelisted {year_count} 4-year college draftees (4YR*).")
    return whitelist


def fetch_postdraft_game_pks(
    seasons: Sequence[int],
    sport_ids: Sequence[int],
    cache_dir: Path,
    session: requests.Session,
) -> List[Tuple[int, int, int]]:
    """
    Step B: Query MLB Stats API schedule for sportIds [16, 14, 13] strictly within
    `YYYY-07-10` to `YYYY-09-30`. Returns list of `(gamePk, season, sportId)`.
    """
    games: List[Tuple[int, int, int]] = []
    seen_pks = set()
    for year in seasons:
        start_dt = f"{year}-07-10"
        end_dt = f"{year}-09-30"
        for sid in sport_ids:
            url = (
                f"{MLB_API_BASE}/v1/schedule?sportId={sid}"
                f"&startDate={start_dt}&endDate={end_dt}&gameType=R"
            )
            cache_path = cache_dir / f"schedule_{year}_sport{sid}.json"
            data = _throttled_get_json(session, url, cache_path=cache_path)
            if not data:
                continue
            for dt_entry in data.get("dates") or []:
                for gm in dt_entry.get("games") or []:
                    pk = gm.get("gamePk")
                    if pk is not None and int(pk) not in seen_pks:
                        seen_pks.add(int(pk))
                        games.append((int(pk), int(year), int(sid)))
        logger.info(f"Season {year}: total unique post-draft games queued = {len(seen_pks)}")
    return games


def compute_exact_age_years(birth_date_str: Optional[str], game_date_str: Optional[str]) -> Optional[float]:
    """Compute exact age in years on `game_date_str` from `birthDate` (`YYYY-MM-DD`)."""
    if not birth_date_str or not game_date_str:
        return None
    try:
        b = datetime.strptime(str(birth_date_str)[:10], "%Y-%m-%d").date()
        g = datetime.strptime(str(game_date_str)[:10], "%Y-%m-%d").date()
        return (g - b).days / 365.2425
    except ValueError:
        return None


def estimate_xwobacon_from_hitdata(launch_speed: Any, launch_angle: Any) -> float:
    """Estimate batted-ball xwOBAcon from exit velocity (mph) and launch angle (deg)."""
    if launch_speed is None or launch_angle is None:
        return 0.340
    try:
        ev = float(launch_speed)
        la = float(launch_angle)
    except (TypeError, ValueError):
        return 0.340
    if not (math.isfinite(ev) and math.isfinite(la)):
        return 0.340
    # Smooth empirical barrel/sweet-spot xwOBAcon surface anchored around NCAA/Low-A contact
    ev_term = 0.011 * (ev - 87.0)
    la_sweet = math.exp(-0.5 * ((la - 16.0) / 14.0) ** 2)
    la_gb_pen = -0.14 if la < 0.0 else (-0.12 if la > 45.0 else 0.0)
    val = 0.24 + ev_term + 0.34 * la_sweet + la_gb_pen
    return float(max(0.050, min(1.350, val)))


def compute_expected_vaa_and_haa(
    rel_speed_mph: float,
    rel_height_ft: float,
    rel_side_norm_ft: float,
    plate_z_ft: float,
    plate_x_norm_ft: float,
) -> Tuple[float, float]:
    """Exact parity with `college_stuff_engine.compute_expected_vaa_and_haa` & `index.html`."""
    exp_vaa = (
        -4.85
        + 1.32 * (plate_z_ft - 2.50)
        + 0.085 * (rel_speed_mph - 91.5)
        - 0.85 * (rel_height_ft - 5.85)
    )
    exp_haa = -1.25 + 1.05 * plate_x_norm_ft - 0.90 * (rel_side_norm_ft - 1.85)
    return exp_vaa, exp_haa


def extract_pitches_from_live_feed(
    feed: Mapping[str, Any],
    game_pk: int,
    game_year: int,
    sport_id: int,
    whitelist: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Step C & D: Apply strict 4YR draftee, age [20.0, 23.0), and optical-tracking filters,
    and normalize features to exact EAE schema parity.
    """
    game_data = feed.get("gameData") or {}
    official_date = ((game_data.get("datetime") or {}).get("officialDate")) or f"{game_year}-08-01"
    players_meta = game_data.get("players") or {}

    all_plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
    rows: List[Dict[str, Any]] = []

    for play in all_plays:
        matchup = play.get("matchup") or {}
        pitcher_obj = matchup.get("pitcher") or {}
        batter_obj = matchup.get("batter") or {}
        pid = pitcher_obj.get("id")
        if pid is None:
            continue
        pid_int = int(pid)
        draft_info = whitelist.get((pid_int, game_year))
        if not draft_info:
            continue

        p_key = f"ID{pid_int}"
        p_meta = players_meta.get(p_key) or {}
        birth_date = p_meta.get("birthDate") or draft_info.get("birth_date")
        age_on_game = compute_exact_age_years(birth_date, official_date)
        if age_on_game is None or not (20.0 <= age_on_game < 23.0):
            continue

        p_throws_raw = ((matchup.get("pitchHand") or {}).get("code")) or ((p_meta.get("pitchHand") or {}).get("code")) or "R"
        p_throws = "LHP" if str(p_throws_raw).upper().startswith("L") else "RHP"
        sign_hand = -1.0 if p_throws == "LHP" else 1.0

        b_side_raw = ((matchup.get("batSide") or {}).get("code")) or "R"
        b_side = "LHB" if str(b_side_raw).upper().startswith("L") else "RHB"

        for ev in play.get("playEvents") or []:
            if not ev.get("isPitch"):
                continue
            pd_obj = ev.get("pitchData") or {}
            breaks = pd_obj.get("breaks") or {}
            coords = pd_obj.get("coordinates") or {}

            start_speed = pd_obj.get("startSpeed")
            ivb = breaks.get("breakVerticalInduced")
            hb_raw = breaks.get("breakHorizontal")
            spin_rate = breaks.get("spinRate")
            x0 = coords.get("x0")
            z0 = coords.get("z0")
            ext = pd_obj.get("extension")

            # Strict optical tracking gate
            if any(v is None for v in (start_speed, ivb, hb_raw, spin_rate, x0, z0, ext)):
                continue

            try:
                velo = float(start_speed)
                ivb_f = float(ivb)
                hb_f = float(hb_raw)
                spin_f = float(spin_rate)
                x0_f = float(x0)
                z0_f = float(z0)
                ext_f = float(ext)
            except (TypeError, ValueError):
                continue

            if not (55.0 <= velo <= 106.0 and 400.0 <= spin_f <= 3600.0 and 3.5 <= ext_f <= 8.0 and 3.0 <= z0_f <= 7.8):
                continue

            # MLB Stats API `breakHorizontal` is catcher-view inches (+ = toward 1B / right)
            # TrackMan Pitcher-view HB = -breakHorizontal; HB_norm (arm-side positive) = TrackMan_HB * sign_hand
            hb_trackman = -hb_f
            hb_norm = hb_trackman * sign_hand
            rel_side_norm = abs(x0_f)

            details = ev.get("details") or {}
            raw_type = ((details.get("type") or {}).get("code")) or ((details.get("type") or {}).get("description")) or "FF"
            pitch_family = map_pitch_family(raw_type, ivb_f, hb_norm)

            spin_axis_raw = breaks.get("spinDirection")
            if spin_axis_raw is not None:
                sa = float(spin_axis_raw)
                spin_axis_norm = (360.0 - sa) % 360.0 if p_throws == "LHP" else sa % 360.0
            else:
                spin_axis_norm = 200.0 if pitch_family in ("FF", "SI") else 120.0

            pz = float(coords.get("pZ") if coords.get("pZ") is not None else 2.50)
            px = float(coords.get("pX") if coords.get("pX") is not None else 0.0)
            # MLB Stats API pX is catcher-view (+ = right / 1B); PlateLocSide_norm (arm-side positive)
            plate_x_norm = (-px) * sign_hand

            d_flight = max(46.0, 60.5 - ext_f - 1.417)
            perceived_speed = velo + 1.45 * (ext_f - 6.0)
            bauer_units = spin_f / max(50.0, velo)
            arm_slot_angle = math.degrees(math.atan2(z0_f - 5.05, max(0.05, rel_side_norm - 0.45)))

            vy0 = float(coords.get("vY0") or (-velo * 1.467 * 0.992))
            vz0 = float(coords.get("vZ0") or -2.5)
            vx0 = float(coords.get("vX0") or 0.0)
            vaa_est = math.degrees(math.atan2(vz0, abs(vy0))) if vy0 != 0 else -4.85
            haa_est = math.degrees(math.atan2(-vx0 * sign_hand, abs(vy0))) if vy0 != 0 else -1.25

            exp_vaa, exp_haa = compute_expected_vaa_and_haa(velo, z0_f, rel_side_norm, pz, plate_x_norm)
            vaa_diff = vaa_est - exp_vaa
            haa_diff = haa_est - exp_haa

            zone_dist = math.hypot(max(0.0, abs(px) - 0.833), max(0.0, pz - 3.5, 1.5 - pz))
            in_strike_zone = 1 if (abs(px) <= 0.833 and 1.5 <= pz <= 3.5) else 0

            call_desc = str(details.get("description") or "").lower()
            call_code = str(details.get("code") or "").upper()
            is_in_play = 1 if (details.get("isInPlay") or call_code in ("X", "D", "E")) else 0
            is_whiff = 1 if ("swinging" in call_desc or call_code in ("S", "W", "M")) else 0
            is_called_strike = 1 if ("called strike" in call_desc or call_code == "C") else 0
            is_foul = 1 if ("foul" in call_desc and "tip" not in call_desc) else 0
            is_swing = 1 if (is_whiff or is_foul or is_in_play or "foul" in call_desc) else 0

            hit_data = ev.get("hitData") or {}
            xwobacon = estimate_xwobacon_from_hitdata(hit_data.get("launchSpeed"), hit_data.get("launchAngle")) if is_in_play else 0.340

            cnt = ev.get("count") or {}
            balls = int(cnt.get("balls") or 0)
            strikes = int(cnt.get("strikes") or 0)
            count_leverage = float(strikes - balls)
            platoon_same = 1 if (p_throws[0] == b_side[0]) else 0

            rows.append({
                "game_pk": game_pk,
                "game_year": game_year,
                "game_date": official_date,
                "sport_id": sport_id,
                "PitcherId": f"milb_{pid_int}_{game_year}",
                "pitcher_mlbam_id": pid_int,
                "school_name": draft_info["school_name"],
                "school_class": draft_info["school_class"],
                "draft_round": draft_info["pick_round"],
                "age_on_game_date": round(age_on_game, 3),
                "PitcherThrows": p_throws,
                "BatterSide": b_side,
                "TaggedPitchType": raw_type,
                "PitchFamily": pitch_family,
                "RelSpeed": velo,
                "PerceivedSpeed": perceived_speed,
                "SpinRate": spin_f,
                "BauerUnits": bauer_units,
                "SpinAxis_norm": spin_axis_norm,
                "InducedVertBreak": ivb_f,
                "HorzBreak": hb_trackman,
                "HB_norm": hb_norm,
                "RelHeight": z0_f,
                "RelSide": x0_f,
                "RelSide_norm": rel_side_norm,
                "Extension": ext_f,
                "ArmSlotAngle": arm_slot_angle,
                "VertApprAngle": vaa_est,
                "HorzApprAngle_norm": haa_est,
                "VAA_diff": vaa_diff,
                "HAA_diff": haa_diff,
                "PlateLocHeight": pz,
                "PlateLocSide_norm": plate_x_norm,
                "ZoneDist": zone_dist,
                "InStrikeZone": in_strike_zone,
                "PlatoonSameHand": platoon_same,
                "IsLHP": 1 if p_throws == "LHP" else 0,
                "IsLHB": 1 if b_side == "LHB" else 0,
                "CountLeverage": count_leverage,
                "Balls": balls,
                "Strikes": strikes,
                "is_swing": is_swing,
                "is_whiff": is_whiff,
                "is_called_strike": is_called_strike,
                "is_foul": is_foul,
                "is_in_play": is_in_play,
                "is_contact": 1 if (is_in_play or is_foul) else 0,
                "target_xwobacon": xwobacon,
            })
    return rows


def add_cross_level_fastball_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by `(pitcher_mlbam_id, game_year)` across all three sportIds (16, 14, 13)
    to compute each pitcher's mean FF/SI `fb_velo_anchor`, `fb_ivb_anchor`, and
    `fb_hb_anchor` across their entire post-draft summer, and compute `VeloDiff_FB`,
    `IVBDiff_FB`, `HBDiff_FB`. Drops secondary rows if pitcher logged 0 fastballs.
    """
    if df.empty:
        return df

    fb_mask = df["PitchFamily"].isin(["FF", "SI"])
    fb_anchors = (
        df.loc[fb_mask]
        .groupby(["pitcher_mlbam_id", "game_year"], as_index=False)
        .agg(
            fb_velo_anchor=("RelSpeed", "mean"),
            fb_ivb_anchor=("InducedVertBreak", "mean"),
            fb_hb_anchor=("HB_norm", "mean"),
        )
    )
    merged = df.merge(fb_anchors, on=["pitcher_mlbam_id", "game_year"], how="inner")
    merged["VeloDiff_FB"] = np.where(
        merged["PitchFamily"].isin(["FF", "SI"]),
        0.0,
        merged["RelSpeed"] - merged["fb_velo_anchor"],
    )
    merged["IVBDiff_FB"] = np.where(
        merged["PitchFamily"].isin(["FF", "SI"]),
        0.0,
        merged["InducedVertBreak"] - merged["fb_ivb_anchor"],
    )
    merged["HBDiff_FB"] = np.where(
        merged["PitchFamily"].isin(["FF", "SI"]),
        0.0,
        merged["HB_norm"] - merged["fb_hb_anchor"],
    )
    return merged


def print_summary_breakdown(df: pd.DataFrame) -> None:
    """Print summary breakdown by sport_id, school_class, PitchFamily, and velocity quantiles."""
    if df.empty:
        logger.warning("No qualifying post-draft 4YR pitches found.")
        return
    logger.info(f"Total qualifying post-draft 4YR college pitches: {len(df):,}")
    logger.info("Breakdown by sport_id (16=Complex, 14=Single-A, 13=High-A):\n" + str(df["sport_id"].value_counts().sort_index()))
    logger.info("Breakdown by school_class:\n" + str(df["school_class"].value_counts()))
    quant = df.groupby("PitchFamily")["RelSpeed"].quantile([0.10, 0.50, 0.90]).unstack()
    quant.columns = ["p10_mph", "p50_mph", "p90_mph"]
    quant["count"] = df.groupby("PitchFamily")["RelSpeed"].count()
    logger.info("Breakdown by PitchFamily & Velocity Quantiles:\n" + str(quant.round(1)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch MLB Stats API post-draft 4YR college MiLB pitch telemetry.")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_SEASONS))
    parser.add_argument("--sport-ids", nargs="+", type=int, default=list(DEFAULT_SPORT_IDS))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-games", type=int, default=0, help="Optional cap on live game feeds (0 = all)")
    args = parser.parse_args(argv)

    session = requests.Session()
    session.headers.update({"User-Agent": "EAE-Baseball-Research/2B-1"})

    whitelist = build_4yr_college_draft_whitelist(args.seasons, args.cache_dir, session)
    games = fetch_postdraft_game_pks(args.seasons, args.sport_ids, args.cache_dir, session)
    if args.max_games > 0:
        games = games[: args.max_games]

    all_rows: List[Dict[str, Any]] = []
    for idx, (game_pk, year, sid) in enumerate(games, start=1):
        url = f"{MLB_API_BASE}/v1.1/game/{game_pk}/feed/live"
        cache_path = args.cache_dir / f"game_{game_pk}.json"
        feed = _throttled_get_json(session, url, cache_path=cache_path)
        if not feed:
            continue
        rows = extract_pitches_from_live_feed(feed, game_pk, year, sid, whitelist)
        if rows:
            all_rows.extend(rows)
        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(games)} games | Accumulated {len(all_rows):,} qualifying pitches")

    df = pd.DataFrame(all_rows)
    df = add_cross_level_fastball_anchors(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved clean post-draft dataset to {args.output_csv}")
    print_summary_breakdown(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
