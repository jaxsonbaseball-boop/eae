"""Project Apex v3.5 High School-to-NCAA Division 1 Stuff+ & 3-Year College Growth Projection Engine."""

from dataclasses import dataclass
import math
from typing import Dict, Optional

from src.physics.constants import NCAA_D1_BASELINES, D1PitchBaseline


@dataclass
class StuffGradeResult:
    pitch_type: str
    platoon_side: str
    projection_mode: str
    raw_stuff_plus: float
    velo_component: float
    movement_component: float
    plane_vaa_component: float
    tunnel_delta_component: float
    sub88_cliff_penalty: float
    pvaa_deg: float
    phaa_deg: float
    projected_velo_mph: float
    projected_extension_ft: float


def compute_location_adjusted_pvaa(
    vaa_deg: float,
    plate_z_ft: float,
    rel_height_ft: float = 5.85,
) -> float:
    """
    Compute Baseball Prospectus-style Location-Adjusted Vertical Approach Angle (PVAA).
    Removes the geometric effect of pitch height at the plate (approx +1.32 deg per foot of PlateLocHeight).
    """
    expected_vaa = -4.85 + 1.32 * (plate_z_ft - 2.50) - 0.85 * (rel_height_ft - 5.85)
    return round(vaa_deg - expected_vaa, 2)


def compute_fastball_anchor_viability(fb_velo_mph: float) -> float:
    """
    Fastball Anchor Viability Factor gamma_anchor in [0.15, 1.00].
    Discounts secondary pitch tunnel credit when a pitcher's primary fastball is below 90 mph.
    """
    return round(min(1.0, max(0.15, (fb_velo_mph - 78.0) / 12.0)), 4)


def score_pitch_stuff_plus(
    pitch_type: str,
    rel_speed_mph: float,
    ivb_norm_in: float,
    hb_norm_in: float,
    spin_rate_rpm: float,
    spin_efficiency_pct: float,
    extension_ft: float,
    rel_height_ft: float,
    rel_side_ft: float,
    vaa_deg: float,
    haa_deg: float = -1.2,
    plate_z_ft: float = 2.50,
    pitcher_throws: str = "RHP",
    platoon_batter: str = "RHB",
    primary_fb_velo_mph: Optional[float] = None,
    primary_fb_ivb_in: Optional[float] = None,
    primary_fb_hb_in: Optional[float] = None,
    college_growth_projection: bool = False,
) -> StuffGradeResult:
    """
    Compute NCAA Division 1 calibrated Stuff+ (100 = D1 Mean, SD = 10) for any pitch,
    incorporating:
      - Non-linear sub-88 mph fastball velocity cliff: -0.45 * (88.0 - v_fb)^1.6
      - 3.85 Stuff+ per mph fastball velocity slope
      - Fastball Anchor Viability Factor gamma_anchor gating offspeed tunnel credit
      - Present D1 Reality vs. 3-Year College Growth Projection (+5.0 mph FB, +3.5 mph offspeed, +0.3 ft ext)
      - Opposite-handed platoon splits (vs. RHB / vs. LHB)
    """
    pt = pitch_type.upper()
    baseline: D1PitchBaseline = NCAA_D1_BASELINES.get(pt, NCAA_D1_BASELINES["FF"])
    is_fastball = pt in ("FF", "SI", "FC")

    # Apply 3-Year College Strength & Biomechanical Growth Projection if enabled
    if college_growth_projection:
        velo_boost = 5.0 if pt in ("FF", "SI") else 3.5
        ext_boost = 0.30
    else:
        velo_boost = 0.0
        ext_boost = 0.0

    v_eff_mph = rel_speed_mph + velo_boost
    ext_eff_ft = extension_ft + ext_boost
    fb_ref_velo = (primary_fb_velo_mph + (5.0 if college_growth_projection else 0.0)) if primary_fb_velo_mph else v_eff_mph

    # Canonicalize horizontal sign (positive = arm-side run, negative = glove-side sweep)
    sign_h = -1.0 if pitcher_throws.upper().startswith("L") else 1.0
    hb_canonical = hb_norm_in * sign_h

    pvaa = compute_location_adjusted_pvaa(vaa_deg, plate_z_ft=plate_z_ft, rel_height_ft=rel_height_ft)
    phaa = round(haa_deg * sign_h, 2)

    sub88_penalty = 0.0
    tunnel_component = 0.0

    if pt in ("FF", "SI"):
        # Fastball velocity slope (3.85 Stuff+/mph) + non-linear sub-88 mph cliff
        velo_component = 3.85 * (v_eff_mph - baseline.velo_mph)
        if v_eff_mph < 88.0:
            sub88_penalty = -0.45 * ((88.0 - v_eff_mph) ** 1.6)
        if pt == "FF":
            movement_component = 0.85 * (ivb_norm_in - baseline.ivb_in) + 0.30 * max(0.0, hb_canonical - baseline.hb_in)
            plane_component = 2.65 * pvaa + 1.40 * (ext_eff_ft - baseline.extension_ft)
        else:
            # Sinker rewards arm-side run and steep sink (negative IVB delta + negative PVAA)
            movement_component = 0.75 * (hb_canonical - baseline.hb_in) + 0.55 * (baseline.ivb_in - ivb_norm_in)
            plane_component = -1.50 * pvaa + 1.35 * (ext_eff_ft - baseline.extension_ft)
    elif pt == "FC":
        velo_component = 2.60 * (v_eff_mph - baseline.velo_mph)
        if v_eff_mph < 84.0:
            sub88_penalty = -0.35 * ((84.0 - v_eff_mph) ** 1.5)
        movement_component = 0.70 * abs(hb_canonical - baseline.hb_in) + 0.45 * (ivb_norm_in - baseline.ivb_in)
        plane_component = 1.20 * (ext_eff_ft - baseline.extension_ft)
    else:
        # Secondary pitch (SL, ST, CU, CH, FS)
        velo_component = 1.65 * (v_eff_mph - baseline.velo_mph)
        if pt in ("SL", "ST", "CU"):
            glove_sweep = max(0.0, -hb_canonical)
            base_sweep = abs(baseline.hb_in)
            depth_gain = max(0.0, baseline.ivb_in - ivb_norm_in)
            spin_bonus = 0.008 * max(0.0, spin_rate_rpm - baseline.spin_rpm)
            movement_component = 0.80 * (glove_sweep - base_sweep) + 0.65 * depth_gain + spin_bonus
        else:
            # Changeup / Splitter rewards vertical separation and arm-side fade + low spin
            fb_ivb = primary_fb_ivb_in if primary_fb_ivb_in is not None else 15.5
            ivb_drop = max(0.0, fb_ivb - ivb_norm_in)
            arm_fade = max(0.0, hb_canonical - baseline.hb_in)
            low_spin_bonus = 0.006 * max(0.0, baseline.spin_rpm - spin_rate_rpm)
            movement_component = 0.85 * (ivb_drop - 9.0) + 0.65 * arm_fade + low_spin_bonus

        plane_component = 1.15 * (ext_eff_ft - baseline.extension_ft)

        # Tunnel differential component gated by Fastball Anchor Viability Factor gamma_anchor
        gamma_anchor = compute_fastball_anchor_viability(fb_ref_velo)
        velo_gap = fb_ref_velo - v_eff_mph
        ideal_gap = 9.2 if pt in ("CH", "FS") else 8.5
        raw_tunnel_credit = max(0.0, 5.5 - 0.55 * abs(velo_gap - ideal_gap))
        tunnel_component = gamma_anchor * raw_tunnel_credit

    # Platoon adjustment (same-handed vs opposite-handed batter)
    same_hand = pitcher_throws.upper()[0] == platoon_batter.upper()[0]
    if pt in ("SL", "ST"):
        platoon_adj = 2.8 if same_hand else -2.2
    elif pt in ("CH", "FS"):
        platoon_adj = -2.5 if same_hand else 3.4
    elif pt == "SI":
        platoon_adj = 2.1 if same_hand else -1.6
    else:
        platoon_adj = 0.5 if same_hand else 0.0

    raw_stuff = (
        100.0
        + velo_component
        + movement_component
        + plane_component
        + tunnel_component
        + sub88_penalty
        + platoon_adj
    )
    clamped_stuff = round(max(40.0, min(160.0, raw_stuff)), 1)

    return StuffGradeResult(
        pitch_type=pt,
        platoon_side=platoon_batter.upper(),
        projection_mode="3YR_COLLEGE_GROWTH" if college_growth_projection else "PRESENT_D1_REALITY",
        raw_stuff_plus=clamped_stuff,
        velo_component=round(velo_component, 2),
        movement_component=round(movement_component, 2),
        plane_vaa_component=round(plane_component, 2),
        tunnel_delta_component=round(tunnel_component, 2),
        sub88_cliff_penalty=round(sub88_penalty, 2),
        pvaa_deg=pvaa,
        phaa_deg=phaa,
        projected_velo_mph=round(v_eff_mph, 1),
        projected_extension_ft=round(ext_eff_ft, 2),
    )
