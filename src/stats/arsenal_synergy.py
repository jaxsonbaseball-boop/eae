"""Arsenal+ Synergy Matrix: 50% Fastball Baseline Weight, Anchor Viability Gate, Tunnel & EvD."""

from dataclasses import dataclass
import math
from typing import Dict, List

from src.models.college_stuff_engine import compute_fastball_anchor_viability


@dataclass
class ArsenalSynergyReport:
    arsenal_plus: float
    fastball_stuff_plus: float
    secondary_weighted_stuff_plus: float
    gamma_anchor: float
    tunnel_bonus: float
    effective_velocity_dispersion_bonus: float
    mirror_spin_bonus: float
    platoon_coverage_bonus: float
    scouting_20_80_grade: int


def stuff_plus_to_20_80_scale(stuff_plus: float) -> int:
    """Map NCAA D1 Stuff+ (100 mean, 10 SD) to traditional MLB/NCAA 20-80 scouting scale (50 = D1 average)."""
    raw_grade = 50.0 + (stuff_plus - 100.0)
    rounded_5 = int(5 * round(raw_grade / 5.0))
    return max(20, min(80, rounded_5))


def compute_arsenal_plus(
    pitches: List[Dict[str, float]],
) -> ArsenalSynergyReport:
    """
    Compute overall Arsenal+ using Project Apex's reality-grounded formulation:
      - Primary fastball carries a fixed 50% baseline weight (w_FB = 0.50).
      - Secondary pitches share the remaining 50% weight.
      - Tunnel, Effective Velocity Dispersion (EvD), and Mirror Spin bonuses are multiplied
        by the Fastball Anchor Viability Factor gamma_anchor in [0.15, 1.00].
    """
    if not pitches:
        return ArsenalSynergyReport(90.0, 90.0, 90.0, 0.5, 0.0, 0.0, 0.0, 0.0, 40)

    fastballs = [p for p in pitches if str(p.get("pitch_type", "FF")).upper() in ("FF", "SI", "FC")]
    secondaries = [p for p in pitches if str(p.get("pitch_type", "FF")).upper() not in ("FF", "SI", "FC")]

    primary_fb = max(fastballs, key=lambda x: x.get("rel_speed_mph", 0.0)) if fastballs else pitches[0]
    fb_stuff = float(primary_fb.get("stuff_plus", 90.0))
    fb_velo = float(primary_fb.get("rel_speed_mph", 85.0))
    gamma_anchor = compute_fastball_anchor_viability(fb_velo)

    if secondaries:
        sec_stuff = sum(float(p.get("stuff_plus", 90.0)) for p in secondaries) / len(secondaries)
    else:
        sec_stuff = fb_stuff

    # 50% Fastball Anchor Weight + 50% Secondary Weight
    base_weighted = 0.50 * fb_stuff + 0.50 * sec_stuff

    # Effective Velocity Dispersion (EvD) across repertoire
    velos = [float(p.get("rel_speed_mph", fb_velo)) for p in pitches]
    mean_v = sum(velos) / len(velos)
    evd_sd = math.sqrt(sum((v - mean_v) ** 2 for v in velos) / max(1, len(velos)))
    raw_evd_bonus = min(3.2, max(0.0, (evd_sd - 3.2) * 0.95))

    # Tunnel separation at 167 ms decision point vs break plane separation
    raw_tunnel_bonus = min(3.8, 1.25 * len(secondaries))

    # Mirror-axis spin efficiency synergy (FF backspin + CU/SL gyro/topspin pairing)
    raw_mirror_bonus = 1.6 if len(secondaries) >= 2 else 0.6

    # Platoon coverage (both glove-side breaker and arm-side changeup/splitter present)
    sec_types = {str(p.get("pitch_type", "")).upper() for p in secondaries}
    has_glove = bool(sec_types.intersection({"SL", "ST", "CU", "FC"}))
    has_arm = bool(sec_types.intersection({"CH", "FS", "SI"}))
    raw_platoon_bonus = 1.8 if (has_glove and has_arm) else 0.5

    tunnel_bonus = round(gamma_anchor * raw_tunnel_bonus, 2)
    evd_bonus = round(gamma_anchor * raw_evd_bonus, 2)
    mirror_bonus = round(gamma_anchor * raw_mirror_bonus, 2)
    platoon_bonus = round(gamma_anchor * raw_platoon_bonus, 2)

    arsenal_plus = round(
        base_weighted + tunnel_bonus + evd_bonus + mirror_bonus + platoon_bonus,
        1,
    )

    return ArsenalSynergyReport(
        arsenal_plus=arsenal_plus,
        fastball_stuff_plus=round(fb_stuff, 1),
        secondary_weighted_stuff_plus=round(sec_stuff, 1),
        gamma_anchor=gamma_anchor,
        tunnel_bonus=tunnel_bonus,
        effective_velocity_dispersion_bonus=evd_bonus,
        mirror_spin_bonus=mirror_bonus,
        platoon_coverage_bonus=platoon_bonus,
        scouting_20_80_grade=stuff_plus_to_20_80_scale(arsenal_plus),
    )
