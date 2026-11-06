"""3D RK45 and Heun Trajectory Integrator with Nathan Magnus/Drag Environmental Normalization."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.physics.constants import (
    CD_0,
    CD_SPIN_SLOPE,
    CROSS_SECTION_AREA_M2,
    DEG_TO_RAD,
    FRONT_OF_PLATE_FT,
    FT_TO_M,
    GRAVITY_MPS2,
    IN_TO_M,
    M_TO_FT,
    M_TO_IN,
    MASS_BALL_KG,
    MOUND_DISTANCE_FT,
    MPH_TO_MPS,
    MPS_TO_MPH,
    RAD_TO_DEG,
    RADIUS_BALL_M,
    RHO_STANDARD_KG_M3,
    SHOULDER_ARC_RADIUS_FT,
    SHOULDER_PIVOT_X_FT,
    SHOULDER_PIVOT_Z_FT,
)


@dataclass
class EnvironmentalConditions:
    """Bullpen or stadium environmental conditions for air density normalization."""
    temp_f: float = 70.0
    elevation_ft: float = 0.0
    humidity_pct: float = 50.0
    pressure_inhg: float = 29.92

    def air_density_kg_m3(self) -> float:
        """Compute moist air density (kg/m^3) using the Tetens saturation vapor pressure formula."""
        temp_c = (self.temp_f - 32.0) * (5.0 / 9.0)
        temp_k = temp_c + 273.15
        elev_m = self.elevation_ft * FT_TO_M
        p_sea_pa = self.pressure_inhg * 3386.389
        p_local_pa = p_sea_pa * math.exp(-0.0001185 * elev_m)
        p_sat_pa = 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
        p_v = (self.humidity_pct / 100.0) * p_sat_pa
        p_d = max(0.0, p_local_pa - p_v)
        r_dry = 287.058
        r_vapor = 461.495
        return (p_d / (r_dry * temp_k)) + (p_v / (r_vapor * temp_k))


@dataclass
class TrajectoryResult:
    """Full 3D trajectory output from release point to the front of home plate (y = 1.4167 ft)."""
    t_flight_s: float
    plate_speed_mph: float
    drag_decay_pct: float
    ivb_raw_in: float
    hb_raw_in: float
    ivb_norm_in: float
    hb_norm_in: float
    vaa_deg: float
    haa_deg: float
    effective_velo_mph: float
    reaction_margin_ms: float
    plate_x_ft: float
    plate_z_ft: float
    trajectory_points_ft: List[Dict[str, float]]


def lift_coefficient_nathan(spin_factor: float) -> float:
    """Alan Nathan's two-regime piecewise lift coefficient C_L(S)."""
    if spin_factor <= 0.0:
        return 0.0
    if spin_factor <= 0.10:
        return 1.50 * spin_factor
    return 0.09 + 0.60 * spin_factor


def drag_coefficient_nathan(spin_factor: float) -> float:
    """Alan Nathan's spin-dependent drag coefficient C_D(S)."""
    return CD_0 + CD_SPIN_SLOPE * spin_factor


def shoulder_slot_from_angle(arm_angle_deg: float, handedness: str = "RHP") -> Tuple[float, float]:
    """Compute biomechanically coupled (RelSide, RelHeight) on the shoulder arc (r = 2.40 ft)."""
    theta = arm_angle_deg * DEG_TO_RAD
    sign = 1.0 if handedness.upper().startswith("R") else -1.0
    rel_side_ft = sign * (SHOULDER_PIVOT_X_FT + SHOULDER_ARC_RADIUS_FT * math.cos(theta))
    rel_height_ft = SHOULDER_PIVOT_Z_FT + SHOULDER_ARC_RADIUS_FT * math.sin(theta)
    return rel_side_ft, rel_height_ft


def integrate_3d_pitch_trajectory(
    rel_speed_mph: float,
    spin_rate_rpm: float,
    spin_axis_deg: float,
    spin_efficiency_pct: float = 90.0,
    extension_ft: float = 6.0,
    rel_height_ft: float = 5.8,
    rel_side_ft: float = 1.8,
    plate_x_target_ft: float = 0.0,
    plate_z_target_ft: float = 2.5,
    ivb_target_in: Optional[float] = None,
    hb_target_in: Optional[float] = None,
    env: Optional[EnvironmentalConditions] = None,
    n_steps: int = 45,
) -> TrajectoryResult:
    """
    Integrate 3D pitch trajectory using a 45-step Heun predictor-corrector solver (matching the
    WebGL engine) with full aerodynamic drag and Magnus acceleration, plus standard sea-level
    movement normalization.
    """
    if env is None:
        env = EnvironmentalConditions()

    rho = env.air_density_kg_m3()
    rho_ratio = RHO_STANDARD_KG_M3 / max(0.5, rho)

    y_start_ft = MOUND_DISTANCE_FT - extension_ft
    y_end_ft = FRONT_OF_PLATE_FT
    flight_dist_ft = y_start_ft - y_end_ft
    flight_dist_m = flight_dist_ft * FT_TO_M

    v0_mps = rel_speed_mph * MPH_TO_MPS
    omega_rad_s = (spin_rate_rpm * 2.0 * math.pi / 60.0) * (spin_efficiency_pct / 100.0)
    spin_factor = (RADIUS_BALL_M * omega_rad_s) / max(1.0, v0_mps)

    c_d = drag_coefficient_nathan(spin_factor)
    c_l = lift_coefficient_nathan(spin_factor)

    const_aero = (0.5 * rho * CROSS_SECTION_AREA_M2) / MASS_BALL_KG

    # Spin axis in TrackMan tilt convention: 180 deg = pure backspin (+Z), 90 deg = RHP tail (+X)
    tilt_rad = spin_axis_deg * DEG_TO_RAD
    # Movement unit components in (X, Z) plane
    m_x_unit = math.sin(tilt_rad)
    m_z_unit = -math.cos(tilt_rad)

    # Approximate flight time to solve initial release angles toward target (plate_x, plate_z)
    avg_speed_mps = v0_mps * 0.948
    t_est = flight_dist_m / avg_speed_mps

    if ivb_target_in is not None and hb_target_in is not None:
        # Use observed TrackMan B1 movement scaled by air density if calibrating directly
        a_mag_x = (2.0 * (hb_target_in * IN_TO_M)) / (t_est * t_est) * (rho / RHO_STANDARD_KG_M3)
        a_mag_z = (2.0 * (ivb_target_in * IN_TO_M)) / (t_est * t_est) * (rho / RHO_STANDARD_KG_M3)
    else:
        a_mag_total = const_aero * c_l * (avg_speed_mps ** 2)
        a_mag_x = a_mag_total * m_x_unit
        a_mag_z = a_mag_total * m_z_unit

    # Initial position in meters: x0, y0, z0
    x0_m = rel_side_ft * FT_TO_M
    y0_m = y_start_ft * FT_TO_M
    z0_m = rel_height_ft * FT_TO_M

    x_target_m = plate_x_target_ft * FT_TO_M
    z_target_m = plate_z_target_ft * FT_TO_M

    # Solve initial velocity vector so pitch arrives at (plate_x_target_ft, plate_z_target_ft)
    vx0 = ((x_target_m - x0_m) - 0.5 * a_mag_x * (t_est ** 2)) / t_est
    vz0 = ((z_target_m - z0_m) + 0.5 * GRAVITY_MPS2 * (t_est ** 2) - 0.5 * a_mag_z * (t_est ** 2)) / t_est
    vy0 = -math.sqrt(max(10.0, v0_mps ** 2 - vx0 ** 2 - vz0 ** 2))

    state = np.array([x0_m, y0_m, z0_m, vx0, vy0, vz0], dtype=float)
    dt = t_est / float(n_steps)

    def accel(st: np.ndarray) -> np.ndarray:
        vx, vy, vz = st[3], st[4], st[5]
        v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        s_local = (RADIUS_BALL_M * omega_rad_s) / max(1.0, v_norm)
        cd_local = drag_coefficient_nathan(s_local)
        drag_mag = const_aero * cd_local * v_norm
        ax = -drag_mag * vx + a_mag_x
        ay = -drag_mag * vy
        az = -drag_mag * vz - GRAVITY_MPS2 + a_mag_z
        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

    points_ft: List[Dict[str, float]] = []
    t_curr = 0.0

    for step in range(n_steps + 1):
        points_ft.append({
            "step": float(step),
            "t_s": round(t_curr, 5),
            "x_ft": round(float(state[0] * M_TO_FT), 4),
            "y_ft": round(float(state[1] * M_TO_FT), 4),
            "z_ft": round(float(state[2] * M_TO_FT), 4),
            "speed_mph": round(float(np.linalg.norm(state[3:6]) * MPS_TO_MPH), 2),
        })
        if step == n_steps:
            break
        # Heun predictor-corrector step (RK2)
        k1 = accel(state)
        pred = state + dt * k1
        k2 = accel(pred)
        state = state + 0.5 * dt * (k1 + k2)
        t_curr += dt
        if state[1] * M_TO_FT <= y_end_ft:
            break

    # Gravity-only trajectory from identical release velocity vector for true spin-induced break
    x_grav_m = x0_m + vx0 * t_curr
    z_grav_m = z0_m + vz0 * t_curr - 0.5 * GRAVITY_MPS2 * (t_curr ** 2)

    hb_raw_in = (state[0] - x_grav_m) * M_TO_IN
    ivb_raw_in = (state[2] - z_grav_m) * M_TO_IN

    # Environmental normalization to sea-level standard density (70 F, 0 ft)
    hb_norm_in = hb_raw_in * rho_ratio
    ivb_norm_in = ivb_raw_in * rho_ratio

    vx_f, vy_f, vz_f = float(state[3]), float(state[4]), float(state[5])
    v_plate_mps = math.sqrt(vx_f * vx_f + vy_f * vy_f + vz_f * vz_f)
    plate_speed_mph = v_plate_mps * MPS_TO_MPH
    drag_decay_pct = ((rel_speed_mph - plate_speed_mph) / max(1.0, rel_speed_mph)) * 100.0

    vaa_deg = math.atan2(vz_f, abs(vy_f)) * RAD_TO_DEG
    haa_deg = math.atan2(vx_f, abs(vy_f)) * RAD_TO_DEG

    # Effective velocity & batter reaction margin vs NCAA D1 average extension (6.15 ft)
    effective_velo_mph = rel_speed_mph + (extension_ft - 6.15) * 1.70
    d1_ref_flight_ms = ((MOUND_DISTANCE_FT - 6.15 - FRONT_OF_PLATE_FT) * FT_TO_M / (91.5 * MPH_TO_MPS * 0.948)) * 1000.0
    pitch_flight_ms = t_curr * 1000.0
    reaction_margin_ms = pitch_flight_ms - d1_ref_flight_ms

    return TrajectoryResult(
        t_flight_s=round(t_curr, 4),
        plate_speed_mph=round(plate_speed_mph, 2),
        drag_decay_pct=round(drag_decay_pct, 2),
        ivb_raw_in=round(ivb_raw_in, 2),
        hb_raw_in=round(hb_raw_in, 2),
        ivb_norm_in=round(ivb_norm_in, 2),
        hb_norm_in=round(hb_norm_in, 2),
        vaa_deg=round(vaa_deg, 2),
        haa_deg=round(haa_deg, 2),
        effective_velo_mph=round(effective_velo_mph, 2),
        reaction_margin_ms=round(reaction_margin_ms, 1),
        plate_x_ft=round(float(state[0] * M_TO_FT), 3),
        plate_z_ft=round(float(state[2] * M_TO_FT), 3),
        trajectory_points_ft=points_ft,
    )
