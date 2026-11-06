"""Physical constants, TrackMan B1 coordinate definitions, and NCAA Division 1 baselines for Project Apex (v3.5)."""

from dataclasses import dataclass
from typing import Dict

# Baseball physical parameters (Regulation NCAA / Rawlings baseball)
MASS_BALL_KG: float = 0.145
RADIUS_BALL_M: float = 0.0366
CIRCUMFERENCE_BALL_M: float = 0.230
CROSS_SECTION_AREA_M2: float = 0.004208

# Unit conversion factors
MPH_TO_MPS: float = 0.44704
MPS_TO_MPH: float = 2.23693629
FT_TO_M: float = 0.3048
M_TO_FT: float = 3.2808399
IN_TO_M: float = 0.0254
M_TO_IN: float = 39.3700787
DEG_TO_RAD: float = 0.017453292519943295
RAD_TO_DEG: float = 57.29577951308232

# Mound and plate geometry (TrackMan B1 convention: Y=0 at back tip of home plate, +Y toward pitcher)
MOUND_DISTANCE_FT: float = 60.5
FRONT_OF_PLATE_FT: float = 17.0 / 12.0  # 1.4167 ft

# Standard gravitational acceleration (m/s^2)
GRAVITY_MPS2: float = 9.80665

# Standard sea-level environmental reference (70 F, 0 ft elevation, 50% RH, 29.92 inHg)
STANDARD_TEMP_F: float = 70.0
STANDARD_ELEVATION_FT: float = 0.0
STANDARD_HUMIDITY_PCT: float = 50.0
STANDARD_PRESSURE_INHG: float = 29.92
RHO_STANDARD_KG_M3: float = 1.1965

# Aerodynamic coefficients (Nathan 2008 / 2021 TrackMan calibration)
CD_0: float = 0.3008
CD_SPIN_SLOPE: float = 0.0292

# NCAA Division 1 Pitch-Type Baselines (Stuff+ = 100, SD = 10)
@dataclass(frozen=True)
class D1PitchBaseline:
    pitch_type: str
    velo_mph: float
    velo_sd: float
    ivb_in: float
    hb_in: float
    spin_rpm: float
    extension_ft: float
    rel_height_ft: float
    vaa_deg: float

NCAA_D1_BASELINES: Dict[str, D1PitchBaseline] = {
    "FF": D1PitchBaseline("FF", velo_mph=91.5, velo_sd=2.45, ivb_in=15.8, hb_in=8.5, spin_rpm=2260.0, extension_ft=6.15, rel_height_ft=5.85, vaa_deg=-4.85),
    "SI": D1PitchBaseline("SI", velo_mph=90.8, velo_sd=2.40, ivb_in=8.5, hb_in=15.2, spin_rpm=2140.0, extension_ft=6.15, rel_height_ft=5.70, vaa_deg=-5.45),
    "FC": D1PitchBaseline("FC", velo_mph=86.5, velo_sd=2.30, ivb_in=7.5, hb_in=-3.2, spin_rpm=2380.0, extension_ft=6.10, rel_height_ft=5.80, vaa_deg=-5.60),
    "SL": D1PitchBaseline("SL", velo_mph=81.5, velo_sd=2.55, ivb_in=1.2, hb_in=-6.5, spin_rpm=2460.0, extension_ft=6.05, rel_height_ft=5.75, vaa_deg=-6.80),
    "ST": D1PitchBaseline("ST", velo_mph=79.8, velo_sd=2.60, ivb_in=1.5, hb_in=-14.5, spin_rpm=2580.0, extension_ft=6.05, rel_height_ft=5.65, vaa_deg=-7.10),
    "CU": D1PitchBaseline("CU", velo_mph=77.2, velo_sd=2.70, ivb_in=-11.5, hb_in=-8.5, spin_rpm=2520.0, extension_ft=6.00, rel_height_ft=5.95, vaa_deg=-8.90),
    "CH": D1PitchBaseline("CH", velo_mph=82.2, velo_sd=2.50, ivb_in=6.2, hb_in=14.2, spin_rpm=1740.0, extension_ft=6.15, rel_height_ft=5.75, vaa_deg=-6.40),
    "FS": D1PitchBaseline("FS", velo_mph=83.5, velo_sd=2.50, ivb_in=3.0, hb_in=11.8, spin_rpm=1380.0, extension_ft=6.15, rel_height_ft=5.80, vaa_deg=-6.65),
}

# Shoulder-arc biomechanical coupling radius (ft) around C7/T1 pivot (x_p = 0.45 ft, z_p = 5.05 ft)
SHOULDER_ARC_RADIUS_FT: float = 2.40
SHOULDER_PIVOT_X_FT: float = 0.45
SHOULDER_PIVOT_Z_FT: float = 5.05
