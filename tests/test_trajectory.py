"""Unit tests for Project Apex 3D trajectory physics, NCAA D1 Stuff+ calibration, and Arsenal+."""

import unittest

from src.physics.trajectory import (
    EnvironmentalConditions,
    integrate_3d_pitch_trajectory,
    shoulder_slot_from_angle,
)
from src.models.college_stuff_engine import (
    compute_fastball_anchor_viability,
    score_pitch_stuff_plus,
)
from src.stats.arsenal_synergy import compute_arsenal_plus
from src.stats.bayesian_scaling import shrink_bullpen_stuff_plus


class TestProjectApexEngine(unittest.TestCase):
    def test_3d_trajectory_drag_decay_and_plate_crossing(self) -> None:
        res = integrate_3d_pitch_trajectory(
            rel_speed_mph=91.5,
            spin_rate_rpm=2260.0,
            spin_axis_deg=210.0,
            spin_efficiency_pct=92.0,
            extension_ft=6.15,
            rel_height_ft=5.85,
            rel_side_ft=1.85,
            plate_x_target_ft=0.0,
            plate_z_target_ft=2.50,
            ivb_target_in=15.8,
            hb_target_in=8.5,
        )
        self.assertGreaterEqual(len(res.trajectory_points_ft), 44)
        self.assertGreater(res.drag_decay_pct, 7.5)
        self.assertLess(res.drag_decay_pct, 11.5)
        self.assertAlmostEqual(res.plate_x_ft, 0.0, delta=0.25)
        self.assertAlmostEqual(res.plate_z_ft, 2.50, delta=0.25)

    def test_altitude_normalization(self) -> None:
        sea_level = EnvironmentalConditions(temp_f=70.0, elevation_ft=0.0)
        high_alt = EnvironmentalConditions(temp_f=85.0, elevation_ft=5280.0)
        self.assertLess(high_alt.air_density_kg_m3(), sea_level.air_density_kg_m3())

        res_high = integrate_3d_pitch_trajectory(
            rel_speed_mph=90.0,
            spin_rate_rpm=2300.0,
            spin_axis_deg=205.0,
            ivb_target_in=14.0,
            hb_target_in=8.0,
            env=high_alt,
        )
        # Sea-level normalized IVB must exceed thin-air raw IVB
        self.assertGreater(res_high.ivb_norm_in, res_high.ivb_raw_in)

    def test_sub88_fastball_velocity_cliff_and_college_growth(self) -> None:
        # 84.2 mph HS Fastball (Present D1 Reality) should grade in the 62-75 range due to sub-88 cliff
        present_hs_fb = score_pitch_stuff_plus(
            pitch_type="FF",
            rel_speed_mph=84.2,
            ivb_norm_in=16.4,
            hb_norm_in=7.8,
            spin_rate_rpm=2240.0,
            spin_efficiency_pct=93.0,
            extension_ft=6.10,
            rel_height_ft=5.75,
            rel_side_ft=1.80,
            vaa_deg=-4.45,
            college_growth_projection=False,
        )
        self.assertLess(present_hs_fb.raw_stuff_plus, 78.0)
        self.assertLess(present_hs_fb.sub88_cliff_penalty, -3.0)

        # Same pitch under 3-Year College Growth (+5.0 mph -> 89.2 mph) jumps by +20 Stuff+ points
        projected_fb = score_pitch_stuff_plus(
            pitch_type="FF",
            rel_speed_mph=84.2,
            ivb_norm_in=16.4,
            hb_norm_in=7.8,
            spin_rate_rpm=2240.0,
            spin_efficiency_pct=93.0,
            extension_ft=6.10,
            rel_height_ft=5.75,
            rel_side_ft=1.80,
            vaa_deg=-4.45,
            college_growth_projection=True,
        )
        self.assertGreater(projected_fb.raw_stuff_plus - present_hs_fb.raw_stuff_plus, 18.0)

    def test_fastball_anchor_viability_and_arsenal_plus(self) -> None:
        gamma_84 = compute_fastball_anchor_viability(84.2)
        gamma_92 = compute_fastball_anchor_viability(92.0)
        self.assertAlmostEqual(gamma_84, 0.5167, places=3)
        self.assertEqual(gamma_92, 1.0)

        arsenal = compute_arsenal_plus([
            {"pitch_type": "FF", "rel_speed_mph": 84.2, "stuff_plus": 68.0},
            {"pitch_type": "SL", "rel_speed_mph": 75.8, "stuff_plus": 96.0},
            {"pitch_type": "CH", "rel_speed_mph": 76.5, "stuff_plus": 94.0},
        ])
        self.assertGreater(arsenal.arsenal_plus, 78.0)
        self.assertLess(arsenal.arsenal_plus, 92.0)

    def test_bayesian_shrinkage_and_shoulder_arc(self) -> None:
        est = shrink_bullpen_stuff_plus("SL", raw_stuff_plus=112.0, sample_size_n=5, prior_mean=92.0)
        self.assertGreater(est.posterior_stuff_plus, 92.0)
        self.assertLess(est.posterior_stuff_plus, 112.0)

        side_ft, height_ft = shoulder_slot_from_angle(38.0, "RHP")
        self.assertGreater(side_ft, 1.5)
        self.assertGreater(height_ft, 6.0)


if __name__ == "__main__":
    unittest.main()
