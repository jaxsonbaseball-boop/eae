"""Automated 3-Sentence Plain-English Scouting Director Narrative Generator (v3.5)."""

from typing import Dict, List


def generate_scouting_director_summary(
    pitcher_name: str,
    pitcher_throws: str,
    arsenal_plus: float,
    primary_fb_velo_mph: float,
    primary_fb_ivb_in: float,
    primary_fb_stuff_plus: float,
    best_secondary_name: str,
    best_secondary_stuff_plus: float,
    college_growth_mode: bool = False,
) -> str:
    """
    Generate a concise, reality-grounded 3-sentence scouting summary suitable for High School
    coaches, parents, and NCAA Division 1 recruiting coordinators.
    """
    mode_label = (
        "under a 3-Year College Strength Projection (+5.0 mph FB / +3.5 mph offspeed)"
        if college_growth_mode
        else "against Present NCAA Division 1 competition"
    )

    if primary_fb_velo_mph < 88.0:
        fb_clause = (
            f"{pitcher_name} ({pitcher_throws}) currently sits {primary_fb_velo_mph:.1f} mph on the four-seam fastball "
            f"with {primary_fb_ivb_in:.1f}" of vertical ride (D1 Stuff+ {primary_fb_stuff_plus:.0f}), meaning the fastball "
            f"requires college strength gains to cross the 88–90 mph Division 1 anchor threshold."
        )
    else:
        fb_clause = (
            f"{pitcher_name} ({pitcher_throws}) anchors the repertoire with an NCAA-viable {primary_fb_velo_mph:.1f} mph "
            f"fastball generating {primary_fb_ivb_in:.1f}" of induced vertical break (D1 Stuff+ {primary_fb_stuff_plus:.0f})."
        )

    sec_clause = (
        f"The top Put-Away weapon is the {best_secondary_name} (D1 Stuff+ {best_secondary_stuff_plus:.0f}), "
        f"which tunnels tightly out of the hand before separating late across the hitting zone."
    )

    overall_clause = (
        f"Overall, the repertoire grades at {arsenal_plus:.0f} Arsenal+ {mode_label}, "
        f"prioritizing lower-half velocity development and consistent fastball plane at the top of the strike zone."
    )

    return f"{fb_clause} {sec_clause} {overall_clause}"
