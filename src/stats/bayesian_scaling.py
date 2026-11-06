"""Empirical Bayes Small-Sample Bullpen Shrinkage & Confidence Interval Engine."""

from dataclasses import dataclass
import math


# Pitch-type stabilization constants k_pitch (in pitches) derived from intraclass correlation (ICC)
STABILIZATION_K_PITCHES = {
    "FF": 4.0,
    "SI": 4.5,
    "FC": 5.5,
    "SL": 6.5,
    "ST": 7.0,
    "CU": 7.5,
    "CH": 8.0,
    "FS": 8.5,
}


@dataclass
class BayesianStuffEstimate:
    pitch_type: str
    sample_size_n: int
    raw_stuff_plus: float
    prior_mean_stuff_plus: float
    posterior_stuff_plus: float
    posterior_sd: float
    ci_90_lower: float
    ci_90_upper: float
    reliability_weight: float


def shrink_bullpen_stuff_plus(
    pitch_type: str,
    raw_stuff_plus: float,
    sample_size_n: int,
    prior_mean: float = 92.0,
    population_sd: float = 10.0,
) -> BayesianStuffEstimate:
    """
    Shrink small-sample bullpen Stuff+ grades toward the demographic prior (92.0 for HS Varsity
    relative to NCAA D1 = 100.0) using Empirical Bayes reliability weighting:
        Stuff+_post = (n / (n + k)) * Stuff+_raw + (k / (n + k)) * mu_prior
    """
    n = max(1, int(sample_size_n))
    k = STABILIZATION_K_PITCHES.get(pitch_type.upper(), 6.0)
    w = n / (n + k)
    posterior_mean = w * raw_stuff_plus + (1.0 - w) * prior_mean
    posterior_sd = population_sd * math.sqrt(1.0 - w * 0.72)
    margin_90 = 1.645 * posterior_sd

    return BayesianStuffEstimate(
        pitch_type=pitch_type.upper(),
        sample_size_n=n,
        raw_stuff_plus=round(raw_stuff_plus, 1),
        prior_mean_stuff_plus=round(prior_mean, 1),
        posterior_stuff_plus=round(posterior_mean, 1),
        posterior_sd=round(posterior_sd, 2),
        ci_90_lower=round(posterior_mean - margin_90, 1),
        ci_90_upper=round(posterior_mean + margin_90, 1),
        reliability_weight=round(w, 3),
    )
