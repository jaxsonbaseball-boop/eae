"""
EAE Phase 2B-1: Blended Empirical MiLB + Calibrated Synthetic NCAA D1 Stuff+ Trainer.

Blends:
  - Empirical post-draft 4-year college draftee pitches (`data/processed/milb_postdraft_pitches.csv`,
    sample_weight = 0.75) with tier-specific whiff calibration offsets:
      * sportId = 16 (Complex/Rookie): -0.045
      * sportId = 14 (Single-A/Low-A): -0.030
      * sportId = 13 (High-A):         -0.025
  - Calibrated synthetic NCAA D1 dataset (76-93 mph coverage, sample_weight = 0.35)
    so the 86-93 mph overlap bridge transitions smoothly without velocity step discontinuities.

Anchors `mu_college` and `sigma_college` (100 = Average D1, SD = 10) strictly to the
NCAA D1 reference distribution (`cohort == 'college'`) and exports
`data/processed/ncaa_d1_stuff_model.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from src.models.college_stuff_engine import (
    CANONICAL_PITCH_FAMILIES,
    CollegeStuffEngine,
    FourHeadPitchFamilyModel,
    HeadValidationMetrics,
    MODEL_FEATURE_COLUMNS,
    PitchFamilyBaselineStats,
    batch_neutralize_location_xrv,
    build_monotonic_constraints,
    compute_expected_rv_from_heads,
    engineer_pitch_features,
    ingest_trackman_dataframe,
)
from src.models.export_for_web import export_model_for_web
from tests.test_college_stuff_engine import generate_synthetic_ncaa_d1_dataset

logger = logging.getLogger("eae.train_blended_stuff_model")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TIER_WHIFF_OFFSETS: Dict[int, float] = {
    16: -0.045,  # Complex / Rookie
    14: -0.030,  # Single-A / Low-A
    13: -0.025,  # High-A
}

EMPIRICAL_SAMPLE_WEIGHT: float = 0.75
SYNTHETIC_D1_SAMPLE_WEIGHT: float = 0.35


def apply_whiff_calibration_offsets(
    df_milb: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply tier-specific whiff-probability calibration offsets (-0.045 Complex,
    -0.030 Single-A, -0.025 High-A) to empirical MiLB swings so pro/complex
    hitter chase rates do not inflate NCAA D1 baselines.
    """
    out = df_milb.copy()
    rng = np.random.default_rng(seed)
    swing_mask = out["is_swing"].to_numpy() == 1
    whiff_arr = out["is_whiff"].to_numpy(dtype=np.int32).copy()
    sport_arr = out["sport_id"].to_numpy(dtype=np.int32)

    for sid, offset in TIER_WHIFF_OFFSETS.items():
        if offset >= 0:
            continue
        drop_prob = abs(offset)
        cand_mask = swing_mask & (whiff_arr == 1) & (sport_arr == sid)
        if np.any(cand_mask):
            flips = rng.random(int(np.sum(cand_mask))) < drop_prob
            cand_indices = np.flatnonzero(cand_mask)
            whiff_arr[cand_indices[flips]] = 0

    out["is_whiff"] = whiff_arr
    return out


class BlendedCollegeStuffEngine(CollegeStuffEngine):
    """
    Extends `CollegeStuffEngine` with smooth tree regularization, `sample_weight`
    blending (0.75 empirical MiLB / 0.35 synthetic D1), and strict college-cohort
    100-mean / 10-SD calibration.
    """

    def _make_classifier(self, monotonic_constraints: Tuple[int, ...]) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.82,
            colsample_bytree=0.82,
            min_child_weight=12,
            reg_lambda=4.0,
            reg_alpha=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
            monotonic_constraints=monotonic_constraints,
        )

    def _make_regressor(self, monotonic_constraints: Tuple[int, ...]) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.82,
            colsample_bytree=0.82,
            min_child_weight=12,
            reg_lambda=4.0,
            reg_alpha=0.5,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
            monotonic_constraints=monotonic_constraints,
        )

    def fit_blended_family(
        self,
        df_family: pd.DataFrame,
        pitch_family: str,
    ) -> FourHeadPitchFamilyModel:
        pt = pitch_family.upper()
        df_f = df_family.reset_index(drop=True)
        n_rows = len(df_f)
        unique_pitchers = df_f["PitcherId"].nunique()
        effective_splits = min(self.n_splits, max(2, unique_pitchers))

        X_all = df_f[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        groups = df_f["PitcherId"].to_numpy()
        weights = df_f["sample_weight"].to_numpy(dtype=np.float64)
        is_college = df_f["cohort"].to_numpy() == "college"

        y_swing = df_f["is_swing"].to_numpy(dtype=np.int32)
        y_whiff = df_f["is_whiff"].to_numpy(dtype=np.int32)
        y_cs = df_f["is_called_strike"].to_numpy(dtype=np.int32)
        y_xw = df_f["target_xwobacon"].to_numpy(dtype=np.float64)
        is_contact = df_f["is_contact"].to_numpy(dtype=np.int32)

        mc_swing = build_monotonic_constraints(pt, "swing")
        mc_whiff = build_monotonic_constraints(pt, "whiff")
        mc_cs = build_monotonic_constraints(pt, "called_strike")
        mc_xw = build_monotonic_constraints(pt, "xwobacon")

        oof_neutral_xrv = np.full(n_rows, np.nan, dtype=np.float64)
        gkf = GroupKFold(n_splits=effective_splits)

        for train_idx, val_idx in gkf.split(X_all, y_swing, groups=groups):
            X_tr, X_val = X_all[train_idx], X_all[val_idx]
            w_tr = weights[train_idx]

            m_swing = self._make_classifier(mc_swing)
            m_swing.fit(X_tr, y_swing[train_idx], sample_weight=w_tr)

            swing_tr = y_swing[train_idx] == 1
            m_whiff = self._make_classifier(mc_whiff)
            if np.sum(swing_tr) >= 10 and len(np.unique(y_whiff[train_idx][swing_tr])) >= 2:
                m_whiff.fit(X_tr[swing_tr], y_whiff[train_idx][swing_tr], sample_weight=w_tr[swing_tr])
            else:
                m_whiff.fit(X_tr, y_whiff[train_idx], sample_weight=w_tr)

            take_tr = y_swing[train_idx] == 0
            m_cs = self._make_classifier(mc_cs)
            if np.sum(take_tr) >= 10 and len(np.unique(y_cs[train_idx][take_tr])) >= 2:
                m_cs.fit(X_tr[take_tr], y_cs[train_idx][take_tr], sample_weight=w_tr[take_tr])
            else:
                m_cs.fit(X_tr, y_cs[train_idx], sample_weight=w_tr)

            cont_tr = is_contact[train_idx] == 1
            m_xw = self._make_regressor(mc_xw)
            if np.sum(cont_tr) >= 10:
                m_xw.fit(X_tr[cont_tr], y_xw[train_idx][cont_tr], sample_weight=w_tr[cont_tr])
            else:
                m_xw.fit(X_tr, y_xw[train_idx], sample_weight=w_tr)

            oof_neutral_xrv[val_idx] = batch_neutralize_location_xrv(
                head_swing=m_swing,
                head_whiff=m_whiff,
                head_called_strike=m_cs,
                head_xwobacon=m_xw,
                df_features=df_f.iloc[val_idx],
            )

        # Anchor mu_college & sigma_college STRICTLY to NCAA D1 reference distribution
        college_xrv = oof_neutral_xrv[is_college] if np.sum(is_college) >= 20 else oof_neutral_xrv
        mu_college = float(np.nanmean(college_xrv))
        sigma_college = float(max(1e-5, np.nanstd(college_xrv, ddof=1)))

        # Fit final production heads on full blended sample
        swing_mask = y_swing == 1
        take_mask = y_swing == 0
        contact_mask = is_contact == 1

        final_swing = self._make_classifier(mc_swing)
        final_swing.fit(X_all, y_swing, sample_weight=weights)

        final_whiff = self._make_classifier(mc_whiff)
        if np.sum(swing_mask) >= 10 and len(np.unique(y_whiff[swing_mask])) >= 2:
            final_whiff.fit(X_all[swing_mask], y_whiff[swing_mask], sample_weight=weights[swing_mask])
        else:
            final_whiff.fit(X_all, y_whiff, sample_weight=weights)

        final_cs = self._make_classifier(mc_cs)
        if np.sum(take_mask) >= 10 and len(np.unique(y_cs[take_mask])) >= 2:
            final_cs.fit(X_all[take_mask], y_cs[take_mask], sample_weight=weights[take_mask])
        else:
            final_cs.fit(X_all, y_cs, sample_weight=weights)

        final_xw = self._make_regressor(mc_xw)
        if np.sum(contact_mask) >= 10:
            final_xw.fit(X_all[contact_mask], y_xw[contact_mask], sample_weight=weights[contact_mask])
        else:
            final_xw.fit(X_all, y_xw, sample_weight=weights)

        ref_df = df_f.loc[is_college] if np.sum(is_college) >= 20 else df_f
        baseline = PitchFamilyBaselineStats(
            pitch_family=pt,
            count=len(ref_df),
            mu_college_xrv=mu_college,
            sigma_college_xrv=sigma_college,
            velo_mean=float(ref_df["RelSpeed"].mean()),
            velo_sd=float(max(0.5, ref_df["RelSpeed"].std(ddof=1))),
            ivb_mean=float(ref_df["InducedVertBreak"].mean()),
            ivb_sd=float(max(0.5, ref_df["InducedVertBreak"].std(ddof=1))),
            hb_mean=float(ref_df["HB_norm"].mean()),
            hb_sd=float(max(0.5, ref_df["HB_norm"].std(ddof=1))),
            ext_mean=float(ref_df["Extension"].mean()),
            ext_sd=float(max(0.1, ref_df["Extension"].std(ddof=1))),
            vaa_mean=float(ref_df["VertApprAngle"].mean()) if "VertApprAngle" in ref_df else -5.0,
            vaa_sd=float(max(0.2, ref_df["VertApprAngle"].std(ddof=1))) if "VertApprAngle" in ref_df else 1.0,
            spin_mean=float(ref_df["SpinRate"].mean()),
            spin_sd=float(max(50.0, ref_df["SpinRate"].std(ddof=1))),
        )
        metrics = HeadValidationMetrics(
            pitch_family=pt,
            sample_size=n_rows,
            n_pitchers=unique_pitchers,
            swing_roc_auc=0.0,
            whiff_roc_auc=0.0,
            called_strike_roc_auc=0.0,
            xwobacon_rmse=0.0,
            xwobacon_r2=0.0,
            mu_college_xrv=mu_college,
            sigma_college_xrv=sigma_college,
        )
        return FourHeadPitchFamilyModel(
            pitch_family=pt,
            feature_names=tuple(getattr(self, "feature_cols", []) or [c for c in df_f.columns if c in ("RelSpeed","InducedVertBreak","HB_norm","SpinRate","Extension","VertApprAngle","HorzApprAngle","RelHeight","RelSide_norm","vel_diff_fb","ivb_diff_fb","hb_diff_fb")]),
            head_swing=final_swing,
            head_whiff=final_whiff,
            head_called_strike=final_cs,
            head_xwobacon=final_xw,
            metrics=metrics,
            baseline_stats=baseline,
        )

    def fit_blended(
        self,
        df_synthetic_raw: pd.DataFrame,
        df_milb_processed: Optional[pd.DataFrame] = None,
    ) -> "BlendedCollegeStuffEngine":
        df_syn_clean = ingest_trackman_dataframe(df_synthetic_raw, min_pitches_per_pitcher=50)
        df_syn_feat = engineer_pitch_features(df_syn_clean)
        df_syn_feat["cohort"] = "college"
        df_syn_feat["sample_weight"] = SYNTHETIC_D1_SAMPLE_WEIGHT

        frames = [df_syn_feat]
        if df_milb_processed is not None and not df_milb_processed.empty:
            df_milb = apply_whiff_calibration_offsets(df_milb_processed)
            df_milb["cohort"] = "milb_4yr"
            df_milb["sample_weight"] = EMPIRICAL_SAMPLE_WEIGHT
            frames.append(df_milb)

        df_all = pd.concat(frames, ignore_index=True)
        self.training_sample_size = len(df_all)

        for family in CANONICAL_PITCH_FAMILIES:
            sub = df_all.loc[df_all["PitchFamily"] == family]
            if len(sub) < 40 or sub["PitcherId"].nunique() < 2:
                continue
            self.family_models[family] = self.fit_blended_family(sub, pitch_family=family)

        if not self.family_models:
            raise RuntimeError("No pitch families met minimum sample size threshold.")
        return self


def verify_monotonic_ff_progression(engine: BlendedCollegeStuffEngine) -> List[Tuple[float, float]]:
    """
    Verify smooth monotonic Stuff+ progression across 80..98 mph FF (17.5" IVB, 6.2 ft ext)
    with no artificial jump across the 86-93 mph synthetic-to-empirical overlap bridge.
    """
    velos = [80.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0]
    progression: List[Tuple[float, float]] = []
    for v in velos:
        res = engine.score_pitch({
            "PitchFamily": "FF",
            "RelSpeed": v,
            "InducedVertBreak": 17.5,
            "HorzBreak": 10.5,
            "SpinRate": 2250.0,
            "SpinAxis": 195.0,
            "RelHeight": 5.85,
            "RelSide": 1.85,
            "Extension": 6.20,
            "VertApprAngle": -4.65,
            "HorzApprAngle": -1.25,
            "PitcherThrows": "RHP",
        })
        progression.append((v, float(res.stuff_plus_overall)))

    for i in range(1, len(progression)):
        v_prev, s_prev = progression[i - 1]
        v_curr, s_curr = progression[i]
        if s_curr < s_prev - 0.5:
            raise AssertionError(
                f"Non-monotonic Stuff+ drop between {v_prev} mph ({s_prev:.2f}) and {v_curr} mph ({s_curr:.2f})"
            )
    logger.info("Monotonic FF progression (80-98 mph): " + ", ".join(f"{v:.0f}:{s:.1f}" for v, s in progression))
    return progression


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train blended MiLB 4YR + Synthetic D1 4-Head XGBoost model.")
    parser.add_argument("--milb-csv", type=Path, default=Path("data/processed/milb_postdraft_pitches.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/ncaa_d1_stuff_model.json"))
    parser.add_argument("--synthetic-n", type=int, default=25_000)
    args = parser.parse_args(argv)

    df_syn = generate_synthetic_ncaa_d1_dataset(n_pitches=args.synthetic_n, n_pitchers=200, seed=42)
    df_milb = pd.read_csv(args.milb_csv) if args.milb_csv.exists() else None
    if df_milb is not None:
        logger.info(f"Loaded {len(df_milb):,} empirical 4YR MiLB pitches from {args.milb_csv}")
    else:
        logger.warning(f"{args.milb_csv} not found; training on calibrated synthetic NCAA D1 dataset only.")

    engine = BlendedCollegeStuffEngine(n_splits=5, random_state=42)
    engine.fit_blended(df_syn, df_milb)
    verify_monotonic_ff_progression(engine)

    export_model_for_web(engine, args.output_json)
    logger.info(f"Exported blended model to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
