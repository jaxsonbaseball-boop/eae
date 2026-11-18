"""
Project Apex — Phase 2: Browser-Ready JSON Exporter & Pure-JS Inference Engine Generator.

Serializes the trained 4-Head Multi-Task XGBoost Stuff+ Cascade (`CollegeStuffEngine`) into:
  1. `data/processed/ncaa_d1_stuff_model.json`: Compact, zero-dependency JSON artifact containing
     training metadata, NCAA D1 pitch-family baselines (`mu_college`, `sigma_college`, kinematic
     means/SDs), the 10x10 location density grid, and flattened shallow XGBoost decision trees.
  2. `web/stuff_model_inference.js`: Pure-JavaScript ES6/CommonJS/Browser tree-walker and 10x10
     grid location neutralizer that evaluates any bullpen pitch in < 2ms with < 0.01 Stuff+
     parity against Python `xgboost`.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xgboost as xgb

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("project_apex.export_for_web")

from src.models.college_stuff_engine import (
    CollegeStuffEngine,
    DEFAULT_FOUL_SHARE_OF_CONTACT,
    EMPIRICAL_LOCATION_WEIGHTS_10X10,
    FourHeadPitchFamilyModel,
    GRID_PLATE_HEIGHT_FT,
    GRID_PLATE_SIDE_FT,
    MODEL_FEATURE_COLUMNS,
    NCAA_D1_AVG_WOBA,
    NCAA_D1_WOBA_SCALE,
    NEUTRAL_COUNT_RV_BALL,
    NEUTRAL_COUNT_RV_CALLED_STRIKE,
    NEUTRAL_COUNT_RV_FOUL,
    NEUTRAL_COUNT_RV_WHIFF,
    PITCH_FAMILY_MAP,
    build_10x10_location_grid_matrix,
    compute_expected_rv_from_heads,
    compute_sub_distribution_velocity_cliff,
    neutralize_location_xrv,
)


# =============================================================================
# 1. XGBOOST TREE FLATTENER & PYTHON REFERENCE TREE WALKER
# =============================================================================

def _parse_dump_tree_recursive(
    node_dict: Dict[str, Any],
    feature_name_to_idx: Mapping[str, int],
) -> Dict[str, List[Any]]:
    """
    Convert a single tree dictionary from `booster.get_dump(dump_format="json")` into
    compact flat parallel arrays `{f, t, l, r, d, v}` indexed by contiguous node index.
    """
    nodes_by_id: Dict[int, Dict[str, Any]] = {}

    def _collect(n: Dict[str, Any]) -> None:
        nid = int(n["nodeid"])
        nodes_by_id[nid] = n
        for child in n.get("children", []):
            _collect(child)

    _collect(node_dict)
    sorted_ids = sorted(nodes_by_id.keys())
    id_remap = {old_id: idx for idx, old_id in enumerate(sorted_ids)}

    f_arr: List[int] = []
    t_arr: List[float] = []
    l_arr: List[int] = []
    r_arr: List[int] = []
    d_arr: List[int] = []
    v_arr: List[float] = []

    for old_id in sorted_ids:
        n = nodes_by_id[old_id]
        if "leaf" in n:
            f_arr.append(-1)
            t_arr.append(0.0)
            l_arr.append(-1)
            r_arr.append(-1)
            d_arr.append(-1)
            v_arr.append(round(float(n["leaf"]), 7))
        else:
            split_name = str(n["split"])
            if split_name in feature_name_to_idx:
                feat_idx = int(feature_name_to_idx[split_name])
            elif split_name.startswith("f") and split_name[1:].isdigit():
                feat_idx = int(split_name[1:])
            else:
                raise KeyError(f"Unrecognized XGBoost split feature name: {split_name!r}")

            # Ensure float32 threshold parity with XGBoost's internal 32-bit float comparison
            thresh_f32 = float(np.float32(float(n["split_condition"])))
            f_arr.append(feat_idx)
            t_arr.append(round(thresh_f32, 7))
            l_arr.append(int(id_remap[int(n["yes"])]))
            r_arr.append(int(id_remap[int(n["no"])]))
            d_arr.append(int(id_remap[int(n.get("missing", n["yes"]))]))
            v_arr.append(0.0)

    return {
        "f": f_arr,
        "t": t_arr,
        "l": l_arr,
        "r": r_arr,
        "d": d_arr,
        "v": v_arr,
    }


def eval_flat_tree_single(tree: Mapping[str, Sequence[Any]], x_row: np.ndarray) -> float:
    """Evaluate a single flattened tree on a 1D feature vector `x_row`."""
    f_arr = tree["f"]
    t_arr = tree["t"]
    l_arr = tree["l"]
    r_arr = tree["r"]
    d_arr = tree["d"]
    v_arr = tree["v"]

    node = 0
    while f_arr[node] != -1:
        feat_idx = f_arr[node]
        val = float(np.float32(x_row[feat_idx]))
        if math.isnan(val):
            node = d_arr[node]
        elif val < t_arr[node]:
            node = l_arr[node]
        else:
            node = r_arr[node]
    return float(v_arr[node])


def eval_serialized_head_batch(head_payload: Mapping[str, Any], X_mat: np.ndarray) -> np.ndarray:
    """Evaluate a serialized XGBoost head payload across a 2D feature matrix `X_mat`."""
    base_margin = float(head_payload["base_margin"])
    is_logistic = str(head_payload["objective"]) == "binary:logistic"
    trees = head_payload["trees"]

    n_rows = X_mat.shape[0]
    preds = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        row = X_mat[i]
        margin = base_margin
        for tr in trees:
            margin += eval_flat_tree_single(tr, row)
        if is_logistic:
            preds[i] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))
        else:
            preds[i] = margin
    return preds


def serialize_xgboost_head(
    model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
    objective: str,
    feature_names: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> Dict[str, Any]:
    """
    Serialize an `XGBClassifier` or `XGBRegressor` into compact quantized arrays with
    exact `base_margin` calibration so pure-JS inference matches `xgboost` to machine precision.
    """
    booster = model.get_booster()
    feature_map = {name: idx for idx, name in enumerate(feature_names)}
    raw_dumps = booster.get_dump(dump_format="json")
    flat_trees = [_parse_dump_tree_recursive(json.loads(d), feature_map) for d in raw_dumps]

    # Calibrate exact base_margin against XGBoost's native output_margin on a reference vector
    x_ref = np.zeros((1, len(feature_names)), dtype=np.float32)
    dmat_ref = xgb.DMatrix(x_ref)
    xgb_margin = float(booster.predict(dmat_ref, output_margin=True)[0])
    tree_sum = sum(eval_flat_tree_single(tr, x_ref[0]) for tr in flat_trees)
    exact_base_margin = round(xgb_margin - tree_sum, 7)

    return {
        "objective": objective,
        "base_margin": exact_base_margin,
        "n_trees": len(flat_trees),
        "trees": flat_trees,
    }


# =============================================================================
# 2. JSON ARTIFACT EXPORTER (`data/processed/ncaa_d1_stuff_model.json`)
# =============================================================================

def build_web_model_payload(engine: CollegeStuffEngine) -> Dict[str, Any]:
    """Build the complete JSON dictionary for `data/processed/ncaa_d1_stuff_model.json`."""
    if not engine.family_models:
        raise ValueError("CollegeStuffEngine has no fitted pitch-family models to export.")

    validation_metrics: Dict[str, Any] = {}
    college_baselines: Dict[str, Any] = {}
    xgboost_trees: Dict[str, Any] = {}

    for family, fam_model in engine.family_models.items():
        validation_metrics[family] = asdict(fam_model.metrics)
        b = fam_model.baseline_stats
        college_baselines[family] = {
            "pitch_family": b.pitch_family,
            "count": b.count,
            "mu_college": b.mu_college_xrv,
            "sigma_college": b.sigma_college_xrv,
            "velo": {"mean": b.velo_mean, "sd": b.velo_sd},
            "ivb": {"mean": b.ivb_mean, "sd": b.ivb_sd},
            "hb": {"mean": b.hb_mean, "sd": b.hb_sd},
            "ext": {"mean": b.ext_mean, "sd": b.ext_sd},
            "vaa": {"mean": b.vaa_mean, "sd": b.vaa_sd},
            "spin": {"mean": b.spin_mean, "sd": b.spin_sd},
        }
        xgboost_trees[family] = {
            "swing": serialize_xgboost_head(fam_model.head_swing, "binary:logistic"),
            "whiff": serialize_xgboost_head(fam_model.head_whiff, "binary:logistic"),
            "called_strike": serialize_xgboost_head(fam_model.head_called_strike, "binary:logistic"),
            "xwobacon": serialize_xgboost_head(fam_model.head_xwobacon, "reg:squarederror"),
        }

    return {
        "metadata": {
            "model_name": "Project Apex Phase 2 — NCAA D1 4-Head XGBoost Stuff+ Engine",
            "version": "4.0.0",
            "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_sample_size": int(engine.training_sample_size),
            "pitch_families": list(engine.family_models.keys()),
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "max_depth": int(engine.max_depth),
            "n_estimators": int(engine.n_estimators),
            "group_kfold_splits": int(engine.n_splits),
            "validation_metrics": validation_metrics,
            "run_value_constants": {
                "ncaa_d1_avg_woba": NCAA_D1_AVG_WOBA,
                "ncaa_d1_woba_scale": NCAA_D1_WOBA_SCALE,
                "rv_ball": NEUTRAL_COUNT_RV_BALL,
                "rv_called_strike": NEUTRAL_COUNT_RV_CALLED_STRIKE,
                "rv_whiff": NEUTRAL_COUNT_RV_WHIFF,
                "rv_foul": NEUTRAL_COUNT_RV_FOUL,
                "foul_share_of_contact": DEFAULT_FOUL_SHARE_OF_CONTACT,
            },
            "location_grid_10x10": {
                "plate_height_ft": [round(float(z), 4) for z in GRID_PLATE_HEIGHT_FT],
                "plate_side_ft": [round(float(x), 4) for x in GRID_PLATE_SIDE_FT],
                "weights_flat": [round(float(w), 8) for w in EMPIRICAL_LOCATION_WEIGHTS_10X10.ravel()],
            },
        },
        "college_baselines": college_baselines,
        "xgboost_trees": xgboost_trees,
    }


def generate_js_inference_helper_code() -> str:
    """
    Generate the standalone, zero-dependency JavaScript inference module
    `web/stuff_model_inference.js` that walks the exported 4-head XGBoost trees across
    the 10x10 synthetic strike-zone grid in < 2ms per pitch.
    """
    return """/**
 * Project Apex Phase 2 — Pure-JavaScript 4-Head XGBoost Stuff+ Browser Inference Engine
 * Zero external dependencies. Evaluates a 10x10 location-neutralized pitch across
 * 4 XGBoost heads (RHB + LHB splits) in < 2ms with < 0.01 Stuff+ parity vs. Python xgboost.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.ApexStuffInference = api;
    root.evaluatePitchStuffPlus = api.evaluatePitchStuffPlus;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var PITCH_FAMILY_MAP = {
    FF: "FF", FASTBALL: "FF", FOURSEAMFASTBALL: "FF", "FOUR-SEAM": "FF", "4-SEAM": "FF", FA: "FF",
    SI: "SI", SINKER: "SI", TWOSEAMFASTBALL: "SI", "TWO-SEAM": "SI", "2-SEAM": "SI", FT: "SI",
    FC: "FC", CUTTER: "FC",
    SL: "SL", SLIDER: "SL",
    SW: "SW", ST: "SW", SWEEPER: "SW", SLURVE: "SW",
    CU: "CU", CURVEBALL: "CU", KC: "CU", KNUCKLECURVE: "CU", CS: "CU",
    CH: "CH", CHANGEUP: "CH", CHANGE: "CH",
    FS: "FS", SPLITTER: "FS", SPLITFINGER: "FS", FO: "FS"
  };

  // Indices of features that vary across the 10x10 (PlateLocHeight, PlateLocSide) grid:
  // 11: VertApprAngle, 12: HorzApprAngle_norm, 18: PlateLocHeight, 19: PlateLocSide_norm, 20: ZoneDist, 21: InStrikeZone
  var LOCATION_DEPENDENT_FEATURES = { 11: true, 12: true, 18: true, 19: true, 20: true, 21: true };

  function mapPitchFamily(rawTag, ivb, hbNorm) {
    var clean = String(rawTag || "FF").trim().toUpperCase().replace(/[\s_]+/g, "");
    var mapped = PITCH_FAMILY_MAP[clean] || "FF";
    if (mapped === "SL" && typeof hbNorm === "number" && typeof ivb === "number") {
      if (hbNorm <= -12.0 && ivb >= -4.5) {
        return "SW";
      }
    }
    return mapped;
  }

  function computeExpectedVaaAndHaa(relSpeed, relHeight, relSideNorm, plateZ, plateXNorm) {
    var expVaa = -4.85 + 1.32 * (plateZ - 2.50) + 0.085 * (relSpeed - 91.5) - 0.85 * (relHeight - 5.85);
    var expHaa = -1.25 + 1.05 * plateXNorm - 0.90 * (relSideNorm - 1.85);
    return [expVaa, expHaa];
  }

  function evalFlatTree(tree, featureVec) {
    var f = tree.f;
    var t = tree.t;
    var l = tree.l;
    var r = tree.r;
    var d = tree.d;
    var v = tree.v;
    var node = 0;
    while (f[node] !== -1) {
      var val = Math.fround(featureVec[f[node]]);
      if (Number.isNaN(val)) {
        node = d[node];
      } else if (val < t[node]) {
        node = l[node];
      } else {
        node = r[node];
      }
    }
    return v[node];
  }

  function prepareHeadPartition(headModel) {
    if (headModel._partitioned) {
      return;
    }
    var kinTrees = [];
    var locTrees = [];
    var trees = headModel.trees;
    for (var i = 0; i < trees.length; i++) {
      var tr = trees[i];
      var fArr = tr.f;
      var isLoc = false;
      for (var j = 0; j < fArr.length; j++) {
        if (fArr[j] !== -1 && LOCATION_DEPENDENT_FEATURES[fArr[j]]) {
          isLoc = true;
          break;
        }
      }
      if (isLoc) {
        locTrees.push(tr);
      } else {
        kinTrees.push(tr);
      }
    }
    headModel._kinTrees = kinTrees;
    headModel._locTrees = locTrees;
    headModel._partitioned = true;
  }

  function evalTreeListSum(treeList, featureVec) {
    var sum = 0.0;
    var n = treeList.length;
    for (var i = 0; i < n; i++) {
      sum += evalFlatTree(treeList[i], featureVec);
    }
    return sum;
  }

  function finalizeHeadPrediction(margin, isLogistic) {
    if (isLogistic) {
      var clamped = margin < -30.0 ? -30.0 : (margin > 30.0 ? 30.0 : margin);
      return 1.0 / (1.0 + Math.exp(-clamped));
    }
    return margin;
  }

  function evalHead(headModel, featureVec) {
    var margin = headModel.base_margin + evalTreeListSum(headModel.trees, featureVec);
    return finalizeHeadPrediction(margin, headModel.objective === "binary:logistic");
  }

  function computeSubDistributionVelocityCliff(pitchFamily, relSpeed) {
    if ((pitchFamily === "FF" || pitchFamily === "SI") && relSpeed < 86.0) {
      var delta = 86.0 - relSpeed;
      return -2.1 * delta - 0.35 * Math.pow(delta, 1.5);
    }
    return 0.0;
  }

  function computeExpectedRvFromHeads(pSwing, pWhiff, pCs, xwobacon, rvConsts) {
    var pSw = Math.min(1.0 - 1e-4, Math.max(1e-4, pSwing));
    var pWh = Math.min(1.0 - 1e-4, Math.max(1e-4, pWhiff));
    var pCsClamped = Math.min(1.0 - 1e-4, Math.max(1e-4, pCs));
    var xw = Math.min(1.0, Math.max(0.0, xwobacon));

    var pTake = 1.0 - pSw;
    var xrvTake = pCsClamped * rvConsts.rv_called_strike + (1.0 - pCsClamped) * rvConsts.rv_ball;
    var bipRv = (xw - rvConsts.ncaa_d1_avg_woba) / rvConsts.ncaa_d1_woba_scale;
    var contactRv = rvConsts.foul_share_of_contact * rvConsts.rv_foul + (1.0 - rvConsts.foul_share_of_contact) * bipRv;
    var xrvSwing = pWh * rvConsts.rv_whiff + (1.0 - pWh) * contactRv;

    return pTake * xrvTake + pSw * xrvSwing;
  }

  function scorePitch10x10(modelArtifact, pitchRow, rhbWeight) {
    if (typeof rhbWeight !== "number") {
      rhbWeight = 0.58;
    }
    var throwsStr = String(pitchRow.PitcherThrows || "RHP").trim().toUpperCase();
    var isLhp = throwsStr.indexOf("L") === 0 ? 1.0 : 0.0;
    var signHand = isLhp > 0.5 ? -1.0 : 1.0;

    var relSpeed = Number(pitchRow.RelSpeed !== undefined ? pitchRow.RelSpeed : 91.5);
    var spinRate = Number(pitchRow.SpinRate !== undefined ? pitchRow.SpinRate : 2260.0);
    var spinAxis = Number(pitchRow.SpinAxis !== undefined ? pitchRow.SpinAxis : 210.0);
    var extRaw = Number(pitchRow.Extension !== undefined ? pitchRow.Extension : 6.15);
    var ext = Math.min(8.0, Math.max(3.5, extRaw));
    var relHeight = Number(pitchRow.RelHeight !== undefined ? pitchRow.RelHeight : 5.85);
    var relSide = Number(pitchRow.RelSide !== undefined ? pitchRow.RelSide : 1.85);
    var ivb = Number(pitchRow.InducedVertBreak !== undefined ? pitchRow.InducedVertBreak : 15.8);
    var hb = Number(pitchRow.HorzBreak !== undefined ? pitchRow.HorzBreak : 8.5);

    var hbNorm = pitchRow.HB_norm !== undefined ? Number(pitchRow.HB_norm) : hb * signHand;
    var relSideNorm = pitchRow.RelSide_norm !== undefined ? Number(pitchRow.RelSide_norm) : Math.abs(relSide);
    var rawTag = pitchRow.PitchFamily || pitchRow.TaggedPitchType || "FF";
    var pitchFamily = mapPitchFamily(rawTag, ivb, hbNorm);

    var familyTrees = modelArtifact.xgboost_trees[pitchFamily] || modelArtifact.xgboost_trees[Object.keys(modelArtifact.xgboost_trees)[0]];
    var familyBase = modelArtifact.college_baselines[pitchFamily] || modelArtifact.college_baselines[Object.keys(modelArtifact.college_baselines)[0]];
    var rvConsts = modelArtifact.metadata.run_value_constants;
    var gridSpec = modelArtifact.metadata.location_grid_10x10;

    prepareHeadPartition(familyTrees.swing);
    prepareHeadPartition(familyTrees.whiff);
    prepareHeadPartition(familyTrees.called_strike);
    prepareHeadPartition(familyTrees.xwobacon);

    var perceivedSpeed = pitchRow.PerceivedSpeed !== undefined
      ? Number(pitchRow.PerceivedSpeed)
      : relSpeed * ((60.5 - 6.15) / Math.max(45.0, 60.5 - ext));
    var bauerUnits = pitchRow.BauerUnits !== undefined
      ? Number(pitchRow.BauerUnits)
      : spinRate / Math.max(50.0, relSpeed);
    var spinAxisNorm = pitchRow.SpinAxis_norm !== undefined
      ? Number(pitchRow.SpinAxis_norm)
      : (isLhp > 0.5 ? ((360.0 - spinAxis) % 360.0 + 360.0) % 360.0 : ((spinAxis % 360.0) + 360.0) % 360.0);
    var armSlotAngle = pitchRow.ArmSlotAngle !== undefined
      ? Number(pitchRow.ArmSlotAngle)
      : Math.atan2(relHeight - 5.05, Math.max(0.05, relSideNorm - 0.45)) * (180.0 / Math.PI);

    var obsPlateZ = Number(pitchRow.PlateLocHeight !== undefined ? pitchRow.PlateLocHeight : 2.50);
    var obsPlateX = Number(pitchRow.PlateLocSide !== undefined ? pitchRow.PlateLocSide : 0.0);
    var obsPlateXNorm = obsPlateX * signHand;
    var obsVaa = Number(pitchRow.VertApprAngle !== undefined ? pitchRow.VertApprAngle : -4.85);
    var obsHaa = Number(pitchRow.HorzApprAngle !== undefined ? pitchRow.HorzApprAngle : -1.25 * signHand);
    var obsHaaNorm = obsHaa * signHand;

    var expObs = computeExpectedVaaAndHaa(relSpeed, relHeight, relSideNorm, obsPlateZ, obsPlateXNorm);
    var vaaDiff = pitchRow.VAA_diff !== undefined ? Number(pitchRow.VAA_diff) : (obsVaa - expObs[0]);
    var haaDiff = pitchRow.HAA_diff !== undefined ? Number(pitchRow.HAA_diff) : (obsHaaNorm - expObs[1]);

    var isPrimaryFb = (pitchFamily === "FF" || pitchFamily === "SI");
    var fbVelo = Number(pitchRow.fb_velo_anchor !== undefined ? pitchRow.fb_velo_anchor : (isPrimaryFb ? relSpeed : 91.5));
    var fbIvb = Number(pitchRow.fb_ivb_anchor !== undefined ? pitchRow.fb_ivb_anchor : (isPrimaryFb ? ivb : 15.8));
    var fbHb = Number(pitchRow.fb_hb_anchor !== undefined ? pitchRow.fb_hb_anchor : (isPrimaryFb ? hbNorm : 8.5));

    var veloDiffFb = isPrimaryFb ? 0.0 : (pitchRow.VeloDiff_FB !== undefined ? Number(pitchRow.VeloDiff_FB) : (relSpeed - fbVelo));
    var ivbDiffFb = isPrimaryFb ? 0.0 : (pitchRow.IVBDiff_FB !== undefined ? Number(pitchRow.IVBDiff_FB) : (ivb - fbIvb));
    var hbDiffFb = isPrimaryFb ? 0.0 : (pitchRow.HBDiff_FB !== undefined ? Number(pitchRow.HBDiff_FB) : (hbNorm - fbHb));

    var heights = gridSpec.plate_height_ft;
    var sides = gridSpec.plate_side_ft;
    var weights = gridSpec.weights_flat;

    var featVec = new Float64Array(26);
    featVec[0] = relSpeed;
    featVec[1] = perceivedSpeed;
    featVec[2] = spinRate;
    featVec[3] = bauerUnits;
    featVec[4] = spinAxisNorm;
    featVec[5] = ivb;
    featVec[6] = hbNorm;
    featVec[7] = relHeight;
    featVec[8] = relSideNorm;
    featVec[9] = ext;
    featVec[10] = armSlotAngle;
    featVec[13] = vaaDiff;
    featVec[14] = haaDiff;
    featVec[15] = veloDiffFb;
    featVec[16] = ivbDiffFb;
    featVec[17] = hbDiffFb;
    featVec[23] = isLhp;
    featVec[25] = 0.0;

    function evalGridForBatter(isLhbVal) {
      var platoonSame = (isLhp === isLhbVal) ? 1.0 : 0.0;
      featVec[22] = platoonSame;
      featVec[24] = isLhbVal;

      // Pre-evaluate location-invariant trees once per batter side
      var swBase = familyTrees.swing.base_margin + evalTreeListSum(familyTrees.swing._kinTrees, featVec);
      var whBase = familyTrees.whiff.base_margin + evalTreeListSum(familyTrees.whiff._kinTrees, featVec);
      var csBase = familyTrees.called_strike.base_margin + evalTreeListSum(familyTrees.called_strike._kinTrees, featVec);
      var xwBase = familyTrees.xwobacon.base_margin + evalTreeListSum(familyTrees.xwobacon._kinTrees, featVec);

      var swLocTrees = familyTrees.swing._locTrees;
      var whLocTrees = familyTrees.whiff._locTrees;
      var csLocTrees = familyTrees.called_strike._locTrees;
      var xwLocTrees = familyTrees.xwobacon._locTrees;

      var xrvSum = 0.0;
      var swSum = 0.0;
      var whSum = 0.0;
      var csSum = 0.0;
      var xwSum = 0.0;
      var idx = 0;

      for (var iz = 0; iz < heights.length; iz++) {
        var gz = heights[iz];
        var zScaled = (gz - 2.50) / 1.05;
        var inZoneZ = (gz >= 1.50 && gz <= 3.50);

        for (var ix = 0; ix < sides.length; ix++) {
          var gx = sides[ix];
          var w = weights[idx++];

          var expGrid = computeExpectedVaaAndHaa(relSpeed, relHeight, relSideNorm, gz, gx);
          featVec[11] = expGrid[0] + vaaDiff;
          featVec[12] = expGrid[1] + haaDiff;
          featVec[18] = gz;
          featVec[19] = gx;
          var xScaled = gx / (17.0 / 24.0);
          featVec[20] = Math.sqrt(xScaled * xScaled + zScaled * zScaled);
          featVec[21] = (inZoneZ && Math.abs(gx) <= 0.83) ? 1.0 : 0.0;

          var pSw = finalizeHeadPrediction(swBase + evalTreeListSum(swLocTrees, featVec), true);
          var pWh = finalizeHeadPrediction(whBase + evalTreeListSum(whLocTrees, featVec), true);
          var pCs = finalizeHeadPrediction(csBase + evalTreeListSum(csLocTrees, featVec), true);
          var xw = Math.min(1.0, Math.max(0.0, finalizeHeadPrediction(xwBase + evalTreeListSum(xwLocTrees, featVec), false)));
          var nodeXrv = computeExpectedRvFromHeads(pSw, pWh, pCs, xw, rvConsts);

          xrvSum += w * nodeXrv;
          swSum += w * pSw;
          whSum += w * pWh;
          csSum += w * pCs;
          xwSum += w * xw;
        }
      }
      return { xrv: xrvSum, pSw: swSum, pWh: whSum, pCs: csSum, xw: xwSum };
    }

    var resRhb = evalGridForBatter(0.0);
    var resLhb = evalGridForBatter(1.0);

    var wRhb = Math.min(1.0, Math.max(0.0, rhbWeight));
    var xrvOverall = wRhb * resRhb.xrv + (1.0 - wRhb) * resLhb.xrv;

    var muC = familyBase.mu_college;
    var sigmaC = Math.max(1e-6, familyBase.sigma_college);
    var cliffAdj = computeSubDistributionVelocityCliff(familyBase.pitch_family, relSpeed);

    var stuffRhb = 100.0 - ((resRhb.xrv - muC) / sigmaC) * 10.0 + cliffAdj;
    var stuffLhb = 100.0 - ((resLhb.xrv - muC) / sigmaC) * 10.0 + cliffAdj;
    var stuffOverall = 100.0 - ((xrvOverall - muC) / sigmaC) * 10.0 + cliffAdj;

    return {
      pitch_family: familyBase.pitch_family,
      stuff_xrv_overall: Number(xrvOverall.toFixed(6)),
      stuff_xrv_vs_rhb: Number(resRhb.xrv.toFixed(6)),
      stuff_xrv_vs_lhb: Number(resLhb.xrv.toFixed(6)),
      stuff_plus_overall: Number(stuffOverall.toFixed(2)),
      stuff_plus_vs_rhb: Number(stuffRhb.toFixed(2)),
      stuff_plus_vs_lhb: Number(stuffLhb.toFixed(2)),
      grid_p_swing: Number((wRhb * resRhb.pSw + (1.0 - wRhb) * resLhb.pSw).toFixed(4)),
      grid_p_whiff_given_swing: Number((wRhb * resRhb.pWh + (1.0 - wRhb) * resLhb.pWh).toFixed(4)),
      grid_p_called_strike_given_take: Number((wRhb * resRhb.pCs + (1.0 - wRhb) * resLhb.pCs).toFixed(4)),
      grid_expected_xwobacon: Number((wRhb * resRhb.xw + (1.0 - wRhb) * resLhb.xw).toFixed(4))
    };
  }

  return {
    evaluatePitchStuffPlus: scorePitch10x10,
    scorePitch10x10: scorePitch10x10,
    computeSubDistributionVelocityCliff: computeSubDistributionVelocityCliff,
    evalFlatTree: evalFlatTree,
    evalHead: evalHead,
    mapPitchFamily: mapPitchFamily
  };
});
"""


def export_model_for_web(
    engine: CollegeStuffEngine,
    json_output_path: Union[str, Path] = "data/processed/ncaa_d1_stuff_model.json",
    js_output_path: Union[str, Path] = "web/stuff_model_inference.js",
) -> Dict[str, Any]:
    """
    Export the trained `CollegeStuffEngine` to `data/processed/ncaa_d1_stuff_model.json`
    and write the pure-JS inference helper `web/stuff_model_inference.js`.
    """
    payload = build_web_model_payload(engine)

    json_path = Path(json_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    js_path = Path(js_output_path)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    js_code = generate_js_inference_helper_code()
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_code)

    logger.info(
        f"Exported browser-ready NCAA D1 Stuff+ JSON ({json_path.stat().st_size / 1024:.1f} KB) "
        f"to {json_path} and JS tree walker to {js_path}."
    )
    return payload


def verify_python_vs_js_parity(
    engine: CollegeStuffEngine,
    json_model_path: Union[str, Path],
    js_helper_path: Union[str, Path],
    test_pitches: Sequence[Mapping[str, Any]],
    tolerance_stuff_plus: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Verify that Python `xgboost` predictions (`neutralize_location_xrv`), Python's serialized
    flat-tree evaluator, and the actual JavaScript `web/stuff_model_inference.js` execution
    (via `node` or macOS `osascript -l JavaScript` when available) match within `tolerance_stuff_plus`
    (<= 0.01 Stuff+).
    """
    with open(json_model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # First verify Python xgboost vs serialized JSON tree structure in Python
    py_results: List[Dict[str, Any]] = []
    for pitch in test_pitches:
        py_out = neutralize_location_xrv(engine, pitch)
        fam = py_out.pitch_family
        fam_trees = payload["xgboost_trees"][fam]
        fam_base = payload["college_baselines"][fam]
        weights_flat = np.array(payload["metadata"]["location_grid_10x10"]["weights_flat"], dtype=np.float64)

        mat_rhb = build_10x10_location_grid_matrix(pitch, batter_side="RHB")
        mat_lhb = build_10x10_location_grid_matrix(pitch, batter_side="LHB")
        mat_both = np.vstack([mat_rhb, mat_lhb])

        p_sw = eval_serialized_head_batch(fam_trees["swing"], mat_both)
        p_wh = eval_serialized_head_batch(fam_trees["whiff"], mat_both)
        p_cs = eval_serialized_head_batch(fam_trees["called_strike"], mat_both)
        xw = np.clip(eval_serialized_head_batch(fam_trees["xwobacon"], mat_both), 0.0, 1.0)
        xrv_all = compute_expected_rv_from_heads(p_sw, p_wh, p_cs, xw)

        xrv_rhb = float(np.dot(xrv_all[:100], weights_flat))
        xrv_lhb = float(np.dot(xrv_all[100:], weights_flat))
        xrv_overall = 0.58 * xrv_rhb + 0.42 * xrv_lhb
        cliff_adj = compute_sub_distribution_velocity_cliff(fam, float(pitch.get("RelSpeed", 91.5)))
        stuff_flat = round(
            100.0 - ((xrv_overall - float(fam_base["mu_college"])) / float(fam_base["sigma_college"])) * 10.0 + cliff_adj,
            2,
        )
        diff_py = abs(py_out.stuff_plus_overall - stuff_flat)
        if diff_py > tolerance_stuff_plus:
            raise AssertionError(
                f"Python XGBoost vs serialized tree mismatch: {py_out.stuff_plus_overall} vs {stuff_flat} (diff={diff_py})"
            )
        py_results.append({
            "pitch_family": fam,
            "python_stuff_plus": py_out.stuff_plus_overall,
            "serialized_stuff_plus": stuff_flat,
            "diff": round(diff_py, 5),
        })

    # If Node.js or macOS JavaScriptCore (`osascript -l JavaScript`) is available, also run actual JS engine!
    node_bin = shutil.which("node")
    osascript_bin = shutil.which("osascript")

    if node_bin or osascript_bin:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp_js:
            with open(js_helper_path, "r", encoding="utf-8") as jf:
                js_lib_code = jf.read()
            runner_script = (
                js_lib_code
                + "\nvar modelData = "
                + json.dumps(payload)
                + ";\nvar pitches = "
                + json.dumps(list(test_pitches))
                + ";\nvar lib = (typeof module === 'object' && module.exports) ? module.exports : this.ApexStuffInference;\n"
                + "var out = pitches.map(function(p) { return lib.scorePitch10x10(modelData, p, 0.58); });\n"
                + "if (typeof console !== 'undefined' && console.log) { console.log(JSON.stringify(out)); } else { JSON.stringify(out); }\n"
            )
            tmp_js.write(runner_script)
            tmp_js_path = tmp_js.name

        try:
            cmd = [node_bin, tmp_js_path] if node_bin else [osascript_bin, "-l", "JavaScript", tmp_js_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
            js_stdout = (proc.stdout or proc.stderr).strip().splitlines()[-1]
            js_outputs = json.loads(js_stdout)
            for idx, (py_res, js_out) in enumerate(zip(py_results, js_outputs)):
                js_stuff = float(js_out["stuff_plus_overall"])
                diff_js = abs(float(py_res["python_stuff_plus"]) - js_stuff)
                py_res["js_stuff_plus"] = js_stuff
                py_res["python_vs_js_diff"] = round(diff_js, 5)
                if diff_js > tolerance_stuff_plus:
                    raise AssertionError(
                        f"Pitch {idx} Python vs JS Stuff+ parity failed: "
                        f"Python={py_res['python_stuff_plus']} vs JS={js_stuff} (diff={diff_js})"
                    )
        finally:
            Path(tmp_js_path).unlink(missing_ok=True)

    return py_results
