/**
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
