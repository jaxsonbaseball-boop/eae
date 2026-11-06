"""TrackMan B1 Portable & Stadium CSV/JSON Schema Normalizer and Coordinate Standardizer."""

import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


PITCH_TYPE_MAP: Dict[str, str] = {
    "FASTBALL": "FF",
    "FOURSEAMFASTBALL": "FF",
    "FOUR-SEAM": "FF",
    "4-SEAM": "FF",
    "FF": "FF",
    "SINKER": "SI",
    "TWOSEAMFASTBALL": "SI",
    "SI": "SI",
    "FT": "SI",
    "CUTTER": "FC",
    "FC": "FC",
    "SLIDER": "SL",
    "SL": "SL",
    "SWEEPER": "ST",
    "ST": "ST",
    "CURVEBALL": "CU",
    "CU": "CU",
    "KC": "CU",
    "CHANGEUP": "CH",
    "CH": "CH",
    "SPLITTER": "FS",
    "FS": "FS",
}


@dataclass
class NormalizedPitchRecord:
    """Standardized TrackMan B1 pitch record ready for the Project Apex Stuff+ and 3D engines."""
    pitch_no: int
    pitcher_name: str
    pitcher_throws: str
    pitch_type: str
    rel_speed_mph: float
    spin_rate_rpm: float
    spin_axis_deg: float
    spin_efficiency_pct: float
    ivb_in: float
    hb_in: float
    rel_height_ft: float
    rel_side_ft: float
    extension_ft: float
    vaa_deg: float
    haa_deg: float
    plate_loc_side_ft: float
    plate_loc_height_ft: float


def normalize_pitch_tag(raw_tag: str) -> str:
    cleaned = (raw_tag or "FF").strip().upper().replace(" ", "")
    return PITCH_TYPE_MAP.get(cleaned, "FF")


def parse_trackman_row(row: Dict[str, Any], index: int = 1) -> NormalizedPitchRecord:
    """Parse a single TrackMan B1 CSV or JSON row into a NormalizedPitchRecord."""
    def _float(keys: List[str], default: float) -> float:
        for k in keys:
            if k in row and row[k] not in (None, "", "NA", "null"):
                try:
                    return float(row[k])
                except ValueError:
                    pass
        return default

    pitch_type = normalize_pitch_tag(
        str(row.get("TaggedPitchType") or row.get("AutoPitchType") or row.get("PitchType") or "FF")
    )
    throws_raw = str(row.get("PitcherThrows") or row.get("Throws") or "Right").strip().upper()
    pitcher_throws = "LHP" if throws_raw.startswith("L") else "RHP"

    return NormalizedPitchRecord(
        pitch_no=int(_float(["PitchNo", "pitch_no"], float(index))),
        pitcher_name=str(row.get("Pitcher") or row.get("PitcherName") or "Unknown Pitcher").strip(),
        pitcher_throws=pitcher_throws,
        pitch_type=pitch_type,
        rel_speed_mph=_float(["RelSpeed", "rel_speed_mph", "Velocity"], 85.0),
        spin_rate_rpm=_float(["SpinRate", "spin_rate_rpm", "Spin"], 2200.0),
        spin_axis_deg=_float(["SpinAxis", "spin_axis_deg", "TiltDeg"], 210.0),
        spin_efficiency_pct=_float(["SpinEfficiency", "TrueSpinPct", "spin_efficiency_pct"], 88.0),
        ivb_in=_float(["InducedVertBreak", "IVB", "ivb_in"], 14.0),
        hb_in=_float(["HorzBreak", "HB", "hb_in"], 8.0),
        rel_height_ft=_float(["RelHeight", "rel_height_ft"], 5.80),
        rel_side_ft=_float(["RelSide", "rel_side_ft"], 1.85),
        extension_ft=_float(["Extension", "extension_ft"], 6.00),
        vaa_deg=_float(["VertApprAngle", "VAA", "vaa_deg"], -5.10),
        haa_deg=_float(["HorzApprAngle", "HAA", "haa_deg"], -1.20),
        plate_loc_side_ft=_float(["PlateLocSide", "plate_x_ft"], 0.0),
        plate_loc_height_ft=_float(["PlateLocHeight", "plate_z_ft"], 2.50),
    )


def load_trackman_csv(csv_path: str) -> List[NormalizedPitchRecord]:
    """Load a TrackMan B1 CSV file into standardized pitch records."""
    records: List[NormalizedPitchRecord] = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            records.append(parse_trackman_row(row, index=idx))
    return records


def load_trackman_json(json_path: str) -> List[NormalizedPitchRecord]:
    """Load a TrackMan B1 JSON payload into standardized pitch records."""
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else data.get("pitches", [])
    return [parse_trackman_row(r, index=i + 1) for i, r in enumerate(rows)]
