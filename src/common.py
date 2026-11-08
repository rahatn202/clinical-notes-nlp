"""Shared paths, the extraction schema, and helpers used by every step.

Keeping the schema in one place means the rule-based extractor, the LLM, the
labeling app and the evaluation all use exactly the same fields and values.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mtsamples.csv"
PROCESSED = ROOT / "data" / "processed"
GOLD = ROOT / "data" / "gold"
OUTPUTS = ROOT / "data" / "outputs"
REPORTS = ROOT / "reports"
PROMPTS = ROOT / "prompts"

GOLD_SAMPLE = GOLD / "gold_sample.csv"
GOLD_LABELS = GOLD / "gold_labels.csv"
RULES_OUT = OUTPUTS / "rules.jsonl"
RUNS_LOG = OUTPUTS / "runs.csv"

# ---------------------------------------------------------------- schema
ANESTHESIA_TYPES = ["general", "mac_sedation", "spinal", "epidural",
                    "regional_block", "local", "topical"]
AIRWAY_VALUES = ["endotracheal_tube", "lma", "nasotracheal", "mask", "not_stated"]
COMPLICATION_VALUES = ["none", "any", "not_stated"]

FIELD_LABELS = {
    "anesthesia_types": "Anesthesia type",
    "airway": "Airway device",
    "ebl": "Estimated blood loss",
    "complications": "Complications",
}

# JSON Schema given to the LLM (Ollama "structured outputs"), so the model can
# only answer with these keys and values.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "anesthesia_types": {"type": "array",
                             "items": {"type": "string", "enum": ANESTHESIA_TYPES}},
        "airway": {"type": "string", "enum": AIRWAY_VALUES},
        "ebl_ml": {"type": ["number", "null"]},
        "ebl_minimal": {"type": "boolean"},
        "complications": {"type": "string", "enum": COMPLICATION_VALUES},
    },
    "required": ["anesthesia_types", "airway", "ebl_ml", "ebl_minimal", "complications"],
}


def empty_record() -> dict:
    """What 'nothing found' looks like."""
    return {"anesthesia_types": [], "airway": "not_stated", "ebl_ml": None,
            "ebl_minimal": False, "complications": "not_stated"}


def validate_record(rec) -> tuple[dict, list[str]]:
    """Check a record against the schema.

    Returns (cleaned record, list of problems). Unknown values are dropped and
    reported, so one bad field doesn't throw away the whole answer.
    """
    problems = []
    if not isinstance(rec, dict):
        return empty_record(), ["not a JSON object"]
    out = empty_record()

    types = rec.get("anesthesia_types", [])
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list):
        problems.append("anesthesia_types is not a list")
        types = []
    for t in types:
        if t in ANESTHESIA_TYPES:
            if t not in out["anesthesia_types"]:
                out["anesthesia_types"].append(t)
        else:
            problems.append(f"unknown anesthesia type: {t!r}")
    out["anesthesia_types"].sort(key=ANESTHESIA_TYPES.index)

    airway = rec.get("airway", "not_stated")
    if airway in AIRWAY_VALUES:
        out["airway"] = airway
    else:
        problems.append(f"unknown airway: {airway!r}")

    ebl = rec.get("ebl_ml")
    if ebl is not None:
        try:
            ebl = float(ebl)
            if ebl < 0:
                raise ValueError
            out["ebl_ml"] = ebl
        except (TypeError, ValueError):
            problems.append(f"ebl_ml is not a number: {ebl!r}")
    out["ebl_minimal"] = bool(rec.get("ebl_minimal", False))

    comp = rec.get("complications", "not_stated")
    if comp in COMPLICATION_VALUES:
        out["complications"] = comp
    else:
        problems.append(f"unknown complications value: {comp!r}")
    for key in rec:
        if key not in JSON_SCHEMA["properties"]:
            problems.append(f"unexpected key: {key!r}")
    return out, problems


# ---------------------------------------------------------------- io helpers
def load_notes(usable_only: bool = True) -> pd.DataFrame:
    path = PROCESSED / "notes.csv"
    if not path.exists():
        raise SystemExit("data/processed/notes.csv not found. Run: python src/prepare_data.py")
    notes = pd.read_csv(path)
    return notes[~notes["is_truncated"]].reset_index(drop=True) if usable_only else notes


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
