"""Step 3a: rule-based (regular expression) extractor. This is the baseline.

Usage: python src/extract_rules.py

It reads the labeled lines that operative notes usually have
("ANESTHESIA: General endotracheal.", "ESTIMATED BLOOD LOSS: 50 mL.",
"COMPLICATIONS: None.") and turns them into the shared schema in common.py.
When there is no heading, it tries a few simple narrative patterns
("...under general anesthesia...").

Rules are fast, so this runs on every usable note (the LLM only runs on the
notes that need it). Output: data/outputs/rules.jsonl
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ANESTHESIA_TYPES, RULES_OUT, load_notes, validate_record, write_jsonl  # noqa: E402

I = re.IGNORECASE

# ---------------------------------------------------------------- patterns
HEADING_ANESTHESIA = re.compile(r"\b(?:TYPE OF )?ANESTHESIA(?: TYPE)?\s*:\s*([^\n]*)")
NARRATIVE_ANESTHESIA = re.compile(
    r"\bunder\s+((?:[\w/-]+\s+){0,3}?(?:anesthesia|sedation|block))", I)
TYPE_PATTERNS = {
    "general": re.compile(r"\bgeneral\b|endotracheal|\blma\b|laryngeal mask|\btiva\b|"
                          r"total intravenous|inhalation", I),
    "mac_sedation": re.compile(r"\bmac\b|monitored anesthesia care|sedation|\bsedated\b", I),
    "spinal": re.compile(r"\bspinal\b|subarachnoid", I),
    "epidural": re.compile(r"epidural|caudal", I),
    "regional_block": re.compile(r"\bblock\b|\bregional\b|\bbier\b", I),
    "local": re.compile(r"\blocal\b|lidocaine|xylocaine|marcaine|bupivacaine|infiltrat", I),
    "topical": re.compile(r"\btopical\b", I),
}
AIRWAY_PATTERNS = [  # checked in this order; the first match wins
    ("nasotracheal", re.compile(r"nasotracheal|nasal intubation|nasally intubated", I)),
    ("lma", re.compile(r"\blma\b|laryngeal mask", I)),
    ("endotracheal_tube", re.compile(r"endotracheal|\bett\b|orotracheal|\bintubated\b|"
                                     r"\bintubation\b", I)),
]
MASK = re.compile(r"\bmask\b", I)
EBL_HEADING = re.compile(r"\b(?:ESTIMATED BLOOD LOSS|EBL)\s*:\s*([^\n]*)")
EBL_NARRATIVE = re.compile(r"estimated blood loss (?:was|of|is)\s+([^.\n]*)", I)
NUMBER_UNIT = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(ml|cc|milliliters?|cubic|liters?|l\b)?", I)
MINIMAL = re.compile(r"minimal|negligible|scant|\bmin\b|insignificant", I)
NO_LOSS = re.compile(r"^\s*(none|nil|zero|no\b)", I)
COMP_HEADING = re.compile(r"\bCOMPLICATIONS?\s*:\s*([^\n]*)")
NO_COMP = re.compile(r"^\s*(none|no\b|nil|there (?:were|was) no|without)", I)
NO_COMP_NARRATIVE = re.compile(r"\b(?:no|without)\s+(?:\w+\s+){0,2}complications\b", I)


def types_from(text: str) -> list[str]:
    found = [t for t, pat in TYPE_PATTERNS.items() if pat.search(text)]
    return sorted(found, key=ANESTHESIA_TYPES.index)


def parse_ebl(value: str) -> tuple[float | None, bool]:
    """'Minimal.' -> (None, True); 'Less than 50 cc.' -> (50, False); 'None' -> (0, False)."""
    if MINIMAL.search(value):
        return None, True
    if NO_LOSS.search(value):
        return 0.0, False
    m = NUMBER_UNIT.search(value)
    if not m:
        return None, False
    number = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "ml").lower()
    if unit.startswith("l"):  # liters
        number *= 1000
    return number, False


def extract(text: str) -> tuple[dict, dict]:
    """Return (record, evidence). Evidence keeps the text each value came from."""
    rec, ev = {}, {}

    m = HEADING_ANESTHESIA.search(text)
    if m and m.group(1).strip():
        source = m.group(1)
        ev["anesthesia_types"] = m.group(0).strip()
    else:
        hits = NARRATIVE_ANESTHESIA.findall(text)
        source = " ; ".join(hits)
        if hits:
            ev["anesthesia_types"] = source
    rec["anesthesia_types"] = types_from(source)

    rec["airway"] = "not_stated"
    for value, pat in AIRWAY_PATTERNS:
        a = pat.search(text)
        if a:
            rec["airway"] = value
            ev["airway"] = text[max(0, a.start() - 40):a.end() + 40].replace("\n", " ")
            break
    else:
        if MASK.search(source):
            rec["airway"] = "mask"
            ev["airway"] = source

    m = EBL_HEADING.search(text) or EBL_NARRATIVE.search(text)
    rec["ebl_ml"], rec["ebl_minimal"] = parse_ebl(m.group(1)) if m else (None, False)
    if m:
        ev["ebl"] = m.group(0).strip()

    m = COMP_HEADING.search(text)
    if m and m.group(1).strip():
        rec["complications"] = "none" if NO_COMP.search(m.group(1)) else "any"
        ev["complications"] = m.group(0).strip()
    else:
        n = NO_COMP_NARRATIVE.search(text)
        rec["complications"] = "none" if n else "not_stated"
        if n:
            ev["complications"] = n.group(0)

    clean, problems = validate_record(rec)
    assert not problems, problems  # rules can only produce schema values
    return clean, ev


def main() -> None:
    notes = load_notes()
    out = []
    for _, r in notes.iterrows():
        rec, ev = extract(r["text"])
        out.append({"note_id": r["note_id"], "method": "rules", "record": rec, "evidence": ev})
    write_jsonl(RULES_OUT, out)
    found = sum(bool(o["record"]["anesthesia_types"]) for o in out)
    print(f"{len(out):,} notes processed, anesthesia type found in {found:,} -> "
          f"{RULES_OUT.relative_to(RULES_OUT.parents[2])}")


if __name__ == "__main__":
    main()
