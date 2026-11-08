"""Step 4b: label the gold-standard notes in your browser.

Usage: streamlit run label_app.py

Shows one note at a time with anesthesia-related words highlighted, and a form
for the four fields. Each save is written straight to data/gold/gold_labels.csv,
so you can stop at any time and continue later.

The app never shows what the rules or the LLM extracted, so your labels are
independent of the methods being evaluated ("blind" labeling).

If data/gold/draft_labels.csv exists, unlabeled notes are pre-filled with that
draft (written by an AI assistant, Claude) and marked clearly. You review the
note and correct anything you disagree with. Each saved row records where the
label came from (label_source) and whether you changed the draft
(changed_from_draft), so the evaluation can report this openly.
"""
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (AIRWAY_VALUES, ANESTHESIA_TYPES, COMPLICATION_VALUES,  # noqa: E402
                    DRAFT_LABELS, GOLD_LABELS, GOLD_SAMPLE, load_notes)

COLUMNS = ["note_id", "anesthesia_types", "airway", "ebl_ml", "ebl_minimal",
           "complications", "unsure", "comment", "labeled_at", "label_source",
           "changed_from_draft"]
# label_source values
HUMAN_GUIDED = "human_with_ai_guidance"   # labeled by the author while discussing notes with Claude
DRAFT_REVIEWED = "ai_draft_human_reviewed"  # pre-filled by Claude, checked and corrected by the author
FIELDS = ["anesthesia_types", "airway", "ebl_ml", "ebl_minimal", "complications"]
HIGHLIGHT = re.compile(
    r"(anesthe\w*|sedation|sedated|\bMAC\b|\bLMA\b|laryngeal mask|intubat\w*|endotracheal|"
    r"nasotracheal|spinal|epidural|caudal|\bblock\b|lidocaine|marcaine|bupivacaine|local|"
    r"topical|blood loss|\bEBL\b|complications?)", re.IGNORECASE)

st.set_page_config(page_title="Gold labeling", layout="wide")


@st.cache_data
def load():
    if not GOLD_SAMPLE.exists():
        st.error("No gold sample. Run: python src/sample_gold.py")
        st.stop()
    sample = pd.read_csv(GOLD_SAMPLE).sort_values("order")
    notes = load_notes().set_index("note_id")
    return sample, notes


def read_labels() -> pd.DataFrame:
    if GOLD_LABELS.exists():
        labels = pd.read_csv(GOLD_LABELS, dtype=str, keep_default_na=False)
        # rows saved before drafts existed were labeled by the author with Claude's guidance
        if "label_source" not in labels:
            labels["label_source"] = HUMAN_GUIDED
        labels["label_source"] = labels["label_source"].replace("", HUMAN_GUIDED)
        if "changed_from_draft" not in labels:
            labels["changed_from_draft"] = ""
        return labels
    return pd.DataFrame(columns=COLUMNS)


def read_drafts() -> pd.DataFrame:
    if DRAFT_LABELS.exists():
        return pd.read_csv(DRAFT_LABELS, dtype=str, keep_default_na=False).set_index("note_id")
    return pd.DataFrame()


def normalize(row) -> tuple:
    """Comparable form of a label, to tell whether the reviewer changed the draft."""
    types = tuple(sorted(t for t in str(row["anesthesia_types"]).split("|") if t))
    ebl = str(row["ebl_ml"]).strip()
    ebl = "" if ebl in ("", "nan") else f"{float(ebl):g}"
    return (types, row["airway"], ebl, str(row["ebl_minimal"]) == "True", row["complications"])


def save_label(row: dict) -> None:
    labels = read_labels()
    labels = labels[labels["note_id"] != row["note_id"]]
    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
    GOLD_LABELS.parent.mkdir(parents=True, exist_ok=True)
    labels[COLUMNS].to_csv(GOLD_LABELS, index=False)


def highlighted(text: str) -> str:
    safe = html.escape(text)
    safe = HIGHLIGHT.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return safe.replace("\n", "<br>")


sample, notes = load()
labels = read_labels()
drafts = read_drafts()
done = set(labels["note_id"])
ids = sample["note_id"].tolist()

if "pos" not in st.session_state:
    pending = [i for i, n in enumerate(ids) if n not in done]
    st.session_state.pos = pending[0] if pending else 0

st.title("Gold-standard labeling")
st.progress(len(done & set(ids)) / len(ids),
            text=f"{len(done & set(ids))} of {len(ids)} notes labeled")
st.caption("Rules: see docs/labeling_guide.md. Label what the note says, not what is likely.")

nav1, nav2, nav3, nav4 = st.columns([1, 1, 1.4, 3])
if nav1.button("Previous", disabled=st.session_state.pos == 0):
    st.session_state.pos -= 1
if nav2.button("Next", disabled=st.session_state.pos >= len(ids) - 1):
    st.session_state.pos += 1
if nav3.button("Next unlabeled"):
    pending = [i for i, n in enumerate(ids) if n not in done]
    if pending:
        st.session_state.pos = pending[0]
pos = st.session_state.pos
nid = ids[pos]
note = notes.loc[nid]
nav4.markdown(f"**Note {pos + 1} of {len(ids)}** · `{nid}` · "
              + ("labeled" if nid in done else ":orange[not labeled yet]"))

left, right = st.columns([3, 2])
with left:
    st.subheader(note["sample_name"])
    st.markdown(f"<div style='max-height:640px;overflow-y:auto;padding:12px;border:1px solid "
                f"rgba(128,128,128,.3);border-radius:8px;line-height:1.6'>"
                f"{highlighted(note['text'])}</div>", unsafe_allow_html=True)

prev = labels[labels["note_id"] == nid]
p = prev.iloc[0] if len(prev) else None
draft = drafts.loc[nid] if nid in drafts.index else None
from_draft = p is None and draft is not None
if from_draft:
    p = draft  # pre-fill the form with the draft
with right:
    if from_draft:
        st.info("**Claude draft, not yet reviewed.** Check every field against the note and "
                "change anything you disagree with before saving.  \n"
                f"Draft reasoning: {draft.get('draft_reasoning', '')}")
    elif p is not None and p.get("label_source") == DRAFT_REVIEWED:
        st.caption("Reviewed Claude draft"
                   + (" (you changed it)" if str(p.get("changed_from_draft")) == "True" else ""))
    with st.form(f"form_{nid}"):
        types = st.multiselect(
            "Anesthesia type (all that apply)", ANESTHESIA_TYPES,
            default=[t for t in (p["anesthesia_types"].split("|") if p is not None and
                                 p["anesthesia_types"] else []) if t in ANESTHESIA_TYPES])
        airway = st.radio("Airway device", AIRWAY_VALUES, horizontal=True,
                          index=AIRWAY_VALUES.index(p["airway"]) if p is not None else
                          AIRWAY_VALUES.index("not_stated"))
        ebl_prev = str(p["ebl_ml"]) if p is not None else ""
        ebl_text = st.text_input("Estimated blood loss (mL, number only; empty if not stated)",
                                 value=ebl_prev.removesuffix(".0"))
        minimal = st.checkbox("Blood loss described as minimal / negligible (no number)",
                              value=p is not None and str(p["ebl_minimal"]) == "True")
        comp = st.radio("Complications", COMPLICATION_VALUES, horizontal=True,
                        index=COMPLICATION_VALUES.index(p["complications"]) if p is not None
                        else COMPLICATION_VALUES.index("not_stated"))
        unsure = st.checkbox("Unsure about this note",
                             value=p is not None and str(p["unsure"]) == "True")
        comment = st.text_input("Comment (optional)",
                                value="" if from_draft or p is None else p["comment"])
        submitted = st.form_submit_button("Save and go to next", type="primary")

    if submitted:
        ebl_clean = ebl_text.strip().replace(",", "")
        try:
            ebl_val = float(ebl_clean) if ebl_clean else ""
        except ValueError:
            st.error("Blood loss must be a number in mL (or empty).")
            st.stop()
        row = {"note_id": nid, "anesthesia_types": "|".join(types), "airway": airway,
               "ebl_ml": ebl_val, "ebl_minimal": minimal, "complications": comp,
               "unsure": unsure, "comment": comment,
               "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
        if draft is not None and (from_draft or (len(prev) and prev.iloc[0]["label_source"] == DRAFT_REVIEWED)):
            row["label_source"] = DRAFT_REVIEWED
            row["changed_from_draft"] = normalize(row) != normalize(draft)
        else:
            row["label_source"] = prev.iloc[0]["label_source"] if len(prev) else HUMAN_GUIDED
            row["changed_from_draft"] = ""
        save_label(row)
        if pos < len(ids) - 1:
            st.session_state.pos = pos + 1
        st.rerun()
