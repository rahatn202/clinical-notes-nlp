"""Step 4a: choose the notes to label by hand (the gold standard).

Usage: python src/sample_gold.py

The sample mixes easy and hard cases on purpose, so the evaluation can show
where each method fails, not just an overall score:

  heading         55 notes with an "ANESTHESIA:" heading (the common case)
  narrative_only  25 notes that mention anesthesia only in the text (hard for rules)
  detector_only   10 notes found only by the classifier from step 2b
  no_mention      10 notes not flagged at all (checks for false alarms)

A fixed random seed means everyone running this gets the same 100 notes.
Output: data/gold/gold_sample.csv (note_id, group, order)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import GOLD_SAMPLE, PROCESSED, load_notes  # noqa: E402

SEED = 42
GROUP_SIZES = {"heading": 55, "narrative_only": 25, "detector_only": 10, "no_mention": 10}


def main() -> None:
    scores = PROCESSED / "detector_scores.csv"
    if not scores.exists():
        sys.exit("Run python src/anesthesia_detector.py first.")
    notes = load_notes().merge(pd.read_csv(scores), on="note_id")
    head, narr, det = (notes["has_anesthesia_heading"], notes["has_anesthesia_narrative"],
                       notes["detector_flag"])
    pools = {"heading": notes[head],
             "narrative_only": notes[narr & ~head],
             "detector_only": notes[det & ~head & ~narr],
             "no_mention": notes[~det & ~head & ~narr]}
    parts = [pools[g].sample(min(n, len(pools[g])), random_state=SEED).assign(group=g)
             for g, n in GROUP_SIZES.items()]
    sample = pd.concat(parts).sample(frac=1, random_state=SEED)  # mix the groups
    sample["order"] = range(1, len(sample) + 1)
    GOLD_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample[["note_id", "group", "order"]].to_csv(GOLD_SAMPLE, index=False)
    print(sample["group"].value_counts().to_string())
    print(f"{len(sample)} notes -> data/gold/gold_sample.csv")


if __name__ == "__main__":
    main()
