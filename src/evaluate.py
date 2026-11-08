"""Step 5a: compare every method against the hand-labeled gold standard.

Usage: python src/evaluate.py

Methods compared: the rule-based extractor and every LLM run found in
data/outputs/llm/ (one per model and prompt version).

Per field:
  anesthesia_types  precision, recall and F1 per type and overall (micro),
                    plus exact-match accuracy (the whole set must be right)
  airway            accuracy
  ebl               accuracy (the number must match, or both must say minimal / not stated)
  complications     accuracy
Accuracy comes with a 95% bootstrap confidence interval, because 100 notes is a
small sample. Results are also split by note group (heading, narrative_only, ...).

Outputs:
  reports/evaluation.md   readable summary
  reports/metrics.csv     every number, for the dashboard
  reports/errors.csv      every disagreement with the gold label, for review
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (ANESTHESIA_TYPES, FIELD_LABELS, GOLD_LABELS, GOLD_SAMPLE, OUTPUTS,  # noqa: E402
                    REPORTS, RULES_OUT, RUNS_LOG, empty_record, read_jsonl)

FIELDS = list(FIELD_LABELS)
N_BOOT = 2000
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
def load_gold() -> pd.DataFrame:
    if not GOLD_LABELS.exists():
        sys.exit("No gold labels yet. Run: streamlit run label_app.py")
    g = pd.read_csv(GOLD_LABELS, dtype={"anesthesia_types": str}, keep_default_na=False)
    sample = pd.read_csv(GOLD_SAMPLE)
    g = g.merge(sample[["note_id", "group"]], on="note_id")
    g["record"] = g.apply(lambda r: {
        "anesthesia_types": [t for t in r["anesthesia_types"].split("|") if t],
        "airway": r["airway"],
        "ebl_ml": float(r["ebl_ml"]) if str(r["ebl_ml"]).strip() not in ("", "nan") else None,
        "ebl_minimal": str(r["ebl_minimal"]) == "True",
        "complications": r["complications"]}, axis=1)
    g["unsure"] = g["unsure"].astype(str) == "True"
    if "label_source" not in g:
        g["label_source"] = "human_with_ai_guidance"
    g["label_source"] = g["label_source"].replace("", "human_with_ai_guidance")
    if "changed_from_draft" not in g:
        g["changed_from_draft"] = ""
    return g


def provenance_lines(gold: pd.DataFrame) -> list[str]:
    """Describe how the gold labels were produced, with counts."""
    n = len(gold)
    guided = int((gold["label_source"] == "human_with_ai_guidance").sum())
    drafted = gold[gold["label_source"] == "ai_draft_human_reviewed"]
    changed = int((drafted["changed_from_draft"].astype(str) == "True").sum())
    lines = ["## How the gold standard was made", "",
             "| Label source | Notes |", "|---|---:|",
             f"| Labeled by the author, discussing notes with an AI assistant (Claude) | {guided} |",
             f"| Drafted by Claude, reviewed and corrected by the author | {len(drafted)} |",
             f"| Total | {n} |", ""]
    if len(drafted):
        lines += [f"The author changed **{changed} of {len(drafted)}** Claude drafts "
                  f"({changed / len(drafted):.0%}) during review.", ""]
    lines += ["Because Claude helped produce the labels, these results measure agreement with "
              "an AI-assisted, human-reviewed reference, not with independent expert annotation. "
              "A second, independent human labeler would make the reference stronger.", ""]
    return lines


def load_methods() -> dict[str, dict]:
    """{method name: {note_id: record}}"""
    methods = {}
    rules = read_jsonl(RULES_OUT)
    if rules:
        methods["rules"] = {r["note_id"]: r["record"] for r in rules}
    for path in sorted((OUTPUTS / "llm").glob("*.jsonl")):
        recs = read_jsonl(path)
        if recs:
            methods[f"llm: {path.stem}"] = {r["note_id"]: r["record"] for r in recs}
    if not methods:
        sys.exit("No extractor outputs found. Run src/extract_rules.py and src/extract_llm.py.")
    return methods


# ---------------------------------------------------------------- comparing
BASELINE = "baseline: always not stated"


def ebl_text(rec: dict) -> str:
    if rec["ebl_minimal"]:
        return "minimal"
    return "not stated" if rec["ebl_ml"] is None else f"{rec['ebl_ml']:g} mL"


def field_value(rec: dict, field: str) -> str:
    if field == "anesthesia_types":
        return ", ".join(rec["anesthesia_types"]) or "(none)"
    if field == "ebl":
        return ebl_text(rec)
    return rec[field]


def same(gold: dict, pred: dict, field: str) -> bool:
    if field == "anesthesia_types":
        return set(gold["anesthesia_types"]) == set(pred["anesthesia_types"])
    if field == "ebl":
        if gold["ebl_minimal"] != pred["ebl_minimal"]:
            return False
        if gold["ebl_minimal"]:  # both say minimal; a number next to it doesn't change that
            return True
        if gold["ebl_ml"] is None or pred["ebl_ml"] is None:
            return gold["ebl_ml"] is None and pred["ebl_ml"] is None
        return abs(gold["ebl_ml"] - pred["ebl_ml"]) <= 0.5
    return gold[field] == pred[field]


def boot_ci(hits: np.ndarray) -> tuple[float, float]:
    if len(hits) == 0:
        return np.nan, np.nan
    means = rng.choice(hits, size=(N_BOOT, len(hits)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) else np.nan
    return p, r, f


def evaluate(gold: pd.DataFrame, methods: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, errors = [], []
    for name, preds in methods.items():
        covered = gold[gold["note_id"].isin(preds)]
        n_missing = len(gold) - len(covered)
        # notes the method never processed count as "nothing found"
        pairs = [(r, preds.get(r["note_id"], empty_record())) for _, r in gold.iterrows()]

        for field in FIELDS:
            hits = np.array([same(g["record"], p, field) for g, p in pairs])
            lo, hi = boot_ci(hits)
            metrics.append({"method": name, "field": field, "group": "all", "metric": "accuracy",
                            "value": hits.mean(), "ci_low": lo, "ci_high": hi, "n": len(hits),
                            "n_missing": n_missing})
            for grp in gold["group"].unique():
                mask = (gold["group"] == grp).to_numpy()
                metrics.append({"method": name, "field": field, "group": grp,
                                "metric": "accuracy", "value": hits[mask].mean(),
                                "n": int(mask.sum())})
            for (g, p), ok in zip(pairs, hits):
                if not ok:
                    errors.append({"method": name, "field": field, "note_id": g["note_id"],
                                   "group": g["group"], "gold": field_value(g["record"], field),
                                   "predicted": field_value(p, field), "gold_unsure": g["unsure"]})

        # per-type precision / recall / F1 for anesthesia types
        totals = np.zeros(3)
        for t in ANESTHESIA_TYPES:
            tp = sum(t in g["record"]["anesthesia_types"] and t in p["anesthesia_types"] for g, p in pairs)
            fp = sum(t not in g["record"]["anesthesia_types"] and t in p["anesthesia_types"] for g, p in pairs)
            fn = sum(t in g["record"]["anesthesia_types"] and t not in p["anesthesia_types"] for g, p in pairs)
            totals += (tp, fp, fn)
            for metric, v in zip(["precision", "recall", "f1"], prf(tp, fp, fn)):
                metrics.append({"method": name, "field": f"type:{t}", "group": "all",
                                "metric": metric, "value": v, "n": tp + fn})
        for metric, v in zip(["precision", "recall", "f1"], prf(*totals)):
            metrics.append({"method": name, "field": "anesthesia_types", "group": "all",
                            "metric": f"micro_{metric}", "value": v, "n": int(totals[0] + totals[2])})
    return pd.DataFrame(metrics), pd.DataFrame(errors)


# ---------------------------------------------------------------- report
def pct(v):
    return "n/a" if pd.isna(v) else f"{v:.0%}"


def write_report(gold, metrics, errors, methods) -> None:
    acc = metrics[(metrics["metric"] == "accuracy") & (metrics["group"] == "all")]
    names = list(methods)
    lines = ["# Evaluation against the gold standard", "",
             "Generated by `src/evaluate.py`.", "",
             f"Gold standard: **{len(gold)} hand-labeled notes** "
             f"({int(gold['unsure'].sum())} marked unsure). "
             "Labels follow [docs/labeling_guide.md](../docs/labeling_guide.md).", "",
             *provenance_lines(gold),
             "## Accuracy by field (95% bootstrap confidence interval)", "",
             "| Field | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for field in FIELDS:
        cells = []
        for n in names:
            r = acc[(acc["method"] == n) & (acc["field"] == field)].iloc[0]
            cells.append(f"{pct(r['value'])} ({pct(r['ci_low'])} to {pct(r['ci_high'])})")
        lines.append(f"| {FIELD_LABELS[field]} | " + " | ".join(cells) + " |")

    micro = metrics[metrics["metric"].str.startswith("micro_")]
    lines += ["", "## Anesthesia type: precision, recall and F1 (all types pooled)", "",
              "| Method | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for n in names:
        m = micro[micro["method"] == n].set_index("metric")["value"]
        lines.append(f"| {n} | {pct(m['micro_precision'])} | {pct(m['micro_recall'])} "
                     f"| {'n/a' if pd.isna(m['micro_f1']) else format(m['micro_f1'], '.2f')} |")

    by_group = metrics[(metrics["metric"] == "accuracy") & (metrics["group"] != "all")
                       & (metrics["field"] == "anesthesia_types")]
    groups = [g for g in ["heading", "narrative_only", "detector_only", "no_mention"]
              if g in set(by_group["group"])]
    lines += ["", "## Anesthesia type accuracy by note group", "",
              "| Group (notes) | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for grp in groups:
        sub = by_group[by_group["group"] == grp]
        n = int(sub["n"].iloc[0])
        cells = [pct(sub[sub["method"] == m]["value"].iloc[0]) for m in names]
        lines.append(f"| {grp} ({n}) | " + " | ".join(cells) + " |")

    if RUNS_LOG.exists():
        runs = pd.read_csv(RUNS_LOG)
        lines += ["", "## LLM runs", "",
                  "| Run | Notes | Valid JSON | Schema OK | Mean seconds per note |",
                  "|---|---:|---:|---:|---:|"]
        for _, r in runs.iterrows():
            lines.append(f"| {r['run_id']} | {r['notes']} | {pct(r['valid_json_rate'])} | "
                         f"{pct(r['schema_ok_rate'])} | {r['mean_seconds']:.1f} |")

    lines += ["", f"Every disagreement is listed in `reports/errors.csv` "
              f"({len(errors)} rows) and can be browsed in the dashboard.", "",
              "## Limitations", "",
              "- One person labeled the gold standard, without clinical training, with help from "
              "an AI assistant (see above). There is no second, independent labeler to measure "
              "agreement.",
              f"- {len(gold)} notes is a small sample; the confidence intervals show how uncertain "
              "each number is.",
              *(["- Prompt versions after v1 were written after reading v1's errors on these same "
                 "notes. There is no held-out set, so their scores are likely somewhat optimistic."]
                if any(not n.endswith("__v1") for n in names if n.startswith("llm: ")) else []),
              "- MTSamples notes are transcription examples, not records from a real hospital.", ""]
    (REPORTS / "evaluation.md").write_text("\n".join(lines))


def main() -> None:
    gold = load_gold()
    methods = load_methods()
    # reference point: answer "nothing found / not stated" for every note
    methods[BASELINE] = {nid: empty_record() for nid in gold["note_id"]}
    metrics, errors = evaluate(gold, methods)
    errors = errors[errors["method"] != BASELINE]
    REPORTS.mkdir(exist_ok=True)
    metrics.round(4).to_csv(REPORTS / "metrics.csv", index=False)
    errors.to_csv(REPORTS / "errors.csv", index=False)
    write_report(gold, metrics, errors, methods)
    acc = metrics[(metrics["metric"] == "accuracy") & (metrics["group"] == "all")]
    print(acc.pivot(index="field", columns="method", values="value").round(2).to_string())
    print(f"{len(gold)} gold notes ({gold['label_source'].value_counts().to_dict()}), "
          f"{len(errors)} disagreements -> reports/evaluation.md")


if __name__ == "__main__":
    main()
