"""Step 2a: what duplicate notes do to a specialty classifier.

Usage: python src/leakage_audit.py

Many projects train a specialty classifier on MTSamples and split the rows at
random. The same note appears under several specialties, so copies of a note
end up in both the training and the test set, usually with DIFFERENT labels
(a knee note filed as "Orthopedic" in training and as "Surgery" in testing).

This script trains the same model (TF-IDF + logistic regression) on the same
rows twice, changing only how the rows are split:

  random split   rows split at random (the common approach)
  grouped split  all copies of a note kept in the same split (GroupShuffleSplit)

For the random split it also checks what the model does on test notes it has
already seen: does it generalize, or repeat the label it memorized?

Both use the 10 most common specialties and are repeated over 5 random seeds.
Output: reports/leakage_audit.md and reports/leakage_audit_runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, REPORTS  # noqa: E402
from prepare_data import clean_text, note_id  # noqa: E402

TOP_N = 10
SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5.0))


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(RAW).dropna(subset=["transcription"])
    df["specialty"] = df["medical_specialty"].str.strip()
    df["text"] = df["transcription"].map(clean_text)
    df["note_id"] = df["text"].map(note_id)
    df = df[df["text"].str.split().str.len() >= 30]
    top = df["specialty"].value_counts().head(TOP_N).index
    return df[df["specialty"].isin(top)].reset_index(drop=True)


def score(df, train_idx, test_idx, seed):
    model = make_model().fit(df.loc[train_idx, "text"], df.loc[train_idx, "specialty"])
    test = df.loc[test_idx]
    pred = model.predict(test["text"])
    ok = pred == test["specialty"].to_numpy()

    # "any-label" accuracy: correct if the prediction is one of the note's specialties
    labels_of = df.groupby("note_id")["specialty"].agg(set)
    any_ok = np.array([p in labels_of[n] for p, n in zip(pred, test["note_id"])])

    # test notes whose text (with some label) was also in the training set
    train_labels = df.loc[train_idx].groupby("note_id")["specialty"].agg(set)
    seen = test["note_id"].isin(train_labels.index).to_numpy()
    repeats_training_label = np.array(
        [p in train_labels.get(n, set()) for p, n in zip(pred, test["note_id"])])

    return {"seed": seed,
            "accuracy": ok.mean(),
            "macro_f1": f1_score(test["specialty"], pred, average="macro"),
            "any_label_accuracy": any_ok.mean(),
            "test_notes_seen_in_training": seen.mean(),
            "accuracy_seen_notes": ok[seen].mean() if seen.any() else np.nan,
            "accuracy_unseen_notes": ok[~seen].mean() if (~seen).any() else np.nan,
            "repeats_training_label": (repeats_training_label[seen].mean()
                                       if seen.any() else np.nan)}


def main() -> None:
    df = load_rows()
    print(f"{len(df):,} rows, {df['note_id'].nunique():,} unique notes, top {TOP_N} specialties")
    results = []
    for seed in SEEDS:
        tr, te = train_test_split(df.index, test_size=TEST_SIZE, random_state=seed,
                                  stratify=df["specialty"])
        results.append({"split": "random", **score(df, tr, te, seed)})
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df, groups=df["note_id"]))
        results.append({"split": "grouped", **score(df, df.index[tr], df.index[te], seed)})
        print(f"seed {seed} done")

    res = pd.DataFrame(results)
    summary = res.groupby("split").agg(["mean", "std"]).drop(columns="seed")
    write_report(df, res, summary)
    print(summary.round(3).to_string())


def fmt(summary, split, col, pct=True):
    m, s = summary.loc[split, (col, "mean")], summary.loc[split, (col, "std")]
    if pd.isna(m):
        return "n/a"
    return f"{m:.0%} ± {s:.0%}" if pct else f"{m:.2f} ± {s:.2f}"


def write_report(df, res, summary) -> None:
    r, g = "random", "grouped"
    row = lambda name, col, pct=True: (f"| {name} | {fmt(summary, r, col, pct)} "
                                       f"| {fmt(summary, g, col, pct)} |")
    repeats = summary.loc[r, ("repeats_training_label", "mean")]
    seen = summary.loc[r, ("test_notes_seen_in_training", "mean")]
    lines = [
        "# Leakage audit: duplicate notes and specialty classification", "",
        "Generated by `src/leakage_audit.py`.", "",
        f"Same model (TF-IDF + logistic regression), same {len(df):,} rows "
        f"({df['note_id'].nunique():,} unique notes, {TOP_N} most common specialties), "
        f"{len(SEEDS)} random seeds (mean ± standard deviation). Only the train/test split changes.", "",
        "| Metric | Random split | Grouped split (by note) |", "|---|---:|---:|",
        row("Test notes also seen in training", "test_notes_seen_in_training"),
        row("Accuracy", "accuracy"),
        row("Macro F1", "macro_f1", False),
        row("Any-label accuracy*", "any_label_accuracy"),
        row("Accuracy on test notes seen in training", "accuracy_seen_notes"),
        row("Accuracy on test notes not seen in training", "accuracy_unseen_notes"),
        row("Seen notes: model repeats a label from the training copy", "repeats_training_label"),
        "",
        "\\* Counted as correct if the predicted specialty is any of the specialties the note "
        "is filed under.", "",
        "## What this shows", "",
        f"- **With a random split, {seen:.0%} of test notes are copies of training notes.**",
        f"- **The model memorizes rather than generalizes on those copies.** It returns a label "
        f"it saw on the training copy {repeats:.0%} of the time. Because the test copy is filed "
        "under a different specialty, plain accuracy on these notes is close to zero.",
        "- **So a random split makes scores misleading in both directions:** plain accuracy "
        "is pushed down (memorized labels are marked wrong), while any-label accuracy is pushed "
        "up (memorized labels count as right).",
        "- **The sound setup** keeps all copies of a note in the same split and evaluates "
        "against all of a note's labels. That is the grouped column.",
        "- **The specialty label itself is ambiguous:** one note can be correctly filed under "
        "several specialties, so no model can reach 100% plain accuracy on this data.", ""]
    (REPORTS / "leakage_audit.md").write_text("\n".join(lines))
    res.round(4).to_csv(REPORTS / "leakage_audit_runs.csv", index=False)


if __name__ == "__main__":
    main()
