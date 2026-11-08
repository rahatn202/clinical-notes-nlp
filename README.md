# Clinical Notes NLP: Anesthesia Information Extraction

Extracting structured anesthesia information from free-text operative notes, and measuring, honestly, how well a rule-based extractor and a local large language model (LLM) do against a human-reviewed gold standard.

| Field extracted | Example values |
|---|---|
| Anesthesia type (one or more) | general, MAC / sedation, spinal, epidural, regional block, local, topical |
| Airway device | endotracheal tube, LMA, nasotracheal, mask, not stated |
| Estimated blood loss | number in mL, or "minimal" |
| Complications | none, any, not stated |

Everything runs locally. The LLM runs through [Ollama](https://ollama.com) on your own computer, and the project uses public sample notes only. No patient data is involved.

## Pipeline

```
MTSamples CSV
  │ 1  prepare_data.py         clean text, remove duplicates, flag truncated notes
  │ 2a leakage_audit.py        what duplicates do to a classifier (random vs grouped split)
  │ 2b anesthesia_detector.py  TF-IDF + logistic regression: which notes involve anesthesia?
  │ 3a extract_rules.py        rule-based extractor (baseline)
  │ 3b extract_llm.py          local LLM with a JSON schema (Ollama)
  │ 4  sample_gold.py + label_app.py   100 notes labeled or reviewed by hand
  │ 5  evaluate.py + app.py    accuracy with confidence intervals, error review, run monitoring
```

## Findings so far

**Data cleaning** ([reports/data_profile.md](reports/data_profile.md))
- **More than half of the rows are duplicates.** MTSamples files the same note under several specialties: 4,966 rows reduce to 2,355 unique notes, and 91% of unique notes carry two or more specialty labels.
- **32 notes are truncated** (they stop after the first heading) and are skipped.
- **The scraped text lost its line breaks** (commas replaced them); the cleaning step restores them.
- **710 usable notes describe anesthesia:** 616 with an "ANESTHESIA:" heading and 94 only in the narrative text, which is the hard case for rules.

**Leakage audit** ([reports/leakage_audit.md](reports/leakage_audit.md))

Specialty classification is the most common task done with this dataset. Training the same model with a random split vs. a split that keeps all copies of a note together shows:
- With a random split, **63% of test notes are copies of training notes**.
- On those copies, the model **repeats the label it memorized from the training copy almost every time**. Because the test copy is filed under a different specialty, its accuracy on them is near zero.
- So random-split scores on this dataset are misleading in both directions, depending on the metric. A sound evaluation keeps copies together (grouped split: 55% accuracy, macro F1 0.45 on 10 specialties) and accounts for notes having several correct labels.

**Anesthesia note detector** ([reports/anesthesia_detector.md](reports/anesthesia_detector.md))
- Trained only on whether a note has an "ANESTHESIA:" heading, **with those heading lines removed from the text**, so it must learn from the rest of the note.
- Cross-validated ROC-AUC **0.94**, recall **0.93**. It also finds **74%** of the narrative-only notes it was never trained on.
- Measured precision (0.66) is a lower bound: 85% of the notes it flags without a heading do mention anesthesia, sedation or intubation.

**Extraction results**: see [reports/evaluation.md](reports/evaluation.md) after step 5, and the dashboard.

## How to run

Requirements: Python 3.10+, and [Ollama](https://ollama.com) for step 3b.

```bash
pip install -r requirements.txt
```

**Data.** Download `mtsamples.csv` from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and put it in `data/raw/`.

**Steps 1 to 4a** (about 2 minutes):
```bash
./run_all.sh prepare
```

**Step 4b: label the gold standard.** Follow [docs/labeling_guide.md](docs/labeling_guide.md):
```bash
streamlit run label_app.py
```
Each save goes straight to `data/gold/gold_labels.csv`, so you can stop and continue later. The app never shows the rule or LLM answers, so the labels stay independent of the methods being tested ("blind").

If `data/gold/draft_labels.csv` exists, the app fills the form with that draft and shows its reasoning. Nothing counts as a gold label until it is reviewed and saved. Each saved row records where it came from (`label_source`) and whether the reviewer changed the draft (`changed_from_draft`).

**Step 3b: run the LLM.** Pick a model that fits your computer's memory:

| Mac memory | Model | Download |
|---|---|---|
| 8 GB | `qwen2.5:3b` or `llama3.2:3b` | about 2 GB |
| 16 GB or more | `qwen2.5:7b` (default) | about 5 GB |

```bash
ollama pull qwen2.5:7b
python src/extract_llm.py                        # the 100 gold notes
python src/extract_llm.py --model llama3.2:3b    # compare another model
```
If it stops, run the same command again to continue. To test a new prompt, copy `prompts/v1.md` to `prompts/v2.md`, edit it, and run with `--prompt v2`. Each model and prompt is kept as a separate run.

**Step 5: evaluate and explore:**
```bash
python src/evaluate.py
streamlit run app.py
```

## Repository layout

```
src/common.py               shared schema (fields and allowed values) and paths
src/prepare_data.py         step 1  cleaning and deduplication
src/leakage_audit.py        step 2a random vs grouped split
src/anesthesia_detector.py  step 2b note detector
src/extract_rules.py        step 3a rule-based extractor
src/extract_llm.py          step 3b Ollama LLM extractor
src/sample_gold.py          step 4a choose the gold notes
src/evaluate.py             step 5a metrics, confidence intervals, error list
label_app.py                step 4b labeling app
app.py                      step 5b dashboard
prompts/v1.md               LLM instructions (same rules as the labeling guide)
docs/labeling_guide.md      how each field is labeled
reports/                    generated reports
data/gold/                  gold sample and hand labels (committed)
data/outputs/               extractor outputs and LLM run log (committed)
data/raw/, data/processed/  not committed; rebuild with the scripts
```

## Design choices

- **One schema for everything.** `src/common.py` defines the fields and allowed values. The LLM gets it as a JSON schema (Ollama structured outputs), and every answer is validated again; problems are recorded, not hidden.
- **Same rules for the human and the model.** The labeling guide and the LLM prompt describe each field the same way.
- **Gold sample built to find weaknesses.** It mixes notes with a heading (55), narrative-only notes (25), notes found only by the detector (10) and notes with no mention (10), and results are reported for each group.
- **Label sources are recorded.** Every gold row says whether it was labeled by the author (with AI help on hard cases) or drafted by an AI assistant and then reviewed and corrected by the author. The evaluation report counts both groups and how often drafts were changed.
- **Uncertainty is shown.** Accuracy comes with 95% bootstrap confidence intervals.
- **Repeatable runs.** Temperature 0, fixed seed, fixed random samples.

## Related work

- Specialty classification on this dataset is common. Reported scores vary widely, from below 40% to above 95% accuracy, which fits the duplicate problem shown in the leakage audit.
- [Information Extraction from Clinical Notes: Are We Ready to Switch to Large Language Models?](https://arxiv.org/abs/2411.10020) (2024) uses MTSamples among other sources for general clinical entities (problems, tests, medications, treatments).
- Yim et al. (2017), [Annotation of pain and anesthesia events for surgery-related processes and outcomes extraction](https://aclanthology.org/W17-2325/), design an annotation schema for anesthesia events using MTSamples operative notes.
- LLM extraction from real operative notes has been studied in [thoracic surgery](https://jtd.amegroups.org/article/view/108599/html) and [orthopedics](https://www.medrxiv.org/content/10.1101/2025.08.19.25333995).

## Limitations

- The gold standard comes from one person without clinical training; there is no second labeler to measure agreement.
- **AI assistance in the gold standard.** The first 38 notes were labeled by the author, who asked an AI assistant (Claude) about hard cases. The other 62 were drafted by Claude, using the same labeling guide and the conventions set on the first 38, and then reviewed note by note by the author in the labeling app. Claude is a different model from the one being evaluated, but reviewers tend to accept drafts, and AI drafts may share mistakes with AI extractors. So scores on the reviewed drafts may be somewhat too favorable to the LLM. The report gives the number of drafts the author changed.
- 100 gold notes is a small sample; the confidence intervals show the uncertainty.
- MTSamples notes are transcription examples, not records from a real hospital, and their style may differ from real operative notes.

## Data and credit

[MTSamples](https://www.mtsamples.com) transcribed medical reports, as published on [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) by Tara Boyle under a CC0 public domain license.
