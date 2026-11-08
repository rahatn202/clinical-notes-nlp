# Clinical Notes NLP: Anesthesia Information Extraction

Can a small LLM running on a laptop pull structured anesthesia details out of operative notes as well as a set of hand-written rules? I built this project to find out, and to measure the answer carefully instead of trusting a single accuracy number.

From each note I extract four things:

| Field | Possible values |
|---|---|
| Anesthesia type (one or more) | general, MAC / sedation, spinal, epidural, regional block, local, topical |
| Airway device | endotracheal tube, LMA, nasotracheal, mask, not stated |
| Estimated blood loss | number in mL, or "minimal" |
| Complications | none, any, not stated |

Everything runs locally. The LLM runs through [Ollama](https://ollama.com) on my own computer, and the notes are public sample transcriptions (MTSamples), so no patient data is involved.

## Why this project

My day job is research data work with longitudinal health data, where most of the effort goes into checking whether the data says what we think it says. I wanted to apply the same habit to clinical text and LLMs. It's easy to get an impressive number from an LLM on clinical notes; it's harder to know whether that number means anything. So a big part of this project is the checking: duplicates, a labeled reference set, confidence intervals, a simple baseline, and a record of every error.

## Pipeline

```
MTSamples CSV
  │ 1  prepare_data.py         clean text, remove duplicates, flag truncated notes
  │ 2a leakage_audit.py        what duplicates do to a classifier (random vs grouped split)
  │ 2b anesthesia_detector.py  TF-IDF + logistic regression: which notes involve anesthesia?
  │ 3a extract_rules.py        rule-based extractor
  │ 3b extract_llm.py          local LLM with a JSON schema (Ollama)
  │ 4  sample_gold.py + label_app.py   100 reference ("gold") notes, labeled or reviewed by me
  │ 5  evaluate.py + app.py    accuracy with confidence intervals, error review, run monitoring
```

## What I found

### The dataset needed more cleaning than I expected

More than half of the rows in MTSamples are duplicates. The same note is filed under several specialties, so 4,966 rows come down to 2,355 unique notes. The scraped text had also lost its line breaks (they turned into commas), which I restored, and 32 notes were cut off after the first heading, so I dropped them. Of the usable notes, 710 describe anesthesia: 616 have an "ANESTHESIA:" heading, and 94 mention it only in the narrative. Those 94 turned out to be the interesting ones. Details: [reports/data_profile.md](reports/data_profile.md).

### Duplicates make the usual benchmark misleading

A popular task on this dataset is predicting the medical specialty. I trained the same classifier with a random split and with a split that keeps all copies of a note together. With the random split, 63% of the test notes also appear in the training set. On those notes the model just repeats the specialty it saw in training, which is usually the "wrong" one because the copy is filed under a different specialty. Random-split scores here can mislead in either direction, depending on the metric. With copies kept together, the classifier gets 55% accuracy (macro F1 0.45) on 10 specialties. Details: [reports/leakage_audit.md](reports/leakage_audit.md).

### A detector for notes that involve anesthesia

To find the narrative-only notes, I trained a TF-IDF + logistic regression model on whether a note has an "ANESTHESIA:" heading, but I removed the heading lines first, so it has to learn from the rest of the text. It reaches ROC-AUC 0.94 and recall 0.93, and it flags 74% of the narrative-only notes, even though in training those count as "no heading" examples. Its measured precision (0.66) undersells it: when I checked, 85% of the notes it flags without a heading do mention anesthesia, sedation or intubation. Details: [reports/anesthesia_detector.md](reports/anesthesia_detector.md).

### Rules vs. LLM

I compared the rules and the `qwen2.5:3b` model (my laptop has 8 GB of memory) against 100 gold notes. The LLM took about 5 seconds per note and always returned valid JSON. I also added a baseline that answers "not stated" for everything, because many notes simply don't mention these fields and I wanted to see how much of the accuracy was just that.

| Field (accuracy) | Rules | LLM, prompt v1 | LLM, prompt v2 | Always "not stated" |
|---|---:|---:|---:|---:|
| Anesthesia type | 55% | 61% | 61% | 9% |
| Airway device | 93% | 56% | 92% | 57% |
| Estimated blood loss | 91% | 42% | 87% | 65% |
| Complications | 92% | 51% | 82% | 53% |

The first LLM run looked bad, and at first I suspected a bug in my evaluation. Looking at the errors showed something else: when a note didn't say something, the model filled in a typical value anyway. It answered "mask" for the airway in 30 notes, 0 mL of blood loss in 44, and "no complications" in 45, even though the prompt told it to answer "not stated". On those three fields it did no better than the always-"not stated" baseline.

So I wrote a second prompt (v2) that makes "not stated" the default and spells out the edge cases. For example, 0 mL needs words like "no blood loss", and "tolerated the procedure well" doesn't mean "no complications". That one change raised those three fields by 31 to 45 points. One caveat: I wrote v2 after reading v1's mistakes on these same 100 notes, so its numbers are probably a little optimistic. A fair test would need new notes.

On anesthesia type, the two methods fail in different ways. The rules are very precise (96%) but depend on the heading, so they get only 28% of narrative-only notes right, compared with 64–68% for the LLM. Pooled F1 is 0.70 for the rules and 0.75–0.79 for the LLM. Getting the full set of types exactly right is hard for both (55–61%), because many notes combine types, such as general anesthesia plus local anesthetic injected at the incision.

If I keep going, the next step is a hybrid: use the rules when there's a heading and the LLM otherwise, and test it on notes that played no part in writing the prompt.

Full report: [reports/evaluation.md](reports/evaluation.md). The dashboard (`streamlit run app.py`) lets you browse every disagreement.

## How the gold standard was made

The gold standard is 100 notes chosen on purpose to include hard cases: 55 with an anesthesia heading, 25 that describe anesthesia only in the narrative, 10 found only by the detector, and 10 with no mention at all.

I labeled the first 38 myself, in a small Streamlit app I built for this (`label_app.py`), and I talked through the hard cases with an AI assistant (Claude). Those conversations are where most of the rules in [docs/labeling_guide.md](docs/labeling_guide.md) came from. For example: "less than 250 cc" counts as 250, fluids and urine output are not blood loss, risks listed for consent are not complications, and when the heading and the narrative disagree I follow the narrative and mark the note as unsure.

To finish in reasonable time, I had Claude draft labels for the other 62 notes using that guide, and I reviewed every one in the app before saving it. The app records where each label came from and whether I changed the draft. I changed 1 of the 62: Claude had marked "no complications" for a spine surgery where the note says "Complications: None" in the heading, but the narrative describes a dural tear that was repaired during the operation. That is a complication, so I changed it to "any".

I'm being upfront about this because it matters for the results. Claude is not the model being evaluated, but reviewers tend to accept drafts, and AI drafts may share blind spots with AI extractors. So the scores on those 62 notes may lean slightly in the LLM's favor. A second, independent human labeler would make the reference stronger.

## How to run

You need Python 3.10+, and [Ollama](https://ollama.com) for the LLM step.

```bash
pip install -r requirements.txt
```

**Data.** Download `mtsamples.csv` from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and put it in `data/raw/`.

**Clean, audit, detect, run the rules, pick the gold notes** (about 2 minutes):
```bash
./run_all.sh prepare
```

**Label the gold notes** (my labels are already in `data/gold/`):
```bash
streamlit run label_app.py
```
Each save goes straight to `data/gold/gold_labels.csv`, so you can stop and come back later. The app never shows what the rules or the LLM answered, so the labels stay independent of the methods being tested. If `data/gold/draft_labels.csv` exists, the app pre-fills the form with the draft and its reasoning; nothing counts until you review it and save.

**Run the LLM.** Pick a model that fits your memory:

| Memory | Model | Download |
|---|---|---|
| 8 GB | `qwen2.5:3b` (what I used) | about 2 GB |
| 16 GB or more | `qwen2.5:7b` (script default) | about 5 GB |

```bash
ollama pull qwen2.5:3b
python src/extract_llm.py --model qwen2.5:3b --prompt v1
python src/extract_llm.py --model qwen2.5:3b --prompt v2
```
If a run stops, run the same command again and it picks up where it left off. Each model and prompt combination is saved as its own run, so to try a new prompt, copy `prompts/v2.md` to `prompts/v3.md`, edit it, and run with `--prompt v3`.

**Evaluate and explore:**
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
prompts/v1.md, v2.md        LLM instructions (v2 makes "not stated" the default)
docs/labeling_guide.md      how each field is labeled
reports/                    generated reports
data/gold/                  gold sample, labels and Claude's drafts (committed)
data/outputs/               extractor outputs and LLM run log (committed)
data/raw/, data/processed/  not committed; rebuild with the scripts
```

## Choices I made along the way

- One schema for everything. `src/common.py` defines the fields and allowed values. The LLM gets it as a JSON schema (Ollama structured outputs), and every answer is checked again, with problems logged rather than dropped silently.
- The labeling guide and the LLM prompt describe each field the same way, so the model and I were working from the same rules.
- Results are split by note group, because an overall number hides the fact that the rules and the LLM fail on different kinds of notes.
- Accuracy comes with 95% bootstrap confidence intervals. With 100 notes the intervals are wide, and I'd rather show that.
- Runs are repeatable: temperature 0, fixed seed, fixed random samples.

## Related work

- Specialty classification is a popular task on MTSamples. A 2026 preprint, [CLiGNet](https://arxiv.org/abs/2603.22752) (Barman and Barman), reports that earlier studies claimed accuracy above 90%, but that those scores might have inflated by oversampling (SMOTE) the data before splitting it. Without that leak, their best model reaches 36% accuracy on 40 specialties. My leakage audit points to a second, separate problem: duplicate notes filed under several specialties.
- Hu et al. (2024), [Information Extraction from Clinical Notes: Are We Ready to Switch to Large Language Models?](https://arxiv.org/abs/2411.10020), compare LLMs on entity and relation extraction using notes from MTSamples, MIMIC-III, i2b2 and UT Physicians.
- Yim et al. (2017), [Annotation of pain and anesthesia events for surgery-related processes and outcomes extraction](https://aclanthology.org/W17-2325/) (BioNLP workshop), propose an annotation schema for pain, pain management and anesthesia events. Their corpus includes MTSamples operative and clinic notes alongside hospital notes.
- LLMs have been tested on real operative notes in [thoracic surgery](https://jtd.amegroups.org/article/view/108599/html) (Mugu et al., 2025: extracting key elements from 116 Mayo Clinic notes with Gemini, ChatGPT and LLaMA) and [orthopedics](https://www.medrxiv.org/content/10.1101/2025.08.19.25333995) (2025 preprint: classifying hip-preservation procedures from Boston Children's Hospital notes).

## Limitations

- I don't have clinical training, and there is no second labeler to measure agreement.
- 62 of the gold labels started as AI drafts (see above).
- 100 notes is a small sample, and prompt v2 was tuned on the same notes it was scored on.
- MTSamples notes are transcription examples, not real hospital records, and real operative notes may look different.

## Data and credit

[MTSamples](https://www.mtsamples.com) transcribed medical reports, as published on [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) by Tara Boyle under a CC0 public domain license.
