"""Step 3b: LLM extractor using a local model through Ollama.

Usage:
  ollama pull qwen2.5:7b                      # once
  python src/extract_llm.py                   # gold-sample notes, default model and prompt
  python src/extract_llm.py --model llama3.2:3b --prompt v2
  python src/extract_llm.py --notes candidates   # every extraction candidate (slow)

How it works:
  - The prompt (prompts/<version>.md) holds the extraction rules. The note is
    sent as the user message.
  - Ollama's "structured outputs" feature receives the JSON schema from
    common.py, so the model can only return the allowed keys and values.
  - Temperature 0 and a fixed seed make runs repeatable.
  - Every answer is checked again against the schema (validate_record), and
    problems are recorded instead of being hidden.
  - Results go to data/outputs/llm/<model>__<prompt>.jsonl, one line per note.
    If the run stops, running the same command again continues where it left off.
  - A summary of each run (success rate, speed) is added to data/outputs/runs.csv,
    which the dashboard uses for monitoring.

Nothing leaves your computer: Ollama runs locally.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import (GOLD_SAMPLE, JSON_SCHEMA, OUTPUTS, PROCESSED, PROMPTS, RUNS_LOG,  # noqa: E402
                    append_jsonl, load_notes, read_jsonl, validate_record)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_PROMPT = "v1"
DEFAULT_HOST = "http://localhost:11434"


def run_id(model: str, prompt: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{model}__{prompt}")


def check_server(host: str, model: str) -> None:
    try:
        tags = requests.get(f"{host}/api/tags", timeout=5).json()
    except requests.RequestException:
        sys.exit(f"Cannot reach Ollama at {host}. Start the Ollama app (or run `ollama serve`).")
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        have = ", ".join(sorted(names)) or "none"
        sys.exit(f"Model {model!r} is not downloaded. Run: ollama pull {model}\n"
                 f"Models you have: {have}")


def ask(host: str, model: str, system_prompt: str, note: str) -> tuple[str, float, dict]:
    """Send one note; return (raw answer text, seconds, extra stats)."""
    body = {"model": model, "stream": False, "format": JSON_SCHEMA,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": f"Note:\n{note}"}]}
    start = time.perf_counter()
    resp = requests.post(f"{host}/api/chat", json=body, timeout=600)
    resp.raise_for_status()
    seconds = time.perf_counter() - start
    data = resp.json()
    stats = {"prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
    return data["message"]["content"], seconds, stats


def select_notes(which: str) -> pd.DataFrame:
    notes = load_notes()
    if which == "gold":
        if not GOLD_SAMPLE.exists():
            sys.exit("No gold sample yet. Run: python src/sample_gold.py")
        ids = pd.read_csv(GOLD_SAMPLE)["note_id"]
        return notes[notes["note_id"].isin(ids)]
    scores = PROCESSED / "detector_scores.csv"
    if not scores.exists():
        sys.exit("Run python src/anesthesia_detector.py first.")
    cand = pd.read_csv(scores)
    return notes.merge(cand, on="note_id").query("extraction_candidate")


def summarize_run(rid: str, model: str, prompt: str, recs: list[dict]) -> None:
    df = pd.DataFrame(recs)
    row = {"run_id": rid, "model": model, "prompt_version": prompt, "notes": len(df),
           "valid_json_rate": round(df["parse_ok"].mean(), 4),
           "schema_ok_rate": round((df["parse_ok"] & (df["n_problems"] == 0)).mean(), 4),
           "mean_seconds": round(df["seconds"].mean(), 2),
           "p95_seconds": round(df["seconds"].quantile(0.95), 2),
           "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    log = pd.read_csv(RUNS_LOG) if RUNS_LOG.exists() else pd.DataFrame()
    if len(log):
        log = log[log["run_id"] != rid]
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(RUNS_LOG, index=False)
    print(json.dumps(row, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt version: prompts/<version>.md")
    ap.add_argument("--notes", choices=["gold", "candidates"], default="gold")
    ap.add_argument("--limit", type=int, help="process at most this many new notes")
    ap.add_argument("--host", default=DEFAULT_HOST)
    args = ap.parse_args()

    prompt_file = PROMPTS / f"{args.prompt}.md"
    if not prompt_file.exists():
        sys.exit(f"Prompt not found: {prompt_file}")
    system_prompt = prompt_file.read_text()
    check_server(args.host, args.model)

    rid = run_id(args.model, args.prompt)
    out_path = OUTPUTS / "llm" / f"{rid}.jsonl"
    done = {r["note_id"] for r in read_jsonl(out_path)}
    todo = select_notes(args.notes)
    todo = todo[~todo["note_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Run {rid}: {len(done)} notes already done, {len(todo)} to go")

    for i, (_, note) in enumerate(todo.iterrows(), 1):
        try:
            raw, seconds, stats = ask(args.host, args.model, system_prompt, note["text"])
        except requests.RequestException as e:
            print(f"  {note['note_id']}: request failed ({e}); will retry next run")
            continue
        try:
            parsed, parse_ok = json.loads(raw), True
        except json.JSONDecodeError:
            parsed, parse_ok = None, False
        record, problems = validate_record(parsed)
        if not parse_ok:
            problems = ["answer is not valid JSON"]
        append_jsonl(out_path, {
            "note_id": note["note_id"], "method": "llm", "run_id": rid, "model": args.model,
            "prompt_version": args.prompt, "record": record, "raw": raw,
            "parse_ok": parse_ok, "problems": problems, "n_problems": len(problems),
            "seconds": round(seconds, 2), **stats})
        print(f"  [{i}/{len(todo)}] {note['note_id']} {seconds:.1f}s"
              + (f"  problems: {problems}" if problems else ""))

    recs = read_jsonl(out_path)
    if recs:
        summarize_run(rid, args.model, args.prompt, recs)


if __name__ == "__main__":
    main()
