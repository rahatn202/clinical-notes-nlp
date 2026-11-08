#!/usr/bin/env bash
# Run the whole pipeline. Labeling (step 4b) is done by hand in between.
#   ./run_all.sh prepare    steps 1-4a: clean, audit, detector, rules, choose gold notes
#   ./run_all.sh evaluate   steps 3b-5a: LLM extraction on the gold notes, then evaluation
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  prepare)
    python src/prepare_data.py
    python src/leakage_audit.py
    python src/anesthesia_detector.py
    python src/extract_rules.py
    python src/sample_gold.py
    echo "Next: label the gold notes with  streamlit run label_app.py" ;;
  evaluate)
    python src/extract_llm.py "${@:2}"
    python src/evaluate.py
    echo "Next: open the dashboard with  streamlit run app.py" ;;
  *) echo "Usage: ./run_all.sh prepare | evaluate [--model NAME --prompt VERSION]"; exit 1 ;;
esac
