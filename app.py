"""Step 5b: evaluation and monitoring dashboard.

Usage: streamlit run app.py   (after src/evaluate.py)

Tabs:
  Results     accuracy per field for each method, with confidence intervals
  Errors      every disagreement with the gold label, with the note text
  Monitoring  LLM run health: valid JSON, schema problems, speed
  Data        the data profile, leakage audit and detector reports
"""
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from common import (FIELD_LABELS, GOLD_LABELS, OUTPUTS, REPORTS, RULES_OUT,  # noqa: E402
                    RUNS_LOG, load_notes, read_jsonl)

# Colors from the project palette, checked for color-blind separation.
# Up to 3 methods at once; each bar also carries a text label.
SERIES = ["#b27450", "#713d43", "#a0b3b4"]  # terracotta, deep rose, blue-gray
NEUTRAL, TEXT, GRID = "#6e6a5f", "#1f1d1a", "rgba(110,106,95,0.15)"
ON_BAR = {"#b27450": "#000000", "#713d43": "#ffffff", "#a0b3b4": TEXT}  # label color inside each bar
MAX_METHODS = 3
HIGHLIGHT = re.compile(
    r"(anesthe\w*|sedation|sedated|\bMAC\b|\bLMA\b|laryngeal mask|intubat\w*|endotracheal|"
    r"nasotracheal|spinal|epidural|caudal|\bblock\b|lidocaine|marcaine|bupivacaine|local|"
    r"topical|blood loss|\bEBL\b|complications?)", re.IGNORECASE)

st.set_page_config(page_title="Anesthesia extraction: evaluation", page_icon=":stethoscope:",
                   layout="wide")


def style(fig, height):
    fig.update_layout(height=height, font_family="Source Sans Pro, sans-serif", font_color=TEXT,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=10, t=30, b=0), barmode="group", bargap=0.25,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, tickfont_color=NEUTRAL, title_font_color=NEUTRAL)
    fig.update_yaxes(gridcolor=GRID, tickfont_color=NEUTRAL, title_font_color=NEUTRAL,
                     zeroline=False)
    return fig


@st.cache_data
def load():
    metrics_path = REPORTS / "metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    errors = pd.read_csv(REPORTS / "errors.csv") if (REPORTS / "errors.csv").exists() \
        else pd.DataFrame()
    notes = load_notes().set_index("note_id")
    runs = pd.read_csv(RUNS_LOG) if RUNS_LOG.exists() else pd.DataFrame()
    rules = {r["note_id"]: r for r in read_jsonl(RULES_OUT)}
    llm = {p.stem: {r["note_id"]: r for r in read_jsonl(p)}
           for p in sorted((OUTPUTS / "llm").glob("*.jsonl"))}
    n_gold = len(pd.read_csv(GOLD_LABELS)) if GOLD_LABELS.exists() else 0
    return metrics, errors, notes, runs, rules, llm, n_gold


data = load()
st.title("Anesthesia information extraction")
st.markdown("Rule-based extractor vs. local LLM on public MTSamples operative notes, "
            "measured against a hand-labeled gold standard.")
if data is None:
    st.warning("No evaluation results yet. Run `python src/evaluate.py` first.")
    st.stop()
metrics, errors, notes, runs, rules, llm, n_gold = data

all_methods = list(dict.fromkeys(metrics["method"]))
methods = st.multiselect(f"Methods to compare (up to {MAX_METHODS})", all_methods,
                         default=all_methods[:MAX_METHODS], max_selections=MAX_METHODS)
if not methods:
    st.info("Pick at least one method.")
    st.stop()
color = {m: SERIES[i] for i, m in enumerate(methods)}

tab_res, tab_err, tab_mon, tab_data = st.tabs(
    [":material/bar_chart: Results", ":material/rule: Errors",
     ":material/monitor_heart: Monitoring", ":material/description: Data and models"])

# ---------------------------------------------------------------- results
with tab_res:
    acc = metrics[(metrics["metric"] == "accuracy") & (metrics["group"] == "all")
                  & metrics["method"].isin(methods)]
    cols = st.columns(len(methods))
    for c, m in zip(cols, methods):
        a = acc[acc["method"] == m].set_index("field")["value"]
        c.metric(m, f"{a.mean():.0%}", help="Mean accuracy over the four fields", border=True)

    st.markdown(f"**Accuracy by field**, {n_gold} gold notes, with 95% confidence intervals")
    fig = go.Figure()
    for m in methods:
        a = acc[acc["method"] == m].set_index("field").reindex(list(FIELD_LABELS))
        fig.add_bar(name=m, x=[FIELD_LABELS[f] for f in a.index], y=a["value"],
                    marker_color=color[m], text=[f"{v:.0%}" for v in a["value"]],
                    textposition="inside", insidetextanchor="start",
                    textfont=dict(color=ON_BAR[color[m]], size=13),
                    error_y=dict(type="data", symmetric=False, color=NEUTRAL, thickness=1.2,
                                 array=a["ci_high"] - a["value"],
                                 arrayminus=a["value"] - a["ci_low"]),
                    hovertemplate="%{x}: %{y:.0%}<extra>" + m + "</extra>")
    style(fig, 420)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(fig, width="stretch")
    st.caption("Lines on each bar show the 95% bootstrap confidence interval.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Anesthesia type: precision, recall, F1** (all types pooled)")
        micro = metrics[metrics["metric"].str.startswith("micro_") & metrics["method"].isin(methods)]
        t = micro.pivot(index="method", columns="metric", values="value").reindex(methods)
        t.columns = [c.removeprefix("micro_").capitalize() for c in t.columns]
        st.dataframe(t[["Precision", "Recall", "F1"]], width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Precision", "Recall", "F1"]})
    with c2:
        st.markdown("**Anesthesia type accuracy by note group**")
        g = metrics[(metrics["field"] == "anesthesia_types") & (metrics["metric"] == "accuracy")
                    & (metrics["group"] != "all") & metrics["method"].isin(methods)]
        t = g.pivot(index="group", columns="method", values="value")[methods]
        order = [x for x in ["heading", "narrative_only", "detector_only", "no_mention"] if x in t.index]
        n = g.drop_duplicates("group").set_index("group")["n"]
        t = t.reindex(order)
        t.index = [f"{i} ({int(n[i])})" for i in t.index]
        st.dataframe(t * 100, width="stretch",
                     column_config={m: st.column_config.NumberColumn(format="%.0f%%") for m in methods})

    st.markdown("**Per anesthesia type: F1**")
    per = metrics[metrics["field"].str.startswith("type:") & (metrics["metric"] == "f1")
                  & metrics["method"].isin(methods)].copy()
    per["type"] = per["field"].str.removeprefix("type:")
    n_true = per.drop_duplicates("type").set_index("type")["n"]
    t = per.pivot(index="type", columns="method", values="value")[methods]
    t.index = [f"{i} ({int(n_true[i])} in gold)" for i in t.index]
    st.dataframe(t, width="stretch",
                 column_config={m: st.column_config.NumberColumn(format="%.2f") for m in methods})

# ---------------------------------------------------------------- errors
with tab_err:
    if errors.empty:
        st.success("No disagreements with the gold labels.")
    else:
        e = errors[errors["method"].isin(methods)]
        f1, f2, f3 = st.columns(3)
        pick_m = f1.selectbox("Method", methods)
        pick_f = f2.selectbox("Field", ["all"] + list(FIELD_LABELS),
                              format_func=lambda f: "All fields" if f == "all" else FIELD_LABELS[f])
        groups = sorted(e["group"].unique())
        pick_g = f3.selectbox("Note group", ["all"] + groups)
        e = e[e["method"] == pick_m]
        if pick_f != "all":
            e = e[e["field"] == pick_f]
        if pick_g != "all":
            e = e[e["group"] == pick_g]
        st.caption(f"{len(e)} disagreements")
        view = e.assign(field=e["field"].map(FIELD_LABELS))[
            ["note_id", "group", "field", "gold", "predicted", "gold_unsure"]]
        event = st.dataframe(view, width="stretch", hide_index=True, on_select="rerun",
                             selection_mode="single-row",
                             column_config={"gold_unsure": st.column_config.CheckboxColumn("Gold unsure")})
        rows = event.selection.rows
        nid = view.iloc[rows[0]]["note_id"] if rows else (view.iloc[0]["note_id"] if len(view) else None)
        if nid is not None and nid in notes.index:
            st.divider()
            note = notes.loc[nid]
            st.markdown(f"#### {note['sample_name']}  \n:gray[`{nid}` · click a row above to change note]")
            left, right = st.columns([3, 2])
            text = html.escape(note["text"])
            text = HIGHLIGHT.sub(lambda m: f"<mark>{m.group(0)}</mark>", text).replace("\n", "<br>")
            left.markdown(f"<div style='max-height:520px;overflow-y:auto;padding:12px;border:1px "
                          f"solid rgba(128,128,128,.3);border-radius:8px;line-height:1.6'>{text}</div>",
                          unsafe_allow_html=True)
            with right:
                st.markdown("**Gold vs. predicted for this note**")
                st.dataframe(errors[(errors["note_id"] == nid) & errors["method"].isin(methods)]
                             .assign(field=lambda d: d["field"].map(FIELD_LABELS))
                             [["method", "field", "gold", "predicted"]],
                             hide_index=True, width="stretch")
                if nid in rules and rules[nid]["evidence"]:
                    st.markdown("**Text the rules matched**")
                    st.json(rules[nid]["evidence"], expanded=False)
                for run, recs in llm.items():
                    if f"llm: {run}" in methods and nid in recs:
                        st.markdown(f"**Raw LLM answer** ({run})")
                        st.code(recs[nid]["raw"], language="json", wrap_lines=True)
                        if recs[nid]["problems"]:
                            st.warning("Problems: " + "; ".join(recs[nid]["problems"]))

# ---------------------------------------------------------------- monitoring
with tab_mon:
    if runs.empty:
        st.info("No LLM runs yet. Run `python src/extract_llm.py`.")
    else:
        st.markdown("**LLM runs** (from `data/outputs/runs.csv`)")
        st.dataframe(runs, hide_index=True, width="stretch",
                     column_config={"valid_json_rate": st.column_config.NumberColumn("Valid JSON", format="percent"),
                                    "schema_ok_rate": st.column_config.NumberColumn("Schema OK", format="percent"),
                                    "mean_seconds": st.column_config.NumberColumn("Mean s/note", format="%.1f"),
                                    "p95_seconds": st.column_config.NumberColumn("95th pct s/note", format="%.1f")})
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Output health by run**")
            fig = go.Figure()
            for i, (col, name) in enumerate([("valid_json_rate", "Valid JSON"),
                                             ("schema_ok_rate", "Schema OK")]):
                fig.add_bar(name=name, x=runs["run_id"], y=runs[col], marker_color=SERIES[i],
                            text=[f"{v:.0%}" for v in runs[col]], textposition="outside",
                            textfont_color=TEXT)
            style(fig, 340)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
            st.plotly_chart(fig, width="stretch")
        with m2:
            st.markdown("**Seconds per note by run** (mean and 95th percentile)")
            fig = go.Figure()
            for i, (col, name) in enumerate([("mean_seconds", "Mean"), ("p95_seconds", "95th percentile")]):
                fig.add_bar(name=name, x=runs["run_id"], y=runs[col], marker_color=SERIES[i],
                            text=[f"{v:.1f}" for v in runs[col]], textposition="outside",
                            textfont_color=TEXT)
            style(fig, 340)
            st.plotly_chart(fig, width="stretch")
        problems = Counter(re.sub(r":.*", "", p) for recs in llm.values()
                           for r in recs.values() for p in r["problems"])
        st.markdown("**Most common output problems, all runs**")
        if problems:
            st.dataframe(pd.DataFrame(problems.most_common(), columns=["Problem", "Count"]),
                         hide_index=True)
        else:
            st.success("No output problems recorded.")

# ---------------------------------------------------------------- data
with tab_data:
    for name, title in [("data_profile.md", "Data cleaning"), ("leakage_audit.md", "Leakage audit"),
                        ("anesthesia_detector.md", "Anesthesia note detector")]:
        path = REPORTS / name
        if path.exists():
            with st.expander(title, expanded=name == "data_profile.md"):
                st.markdown(path.read_text().split("\n", 1)[1])

st.divider()
st.caption("Built by Rahat Naseem with Python, scikit-learn, Ollama and Streamlit. "
           "Data: MTSamples (CC0, Kaggle). Notes are transcription examples, not patient records.")
