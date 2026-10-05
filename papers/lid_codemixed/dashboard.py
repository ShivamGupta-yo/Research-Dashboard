import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# --------------------------------------------------------------------------- #
# Paths (adjust if the project moves)
# --------------------------------------------------------------------------- #
LID_ROOT = Path("/raid/home/loitongbam/Shivam-PhD/lid-codemixed")
RESULTS_ROOT = LID_ROOT / "outputs" / "results"
CKPT_ROOT = LID_ROOT / "outputs" / "checkpoints"
DEFAULT_EXPERIMENT = "lince_hineng_mbert_bilstm_esf1"
# located relative to this file, so it works from any launch directory
FIGURE = str(Path(__file__).parent / "assets" / "lid_fig9_architecture.png")

STOP_LABELS = {"val_loss": "val loss", "val_weighted_f1": "val weighted F1",
               "val_macro_f1": "val macro F1"}

# Reuse the project's own model/data code so inference matches training
if str(LID_ROOT) not in sys.path:
    sys.path.insert(0, str(LID_ROOT))

# --------------------------------------------------------------------------- #
# Reference numbers from the paper (Chanda & Pal, 2025)
# --------------------------------------------------------------------------- #
PAPER_TABLE13 = [  # LinCE HI-EN development data, F1 (%)
    ("CharRNN", "CRF", 84.80, 79.67),
    ("GloVe", "Bi-LSTM", 89.69, 87.16),
    ("BERT-base-cased", "Bi-LSTM", 95.79, 94.48),
    ("BERT-base-uncased", "Bi-LSTM", 96.23, 94.16),
    ("mBERT-base-cased", "Bi-LSTM", 96.72, 94.78),
    ("XLM-RoBERTa-base", "Bi-LSTM", 96.70, 94.44),
]
PAPER_BEST = 96.72
PAPER_PER_TAG_F1 = {"lang1": 98.0, "lang2": 95.0, "other": 99.0, "ne": 81.0}  # Fig. 24

TAG_INFO = {  # LinCE HI-EN: lang1 = English, lang2 = Hindi
    "lang1": ("English", "#3b82f6"),
    "lang2": ("Hindi", "#22c55e"),
    "other": ("other", "#94a3b8"),
    "ne": ("named entity", "#f59e0b"),
    "fw": ("foreign word", "#a855f7"),
    "mixed": ("mixed", "#ec4899"),
    "ambiguous": ("ambiguous", "#64748b"),
    "unk": ("unknown", "#475569"),
}


# --------------------------------------------------------------------------- #
# Loading results
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)  # re-scan every minute so new runs appear
def discover_experiments():
    """Every results folder that contains a summary.json, default first."""
    if not RESULTS_ROOT.exists():
        return []
    exps = sorted(d.name for d in RESULTS_ROOT.iterdir()
                  if (d / "summary.json").exists())
    if DEFAULT_EXPERIMENT in exps:
        exps.remove(DEFAULT_EXPERIMENT)
        exps.insert(0, DEFAULT_EXPERIMENT)
    return exps


@st.cache_data(ttl=60)
def load_results(exp):
    res_dir = RESULTS_ROOT / exp
    summary_file = res_dir / "summary.json"
    if not summary_file.exists():
        return None, {}
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    seeds = {}
    for sd in summary["seeds"]:
        f = res_dir / f"seed_{sd}.json"
        if f.exists():
            seeds[sd] = json.loads(f.read_text(encoding="utf-8"))
    return summary, seeds


def stop_metric(summary):
    # runs made before the option existed have no key -> they used val loss
    return summary.get("early_stop_metric", "val_loss")


def exp_label(exp):
    summary, _ = load_results(exp)
    return f"{exp}  (stop on {STOP_LABELS[stop_metric(summary)]})"


# --------------------------------------------------------------------------- #
# Live inference
# --------------------------------------------------------------------------- #
@st.cache_resource  # one cached model per (experiment, seed)
def load_tagger(exp, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run on CPU.")
    device = torch.device("cuda")

    from transformers import AutoTokenizer
    from src.model import build_model

    ckpt_path = CKPT_ROOT / exp / f"seed_{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, label2id = ckpt["config"], ckpt["label2id"]
    id2label = {i: t for t, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = build_model(cfg, len(label2id))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[LID Dashboard] GPU: {torch.cuda.get_device_name(device)}")
    return model, tokenizer, label2id, id2label, device


def split_words(text):
    """LinCE is pre-tokenised with punctuation as separate tokens, so we
    split the same way: runs of word characters, or runs of punctuation."""
    return re.findall(r"\w+|[^\w\s]+", text)


def tag_sentence(text, exp, seed):
    from src.data import Collator, encode_example

    model, tokenizer, label2id, id2label, device = load_tagger(exp, seed)
    words = split_words(text)
    if not words:
        return []
    # labels are required by encode_example but ignored for prediction
    dummy = {"id": "input", "words": words,
             "labels": [id2label[0]] * len(words)}
    enc = encode_example(dummy, tokenizer, label2id)
    if len(enc["input_ids"]) > 512:
        raise ValueError("Sentence too long for mBERT (over 512 subtokens).")

    batch = Collator(tokenizer.pad_token_id)([enc])
    batch = {k: v.to(device) for k, v in batch.items() if k != "ids"}
    with torch.no_grad():
        probs = torch.softmax(model(**batch)["logits"][0], dim=-1)
    conf, pred = probs.max(dim=-1)
    return [(w, id2label[int(p)], float(c))
            for w, p, c in zip(words, pred.tolist(), conf.tolist())]


def chips_html(tagged):
    """Word chips with the tag above each word, like Fig. 1 of the paper."""
    parts = []
    for word, tag, conf in tagged:
        _, color = TAG_INFO.get(tag, (tag, "#64748b"))
        parts.append(
            f"<div style='display:inline-block;text-align:center;margin:6px'>"
            f"<div style='background:{color};color:white;border-radius:6px;"
            f"padding:2px 8px;font-size:0.8em'>{tag}</div>"
            f"<div style='font-size:1.15em;margin-top:3px'>{word}</div>"
            f"<div style='font-size:0.7em;opacity:0.6'>{conf:.2f}</div></div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render():
    st.title("Word-Level Language Identification in Code-Mixed Text")
    st.caption("Implementation of Chanda & Pal, 2025 — SN Computer Science")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Paper Overview", "Dataset", "Results", "Live Inference"])

    experiments = discover_experiments()
    if experiments:
        exp = st.sidebar.selectbox("Experiment:", experiments,
                                   format_func=exp_label)
        summary, seeds = load_results(exp)
    else:
        exp, summary, seeds = None, None, {}

    # ------------------------------------------------------------------ #
    with tab1:
        st.header("Paper Overview")
        st.markdown("""
        **Title:** Word Level Language Identification from Social Media Code-Mixed
        Data Leveraging Transformer-Based Models

        **Authors:** Supriya Chanda, Sukomal Pal

        **Published:** SN Computer Science 6:844, 2025 (doi: 10.1007/s42979-025-04377-4)
        """)

        st.subheader("Key Contributions")
        st.markdown("""
        1. **Word-level LID as token classification** — every word in a code-mixed
        social-media sentence receives a language tag
        2. **Contextual vs non-contextual representations** — compares Word2Vec, GloVe
        and FastText baselines against BERT-family encoders
        3. **BERT + Bi-LSTM tagger** — contextual BERT/mBERT representations passed
        through a Bi-LSTM and a softmax layer
        4. **Evaluation on six datasets, three language pairs** — Bengali-English,
        Hindi-English and Spanish-English (ICON_POS, ICON_SAIL, LinCE), with and
        without monolingual training sentences
        """)

        st.subheader("Key Figure: Model Architecture")
        st.image(FIGURE, caption="BERT + Bi-LSTM architecture (Chanda & Pal, 2025, Fig. 9)")

        st.subheader("Limitations (as stated by the authors)")
        st.markdown("""
        - Low performance on **emoticons** and on **mixing within a single word**
        (e.g. an English name with a Bengali suffix)
        - Performance on **universal tags** is close to that of alternative methods
        - **Bengali named entities** are sometimes confused with Bengali-language tags
        - Adding the **Bi-LSTM gives only small gains**, suggesting diminishing returns
        from extra model complexity
        """)

        st.subheader("Replication Notes")
        st.markdown("""
        - Abstract "gains" are **relative** percentages, not F1 points
        (e.g. LinCE HI-EN: 7.83% relative = 7.03 points over GloVe)
        - LinCE results are on the **dev set** and early stopping appears to use the
        same data; our replication early-stops on a **held-out 10% of train** instead
        - The paper states two different learning rates / epoch limits for BERT models
        - The paper reports **single runs**; we report mean ± std over 3 seeds
        - Early stopping on **val loss** gave 96.45 ± 0.02 (best epochs 4–5); on
        **val weighted F1** it gave 96.74 ± 0.11 (best epochs 9–12), matching the
        paper's 96.72, which it reports after 11 epochs. Val loss keeps rising while
        F1 still improves (growing overconfidence), so the gap came from the
        stopping rule, not the model or data
        """)

    # ------------------------------------------------------------------ #
    with tab2:
        st.header("Dataset")
        st.markdown("""
        **Source:** LinCE benchmark — Hindi-English word-level LID
        (HuggingFace mirror: `HuggMachas/Lince_benchmark_lid_hineng_train_valid`)
        **Tags:** `lang1` (English), `lang2` (Hindi), `other`, `ne`, `fw`, `mixed`,
        `unk`, `ambiguous`
        **Test labels** are not public, so (as in the paper) results are on the dev set.
        """)

        st.subheader("Verification against the paper (Table 1)")
        st.dataframe(pd.DataFrame({
            "Split": ["Train", "Dev"],
            "Sentences (ours)": [4823, 744], "Sentences (paper)": [4823, 744],
            "Tokens (ours)": [95224, 15446], "Tokens (paper)": [95224, 15446],
        }), hide_index=True, use_container_width=True)
        st.caption("Dev tag counts match Table 7 except one token "
                   "(ours: other 2231 / ambiguous 1; paper: 2230 / 2).")

        st.subheader("Our protocol")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train", "4,339", "used for learning", delta_color="off")
        c2.metric("Validation", "482", "early stopping only", delta_color="off")
        c3.metric("Dev", "744", "final evaluation only", delta_color="off")
        st.markdown("Two training sentences (ids 217 and 1902) exceeded mBERT's "
                    "512-subtoken limit due to long runs of quotation marks and were "
                    "removed. The dev set is unchanged.")

        if seeds:
            first = next(iter(seeds.values()))
            train_counts = first["data_stats"]["train_label_counts"]
            dev_counts = {t: c["support"]
                          for t, c in first["dev_metrics"]["per_class"].items()}
            tags = sorted(dev_counts, key=lambda t: -dev_counts[t])
            st.subheader("Tag distribution")
            st.dataframe(pd.DataFrame({
                "Tag": tags,
                "Meaning": [TAG_INFO.get(t, (t, ""))[0] for t in tags],
                "Train (after split)": [train_counts.get(t, 0) for t in tags],
                "Dev": [dev_counts[t] for t in tags],
            }), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------ #
    with tab3:
        st.header("Results — Replication vs Paper (LinCE HI-EN)")
        if summary is None:
            st.warning(f"No results found in {RESULTS_ROOT}")
        else:
            st.subheader("All runs")
            rows = []
            for e in experiments:
                sm, _ = load_results(e)
                rows.append({
                    "Experiment": e,
                    "Stopping metric": STOP_LABELS[stop_metric(sm)],
                    "Seeds": len(sm["seeds"]),
                    "Weighted F1": f"{sm['weighted_f1']['mean']:.2f} ± "
                                   f"{sm['weighted_f1']['std']:.2f}",
                    "vs paper": round(sm["weighted_f1"]["mean"] - PAPER_BEST, 2),
                    "Macro F1": f"{sm['macro_f1']['mean']:.2f} ± "
                                f"{sm['macro_f1']['std']:.2f}",
                    "Accuracy": f"{sm['accuracy']['mean']:.2f} ± "
                                f"{sm['accuracy']['std']:.2f}",
                    "Best epochs": ", ".join(str(r["best_epoch"])
                                             for r in sm["per_seed"].values()),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.caption("Change the run shown below with the Experiment "
                       "selector in the sidebar.")

            st.subheader(f"Selected run: {exp}")
            st.caption(f"Early stopping on {STOP_LABELS[stop_metric(summary)]}; "
                       "all numbers are on the LinCE HI-EN dev set.")
            wf1, mf1, acc = (summary["weighted_f1"], summary["macro_f1"],
                             summary["accuracy"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Weighted F1 (ours)",
                      f"{wf1['mean']:.2f} ± {wf1['std']:.2f}",
                      f"{wf1['mean'] - PAPER_BEST:+.2f} vs paper ({PAPER_BEST})")
            c2.metric("Macro F1 (ours)", f"{mf1['mean']:.2f} ± {mf1['std']:.2f}",
                      "not reported in paper", delta_color="off")
            c3.metric("Accuracy (ours)", f"{acc['mean']:.2f} ± {acc['std']:.2f}")

            st.subheader("Paper Table 13 vs our replication")
            rows = [{"Embedding": e, "Model": m, "F1 ALL (%)": a, "F1 CM (%)": c,
                     "Source": "paper"} for e, m, a, c in PAPER_TABLE13]
            rows.append({"Embedding": "mBERT-base-cased", "Model": "Bi-LSTM",
                         "F1 ALL (%)": round(wf1["mean"], 2), "F1 CM (%)": None,
                         "Source": f"ours ({len(seeds)} seeds, stop on "
                                   f"{STOP_LABELS[stop_metric(summary)]})"})
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

            st.subheader("Per-seed results")
            st.dataframe(pd.DataFrame([
                {"Seed": s, "Weighted F1": round(100 * r["weighted_f1"], 2),
                 "Macro F1": round(100 * r["macro_f1"], 2),
                 "Accuracy": round(100 * r["accuracy"], 2),
                 "Best epoch": r["best_epoch"]}
                for s, r in summary["per_seed"].items()]),
                hide_index=True, use_container_width=True)

            if seeds:
                # per-tag F1, averaged over seeds
                tags = list(PAPER_PER_TAG_F1)
                ours = [100 * sum(r["dev_metrics"]["per_class"][t]["f1"]
                                  for r in seeds.values()) / len(seeds)
                        for t in tags]
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Paper (mBERT + Bi-LSTM, Fig. 24)",
                                     x=tags, y=[PAPER_PER_TAG_F1[t] for t in tags]))
                fig.add_trace(go.Bar(name=f"Ours (mean of {len(seeds)} seeds)",
                                     x=tags, y=ours))
                fig.update_layout(
                    barmode="group", title="Per-tag F1: paper vs replication",
                    yaxis_title="F1 (%)", xaxis_title="Tag",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="center", x=0.5))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The paper only plots these four tags; fw, mixed, unk and "
                           "ambiguous have 1–29 dev tokens each.")

                st.subheader("Training curves (held-out validation split)")
                metric = st.radio("Metric", ["val_loss", "val_weighted_f1"],
                                  horizontal=True)
                fig = go.Figure()
                for s, r in seeds.items():
                    h = r["history"]
                    y = [e[metric] * (100 if "f1" in metric else 1) for e in h]
                    fig.add_trace(go.Scatter(
                        x=[e["epoch"] for e in h], y=y, mode="lines+markers",
                        name=f"seed {s}"))
                    b = r["best_epoch"]
                    fig.add_trace(go.Scatter(
                        x=[b], y=[y[b - 1]], mode="markers",
                        marker=dict(symbol="star", size=16),
                        name=f"seed {s}: kept epoch {b}", showlegend=True))
                fig.update_layout(xaxis_title="Epoch", yaxis_title=metric)
                st.caption("★ = epoch kept by early stopping (evaluated on dev).")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Confusion matrix (dev)")
                seed = st.selectbox("Seed", list(seeds))
                cm = seeds[seed]["dev_metrics"]["confusion_matrix"]
                fig = go.Figure(go.Heatmap(
                    z=cm["matrix"], x=cm["labels"], y=cm["labels"],
                    text=cm["matrix"], texttemplate="%{text}",
                    colorscale="Blues"))
                fig.update_layout(xaxis_title="Predicted", yaxis_title="Gold",
                                  yaxis_autorange="reversed", height=550)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Weighted vs macro F1")
            st.markdown("""
            **Weighted F1** averages per-tag F1 weighted by how many words each tag has,
            so it is dominated by `lang1`, `lang2` and `other`. This is the metric the
            paper most likely reports. **Macro F1** gives every tag equal weight, so the
            rare tags (`fw`, `mixed`, `unk`, `ambiguous`), which the model almost never
            predicts, pull it down sharply.
            """)

    # ------------------------------------------------------------------ #
    with tab4:
        st.header("Live Inference: Hindi-English word tagging")
        st.markdown("Type a Hinglish sentence (Roman script). Each word is tagged by "
                    "the trained mBERT + Bi-LSTM model (experiment chosen in the "
                    "sidebar); the number under "
                    "each word is the model's confidence.")

        ckpt_seeds = sorted(int(p.stem.split("_")[1])
                            for p in (CKPT_ROOT / exp).glob("seed_*.pt")) \
            if exp else []
        if not ckpt_seeds:
            st.warning("No checkpoints found for this experiment.")
        seed = st.selectbox("Checkpoint (seed):", ckpt_seeds) \
            if ckpt_seeds else None

        user_input = st.text_input("Enter a sentence:",
                                   "Virat so nice player yaar , kya batting ki aaj")

        if st.button("Tag") and seed is not None:
            with st.spinner("Loading model and tagging..."):
                try:
                    tagged = tag_sentence(user_input, exp, seed)
                except (RuntimeError, ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    tagged = []
            if tagged:
                st.markdown(chips_html(tagged), unsafe_allow_html=True)
                st.caption("lang1 = English, lang2 = Hindi")