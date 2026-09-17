import streamlit as st
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import plotly.graph_objects as go
import os

MODEL_NAME = "facebook/nllb-200-distilled-600M"
# Adjust this path to wherever your actual lsftl-project checkpoints live
ADAPTER_PATH = Path("/raid/home/loitongbam/Shivam-PhD/lsftl-project/checkpoints/lsftl-hi-ms/final")

SRC_LANG = "hin_Deva"
TGT_LANG = "zsm_Latn"

@st.cache_resource

def load_models():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run on CPU.")

    device = torch.device("cuda")
    print(f"[LSFTL Dashboard] Using device: {device}")
    print(f"[LSFTL Dashboard] GPU name: {torch.cuda.get_device_name(device)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False).to(device)
    base_model.eval()

    lora_base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)
    lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_PATH).to(device)
    lora_model.eval()

    return base_model, lora_model, tokenizer, device

def translate(model, tokenizer, device, text):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    target_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    with torch.no_grad():
        generated = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_length=128)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def render():
    st.title("LSFTL: Language-Specific Fine-Tuning with LoRA")
    st.caption("Implementation of Liang et al., 2025 — IEEE Access")

    tab1, tab2, tab3, tab4 = st.tabs(["Paper Overview", "Dataset", "Results", "Live Inference"])

    with tab1:
        st.header("Paper Overview")
        st.markdown("""
        **Title:** Toward Low-Resource Languages Machine Translation: A Language-Specific
        Fine-Tuning With LoRA for Specialized Large Language Models

        **Authors:** Xiao Liang, Yen-Min Jasmina Khaw, Soung-Yue Liew, Tien-Ping Tan, Donghong Qin

        **Published:** IEEE Access, 2025
        """)

        st.subheader("Key Contributions")
        st.markdown("""
        1. **Targeted fine-tuning methodology** — adjusts model parameters at specific
        locations to align with each target language's linguistic characteristics
        2. **Systematic layer/module analysis** — investigates which specific Transformer
        layers and modules are most effective for low-resource translation quality
        3. **Efficiency vs. quality trade-off study** — compares LSFTL against traditional
        full fine-tuning in both translation quality and computational cost
        """)

        st.subheader("Key Figure: LSFTL Adapter Structure")
        st.image("papers/lsftl/assets/figure3.png",
                caption="Model Structure of LSFTL Adapters (Liang et al., 2025)")

        st.subheader("Limitations (as stated by the authors)")
        st.markdown("""
        - For **extremely low-resource languages** (fewer than 10,000 parallel sentences),
        LSFTL may still struggle to capture language-specific nuances adequately
        - As the **number of language pairs increases**, maintaining separate adapters per
        pair grows in computational complexity, potentially limiting scalability for
        very large multilingual systems
        """)
    with tab2:
        st.header("Dataset")
        st.markdown("""
        **Source:** OpenSubtitles (OPUS)
        **Language pair implemented:** Hindi (hi) → Malay (ms)
        **Cleaned & subsampled:** 20,000 sentence pairs
        **Split:** 16,000 train / 2,000 validation / 2,000 test
        """)
        st.info("Additional language pairs (hi-th, hi-vi, ms-th, ms-vi, th-vi) in progress.")

    with tab3:
        st.header("Results — Baseline vs LSFTL")

        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU", "12.35", "+6.98 vs baseline")
        col2.metric("chrF", "33.12", "+7.57 vs baseline")
        col3.metric("COMET (x100)", "69.91", "+7.71 vs baseline")

        metrics = ["BLEU", "chrF", "COMET (x100)"]
        baseline_scores = [5.37, 25.55, 62.20]
        lsftl_scores = [12.35, 33.12, 69.91]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline (no LoRA)", x=metrics, y=baseline_scores))
        fig.add_trace(go.Bar(name="LSFTL (LoRA fine-tuned)", x=metrics, y=lsftl_scores))
        fig.update_layout(
            barmode="group",
            title="Baseline vs LSFTL: Hindi-Malay Translation Quality",
            yaxis_title="Score",
            xaxis_title="Metric",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trainable Parameters")
        st.write("LoRA trainable params: **8,650,752** / 623,724,544 total (**1.39%**)")

    with tab4:
        st.header("Live Inference: Hindi → Malay")
        st.markdown("Compare the base model's zero-shot output against our LSFTL fine-tuned adapter.")

        user_input = st.text_input("Enter a Hindi sentence:", "नमस्ते, आप कैसे हैं?")

        if st.button("Translate"):
            with st.spinner("Loading models and translating..."):
                base_model, lora_model, tokenizer, device = load_models()
                base_output = translate(base_model, tokenizer, device, user_input)
                lora_output = translate(lora_model, tokenizer, device, user_input)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline (no LoRA)")
                st.success(base_output)
            with col2:
                st.subheader("LSFTL (LoRA fine-tuned)")
                st.success(lora_output)
                