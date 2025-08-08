# app.py
import streamlit as st
import torch
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download
import os

# ========================
# 1. Page Config
# ========================
st.set_page_config(
    page_title="Computational Argumentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Computational Argumentation Dashboard")
st.markdown("### Explore argument classification & stance detection interactively")

# ========================
# 2. Load Models & Data from Hugging Face Hub
# ========================
@st.cache_resource
def load_models_and_embeddings():
    repo_id = "YOUR_USERNAME/computational-argumentation-assets"  # TODO: update with your HF repo
    hf_token = st.secrets["HF_TOKEN"]

    # --- Download files from HF Hub (cached automatically) ---
    model1_path = hf_hub_download(repo_id=repo_id, filename="model1.pt", token=hf_token)
    model2_path = hf_hub_download(repo_id=repo_id, filename="model2.pt", token=hf_token)
    embeddings_path = hf_hub_download(repo_id=repo_id, filename="embeddings_metadata.pkl", token=hf_token)

    # --- Load PyTorch models ---
    model1 = torch.load(model1_path, map_location="cpu")
    model1.eval()
    model2 = torch.load(model2_path, map_location="cpu")
    model2.eval()

    # --- Load embeddings + metadata from pickle ---
    with open(embeddings_path, "rb") as f:
        embeddings_metadata = pickle.load(f)

    return model1, model2, embeddings_metadata

model1, model2, embeddings_metadata = load_models_and_embeddings()

# ========================
# 3. Sidebar Controls
# ========================
st.sidebar.header("⚙️ Settings")

input_mode = st.sidebar.radio("Select input mode", ["Single Comment", "Batch Upload"])
top_k = st.sidebar.slider("Top-k theses retrieval", min_value=1, max_value=10, value=3, step=1)
num_args = st.sidebar.slider("Arguments per thesis", min_value=1, max_value=10, value=3, step=1)

manual_thesis = st.sidebar.checkbox("Manually provide thesis")
manual_thesis_text = None
if manual_thesis:
    manual_thesis_text = st.sidebar.text_area("Enter thesis text", "")

# ========================
# 4. Inference Functions (placeholders for now)
# ========================
def classify_argumentative(text):
    # TODO: Replace with real Model 1 first-head inference
    return "Argumentative", 0.92

def retrieve_theses(text, top_k):
    # TODO: Replace with real Model 2 + embedding retrieval from embeddings_metadata
    theses = [{"id": i, "text": f"Sample Thesis {i+1}"} for i in range(top_k)]
    return theses

def classify_stance(comment, thesis, num_args):
    # TODO: Replace with real stance classification logic
    args = []
    for i in range(num_args):
        args.append({
            "retrieved_argument": f"Argument {i+1} text",
            "retrieved_stance": "Pro" if i % 2 == 0 else "Con",
            "predicted_relation": "Same-side" if i % 2 == 0 else "Opposing-side",
            "predicted_stance": "Pro" if i % 2 == 0 else "Con",
            "confidence": round(0.8 - i*0.05, 2)
        })
    return args

# ========================
# 5. Main Logic
# ========================
if input_mode == "Single Comment":
    user_input = st.text_area("Enter your comment", height=150)
    if st.button("Run Analysis") and user_input.strip():
        # Task 1: Argumentative classification
        arg_class, arg_conf = classify_argumentative(user_input)
        st.subheader("Task 1: Argumentative Classification")
        color = "green" if arg_class == "Argumentative" else "red"
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{arg_class}</span> "
                    f"(Confidence: {arg_conf:.2f})", unsafe_allow_html=True)

        # Task 2: Stance classification if argumentative
        if arg_class == "Argumentative":
            if manual_thesis and manual_thesis_text.strip():
                theses = [{"id": "manual", "text": manual_thesis_text.strip()}]
            else:
                theses = retrieve_theses(user_input, top_k)

            st.subheader("Task 2: Stance Classification")
            for thesis in theses:
                with st.expander(f"📜 {thesis['text']}"):
                    stance_results = classify_stance(user_input, thesis, num_args)
                    df = pd.DataFrame(stance_results)
                    st.dataframe(df)

elif input_mode == "Batch Upload":
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_json(uploaded_file)

        st.write("### Uploaded Data Preview")
        st.dataframe(df_input.head())

        if st.button("Run Batch Analysis"):
            st.info("Batch processing not yet implemented in this skeleton.")