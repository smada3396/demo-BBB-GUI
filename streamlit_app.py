import io
import zipfile
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

from bbb_model_predictor import BBBPredictor


# ---- Red Theme Palette ----
LIGHT_ROSE = "#FADBD8"
SOFT_RUBY = "#E57373"
RICH_CARMINE = "#C62828"
DEEP_CRIMSON = "#8E1C1C"
ACCENT_BLOOD = "#7A0E0E"
WHITE = "#FFFFFF"


st.set_page_config(page_title="BBB Permeability Studio", page_icon="🧠", layout="wide")

# Inject a compact red theme CSS
THEME_CSS = f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {LIGHT_ROSE} 0%, #ffd6d6 40%, #ffecec 100%);
}}
.block-container {{
    background-color: rgba(255,255,255,0.98);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(138,43,43,0.08);
}}
/* Buttons */
.stButton > button, .stDownloadButton > button {{
    background: {RICH_CARMINE} !important;
    color: {WHITE} !important;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-weight: 600;
}}
.stButton > button:hover {{
    background: {DEEP_CRIMSON} !important;
    transform: translateY(-1px);
}}
/* Sidebar */
[data-testid="stSidebar"] {{
    background: {ACCENT_BLOOD};
    color: {WHITE};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

h1, h2, h3, h4 {{ color: {RICH_CARMINE}; }}

/* File uploader: keep button text dark for contrast */
[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * {{ color: #000 !important; background: transparent !important; }}

</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def _load_predictor(artifacts_dir: str = "artifacts") -> BBBPredictor:
    return BBBPredictor(artifacts_dir=artifacts_dir)


def _predict_from_smiles(pred: BBBPredictor, smiles: List[str], threshold: float = 0.5) -> pd.DataFrame:
    try:
        df = pred.predict_proba(smiles)
        df["bbb_label"] = (df["p_bbb_plus"] >= threshold).astype(int)
        return df
    except Exception as e:
        raise


st.title("Blood–Brain Barrier (BBB) Permeability Studio")
st.caption("Upload SMILES or paste one per line — the model returns probability of BBB permeability.")

with st.sidebar:
    st.header("Controls")
    artifacts_path = st.text_input("Artifacts folder", value="artifacts", help="Path to trained model artifacts")
    threshold = st.slider("Classification threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.markdown("---")
    st.markdown("About: This UI runs the repository's `BBBPredictor` to score SMILES.")

tab = st.tabs(["Home", "Predict", "Documentation"])

with tab[0]:
    st.header("Welcome")
    st.write(
        "This lightweight app scores molecules for blood–brain barrier permeability using the project's pretrained ensemble."
    )
    st.write("Use the Predict tab to submit SMILES or upload a CSV with a `smiles` column.")

with tab[2]:
    st.header("Documentation")
    st.markdown("- Input: SMILES strings (one per line) or CSV with `smiles` column")
    st.markdown("- Output: `p_bbb_plus` probability, mechanistic stage-1 probs, and binary label at the chosen threshold.")

with tab[1]:
    st.header("Predict — submit molecules")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        smiles_text = st.text_area("SMILES (one per line)")
        uploaded = st.file_uploader("Or upload CSV (must contain `smiles` column)", type=["csv"], accept_multiple_files=False)
    with col_b:
        st.markdown("**Options**")
        show_raw = st.checkbox("Show raw probabilities", value=True)
        run_btn = st.button("Predict BBB", type="primary")

    smiles_list: List[str] = []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "smiles" not in df_up.columns:
                st.error("Uploaded CSV must contain a 'smiles' column.")
            else:
                smiles_list = df_up["smiles"].astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if smiles_text and not smiles_list:
        smiles_list = [s.strip() for s in smiles_text.splitlines() if s.strip()]

    if run_btn:
        if not smiles_list:
            st.error("Provide at least one SMILES string before predicting.")
        else:
            try:
                with st.spinner("Loading model artifacts…"):
                    predictor = _load_predictor(artifacts_dir=artifacts_path)
                with st.spinner("Generating predictions…"):
                    results = _predict_from_smiles(predictor, smiles_list, threshold=threshold)

                if show_raw:
                    st.subheader("Results (probabilities)")
                    st.dataframe(results, use_container_width=True)

                # Offer CSV download
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="bbb_predictions.csv")

                # Offer zipped single-file per molecule (PSEUDO: create per-smiles txt)
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for i, row in results.iterrows():
                        name = f"mol_{i+1}.txt"
                        txt = f"smiles: {row['smiles']}\n p_bbb_plus: {row['p_bbb_plus']:.4f}\nlabel: {int(row.get('bbb_label',0))}\n"
                        zf.writestr(name, txt)
                mem.seek(0)
                st.download_button("Download per-molecule summary (ZIP)", data=mem.getvalue(), file_name="bbb_per_mol.zip")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if not run_btn and smiles_list:
        st.info(f"Ready to predict {len(smiles_list)} molecule(s). Click 'Predict BBB' to run.")


