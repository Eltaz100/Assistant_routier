
from pathlib import Path
import time

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="RoadSign Assistant",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# ---------------- STYLE CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1120;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #111a2e;
    border-right: 1px solid #26334d;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.5px;
}
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #172747, #102038);
    border: 1px solid #2c4163;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
}
.hero-subtitle {
    color: #a9bbd5;
    margin-top: .4rem;
    font-size: 1rem;
}
.section-card {
    background: #111a2e;
    border: 1px solid #263752;
    padding: 1.1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
}
.status {
    padding: .7rem 1rem;
    background: #102c2a;
    border: 1px solid #21665b;
    border-radius: 10px;
    color: #7ee7c3;
    margin-bottom: 1rem;
}
div[data-testid="stMetric"] {
    background: #111a2e;
    border: 1px solid #263752;
    padding: 1rem;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #a9bbd5;
}
div[data-testid="stMetricValue"] {
    color: #60a5fa;
}
.stButton > button {
    background: #2878f0;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 600;
    min-height: 2.7rem;
}
.stButton > button:hover {
    background: #1764d8;
    color: white;
    border: 0;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #3d5c85;
    border-radius: 12px;
    padding: .6rem;
    background: #101b30;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #263752;
    border-radius: 10px;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🚦 RoadSign <span style="color:#60a5fa">Assistant</span></div>
    <div class="hero-subtitle">
        Détection intelligente des panneaux routiers grâce à la vision par ordinateur.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Paramètres de détection")

    uploaded_file = st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png"],
        help="Choisis une image contenant un ou plusieurs panneaux routiers.",
    )

    confidence = st.slider(
        "Seuil de confiance",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        format="%.2f",
        help="Une valeur plus élevée filtre les détections moins sûres.",
    )

    st.divider()
    st.markdown("### 🧠 Modèle IA")
    st.write("**Architecture :** YOLO")
    st.write("**Poids :** `best.pt`")
    st.write("**Tâche :** détection d'objets")

    st.divider()
    st.markdown("### 💡 Conseil")
    st.caption(
        "Utilise une image nette et bien éclairée. "
        "Si aucun panneau n'est détecté, essaie de diminuer le seuil."
    )

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

if not MODEL_PATH.is_file():
    st.error(f"Modèle introuvable : `{MODEL_PATH}`")
    st.info("Place ton fichier entraîné dans le dossier `models/best.pt`.")
    st.stop()

try:
    with st.spinner("Chargement du modèle IA..."):
        model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Le modèle n'a pas pu être chargé : {exc}")
    st.stop()

st.markdown(
    '<div class="status">● &nbsp; Modèle chargé — prêt à analyser une image</div>',
    unsafe_allow_html=True,
)

# ---------------- RESULTS ----------------
if uploaded_file is None:
    st.markdown("""
    <div class="section-card">
        <h3>👋 Bienvenue !</h3>
        <p style="color:#a9bbd5">
        Importe une photo depuis le panneau de gauche pour commencer.
        L'image originale et le résultat annoté apparaîtront ici.
        </p>
        <p style="color:#8297b5">
        Formats acceptés : JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Fonctionnalités")
    a, b, c = st.columns(3)
    a.info("📍 **Détection**\n\nLocalisation des panneaux dans l'image.")
    b.info("🏷️ **Classification**\n\nIdentification de la classe du panneau.")
    c.info("📊 **Confiance**\n\nScore associé à chaque détection.")

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyse de l'image en cours..."):
            start_time = time.perf_counter()
            result = model.predict(
                source=image,
                conf=confidence,
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - start_time

        detections = result.boxes
        annotated = result.plot()[..., ::-1].copy()

        # Préparer les résultats tabulaires
        rows = []
        if detections is not None:
            for box in detections:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [
                    round(float(v), 1)
                    for v in box.xyxy[0].tolist()
                ]
                rows.append({
                    "Classe": result.names[class_id],
                    "Confiance": f"{score:.1%}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

        st.markdown("### 📊 Vue d'ensemble")
        m1, m2, m3 = st.columns(3)
        m1.metric("Panneaux détectés", len(rows))
        m2.metric(
            "Confiance moyenne",
            f"{sum(float(r['Confiance'].strip('%')) / 100 for r in rows) / len(rows):.1%}"
            if rows else "—",
        )
        m3.metric("Temps d'analyse", f"{elapsed:.2f} s")

        st.markdown("### 🖼️ Analyse visuelle")
        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown("#### Image originale")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("#### Résultat IA")
            st.image(annotated, use_container_width=True)

        st.markdown("### 📋 Détail des détections")

        if rows:
            st.success(f"{len(rows)} panneau(x) détecté(s).")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_data = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Exporter les résultats en CSV",
                data=csv_data,
                file_name="roadsign_results.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "Aucun panneau détecté avec ce seuil. "
                "Essaie une autre image ou diminue le seuil de confiance."
            )

    except Exception as exc:
        st.error("Une erreur est survenue pendant l'analyse.")
        st.exception(exc)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:#8297b5;font-size:.85rem'>"
    "RoadSign Assistant · Computer Vision · Python · YOLO · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
