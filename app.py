import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# On importe ta classe depuis ton fichier main.py
from main import HybridAIDetector

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="VerifAI Detector",
    page_icon="🛡️",
    layout="wide"
)

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_and_train_model():
    """
    Cette fonction ne s'exécute qu'une seule fois au démarrage.
    Le décorateur @st.cache_resource garde le modèle en mémoire.
    """
    detector = HybridAIDetector()
    
    # Chemins (assure-toi qu'ils existent)
    dossier_real = "dataset/real"
    dossier_ai = "dataset/ai"
    
    if not os.path.exists(dossier_real) or not os.path.exists(dossier_ai):
        return None, "Datasets manquants. Lance 'python get_images.py' d'abord."
    
    with st.spinner('🧠 Entraînement du modèle en cours (Sobel + FFT)...'):
        detector.train(dossier_real, dossier_ai)
    
    return detector, None

def save_uploaded_file(uploaded_file):
    """Sauvegarde temporaire de l'image uploadée pour OpenCV"""
    try:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return "temp_image.jpg"
    except Exception as e:
        return None

# --- INTERFACE PRINCIPALE ---

st.title("🛡️ VerifAI : Détecteur d'images IA")
st.markdown("""
Cette application analyse les **fréquences invisibles (FFT)** et les **gradients de pixels** pour déterminer si une image est une photo réelle ou générée par une IA.
""")

# 1. Chargement du modèle
detector, error_msg = load_and_train_model()

if error_msg:
    st.error(error_msg)
    st.stop()

# Sidebar avec infos
with st.sidebar:
    st.header("État du système")
    st.success("Modèle entraîné et prêt ✅")
    st.info(f"Points Réels appris : {len([l for l in detector.train_labels if l==0])}")
    st.info(f"Points IA appris : {len([l for l in detector.train_labels if l==1])}")
    st.markdown("---")
    st.write("Coded with ❤️ by You")

# 2. Zone d'upload
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Uploadez une image")
    uploaded_file = st.file_uploader("Glissez votre image ici (JPG, PNG)", type=['jpg', 'jpeg', 'png', 'webp'])

    if uploaded_file is not None:
        # Affichage de l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        # Sauvegarde temporaire pour le backend
        temp_path = save_uploaded_file(uploaded_file)

# 3. Résultats
with col2:
    st.subheader("2. Analyse")
    
    if uploaded_file is not None and temp_path:
        # Prédiction
        verdict, d_real, d_ai = detector.predict(temp_path, show_plot=False)
        
        # --- CALCUL DU POURCENTAGE ---
        # Plus la distance IA est petite par rapport au total, plus c'est de l'IA
        total_dist = d_real + d_ai
        if total_dist == 0: total_dist = 1 # Éviter division par zéro
        
        # Score de "IA-itude" (0 à 1)
        score_ia = d_real / total_dist
        pourcentage = score_ia * 100
        
        # --- AFFICHAGE JOLI ---
        st.write("### Résultat :")
        
        if verdict == "IA":
            st.error(f"🤖 C'est probablement une IA ({pourcentage:.1f}%)")
            st.progress(score_ia)
        else:
            st.success(f"📸 C'est probablement une PHOTO ({100 - pourcentage:.1f}%)")
            st.progress(score_ia) # La barre se remplit quand même pour montrer le niveau
            
        st.write(f"**Détails techniques :**")
        st.code(f"Distance au cluster Réel : {d_real:.2f}\nDistance au cluster IA   : {d_ai:.2f}")

        # --- GRAPHIQUE PCA INTERACTIF ---
        st.write("### Visualisation Vectorielle")
        
        # On récupère les points d'entraînement pour refaire le plot dans Streamlit
        real_points = detector.train_data_pca[np.array(detector.train_labels) == 0]
        ai_points = detector.train_data_pca[np.array(detector.train_labels) == 1]
        
        # On recalcule le point de l'image test (on refait un bout de la logique predict pour l'avoir ici)
        feat = detector._get_features(temp_path)
        feat_scaled = detector.scaler.transform([feat])
        point = detector.pca.transform(feat_scaled)[0]

        fig, ax = plt.subplots()
        ax.scatter(real_points[:, 0], real_points[:, 1], c='blue', alpha=0.5, label='Photos Réelles')
        ax.scatter(ai_points[:, 0], ai_points[:, 1], c='red', alpha=0.5, label='Images IA')
        ax.scatter(point[0], point[1], c='lime', s=200, marker='*', edgecolors='black', label='TON IMAGE')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Projection PCA (2D)")
        
        st.pyplot(fig)
        
        # Nettoyage
        os.remove(temp_path)

    else:
        st.info("En attente d'une image...")