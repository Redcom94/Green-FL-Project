import streamlit as st
import pandas as pd
import subprocess
import sys
import numpy as np
import time

# Configuration
st.set_page_config(page_title="Green FL Platform", layout="wide")

# Initialisation de la navigation (State)
if 'etape' not in st.session_state:
    st.session_state.etape = 1

# --- ÉCRAN 1 : CONFIGURATION ---
if st.session_state.etape == 1:
    st.title("🌱 Green Federated Learning Platform")
    st.markdown("### 🛠️ Étape 1 : Configuration")
    st.divider()

    # Cartes visuelles pour Modèle et Dataset (Séparées)
    col_m, col_d = st.columns(2)

    with col_m:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 15px; border-left: 5px solid #4CAF50; height: 160px;">
            <h4 style="margin-top:0;">🧠 Architecture</h4>
            <p style="font-size: 0.85em; color: #555;">Importez votre modèle (.py ou .pt)</p>
        </div>
        """, unsafe_allow_html=True) # [cite: 3]
        model_file = st.file_uploader("Fichier", type=["py", "pt"], label_visibility="collapsed") # [cite: 4]

    with col_d:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 15px; border-left: 5px solid #2196F3; height: 160px;">
            <h4 style="margin-top:0;">📂 Données</h4>
            <p style="font-size: 0.85em; color: #555;">Choisissez le jeu de données cible</p>
        </div>
        """, unsafe_allow_html=True) # [cite: 5]
        dataset = st.selectbox("Dataset", ["CIFAR-10", "CheXpert"], label_visibility="collapsed") # [cite: 6]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Choix Stratégie (Déroulant) et Hyperparamètres
    st.markdown("#### 🚀 Stratégie & Hyperparamètres")
    c_s, c_h = st.columns([1, 2])
    
    with c_s:
        strategie = st.selectbox("Choix Stratégie", ["FedAvg", "FedProx", "FedAdam", "FedYogi", "SCAFFOLD"]) # [cite: 8]
    with c_h:
        rounds = st.slider("Rounds", 1, 200, 100) # [cite: 11]
        e_col, lr_col = st.columns(2)
        with e_col:
            epochs = st.select_slider("Epochs locales", options=[1, 2, 3]) # [cite: 12]
        with lr_col:
            lr = st.number_input("Learning Rate", value=0.01, format="%.4f", step=0.001) # [cite: 12]

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 LANCER L'EXPÉRIENCE", use_container_width=True, type="primary"):
        if model_file is not None:
            # 1. Sauvegarde du modèle utilisateur
            with open("pytorchexample/user_model.py", "wb") as fin:
                fin.write(model_file.getbuffer())

            # 2. Préparation d'une SEULE chaîne de config bien formatée
            # IMPORTANT : les textes DOIVENT avoir des doubles guillemets (")
            # J'ajoute une valeur par défaut pour learning-rate si non définie
            cmd = [
            "flwr", "run", ".", 
            "--run-config", f"strategy='{strategie.lower()}'",
            "--run-config", f"num-server-rounds={rounds}",
            "--run-config", f"learning-rate={lr}"
            ]
            # 4. Un SEUL lancement en arrière-plan
            subprocess.Popen(cmd)
            
            # 5. Changement d'état et redirection
            st.session_state.etape = 2
            st.rerun()
        else:
            st.error("⚠️ Veuillez importer un modèle (.py) avant de lancer.")

# --- ÉCRAN 2 : ENTRAÎNEMENT ---
elif st.session_state.etape == 2:
    st.title("🔄 Étape 2 : Entraînement en cours") # [cite: 15]
    st.divider()

    # Indicateurs énergétiques [cite: 23]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CPU power", "32 W") # [cite: 24]
    m2.metric("GPU power", "58 W") # [cite: 25]
    m3.metric("CO2 émis", "12.4 g") # [cite: 26]
    m4.metric("Énergie", "34.2 kJ") # [cite: 27]

    st.progress(45/100) # [cite: 16]
    st.write("**Progression : 45 / 100 Rounds**") # [cite: 18]

    st.subheader("Logs en direct") # [cite: 17]
    st.code("Client 3: Epoch 1/1 - loss = 1.22\nAggregating weights...") # [cite: 19, 21]

    col_nav = st.columns([1, 1, 4])
    if col_nav[0].button("⬅️ Retour"):
        st.session_state.etape = 1
        st.rerun()
    if col_nav[1].button("⏹️ Arrêter", type="secondary"): # [cite: 29]
        st.session_state.etape = 3
        st.rerun()

# --- ÉCRAN 3 : RÉSULTATS ---
elif st.session_state.etape == 3:
    st.title("📊 Étape 3 : Résultats finaux") # 
    st.divider()

    # Graphiques comparatifs [cite: 32, 35, 37]
    st.subheader("Performance & Écologie")
    chart_data = pd.DataFrame({'Round': np.arange(1, 11), 'Accuracy': [0.1, 0.4, 0.7, 0.75, 0.75, 0.76, 0.76, 0.77, 0.78, 0.79]})
    st.line_chart(chart_data.set_index('Round'))

    # Tableau comparatif [cite: 39]
    df_res = pd.DataFrame([
        {"Stratégie": "FedAvg", "Acc": 0.72, "CO2": "12.5 g", "Énergie": "30 kJ"}, # [cite: 44]
        {"Stratégie": "FedProx", "Acc": 0.75, "CO2": "18.1 g", "Énergie": "38 kJ"} # [cite: 45]
    ])
    st.table(df_res) # [cite: 40, 41]

    # Exportation [cite: 46, 47, 48]
    if st.download_button("📥 Télécharger CSV", data=df_res.to_csv(), file_name="results.csv"):
        st.success("Fichier prêt !")
    
    if st.button("🔄 Nouvelle expérience"):
        st.session_state.etape = 1
        st.rerun()