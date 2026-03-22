import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import subprocess
import toml
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Green FL Platform", layout="wide")

for key, default in [
    ("etape", 1),
    ("fl_process", None),
    ("selected_strategy", None),
    ("selected_dataset", None),
    ("selected_rounds", None),
    ("selected_epochs", None),
    ("selected_clients", None),
    ("selected_model_name", None),
    ("known_csv_files_before_run", []),
    ("current_run_csv", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def safe_value(value, unit=""):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.6f}{unit}" if abs(value) < 1 else f"{value:.3f}{unit}"
    return str(value)

def get_all_emission_csvs():
    outputs_dir = PROJECT_DIR / "outputs"
    if not outputs_dir.exists():
        return []
    return sorted(outputs_dir.glob("**/emission.csv"), key=lambda p: p.stat().st_mtime)

def get_latest_csv():
    return (get_all_emission_csvs() or [None])[-1]

def get_new_csv_after_run():
    known = set(str(p) for p in st.session_state.known_csv_files_before_run)
    new = [p for p in get_all_emission_csvs() if str(p) not in known]
    return new[-1] if new else None

def read_csv_safely(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def write_pyproject_with_config(strategy, rounds, epochs, clients, lr):
    path = PROJECT_DIR / "pyproject.toml"
    data = toml.load(path)
    data["tool"]["flwr"]["app"]["config"]["strategy"] = strategy.lower()
    data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = rounds
    data["tool"]["flwr"]["app"]["config"]["local-epochs"] = epochs
    data["tool"]["flwr"]["app"]["config"]["fraction-train"] = 1.0
    data["tool"]["flwr"]["app"]["config"]["learning-rate"] = lr
    data["tool"]["flwr"]["app"]["config"].pop("franction-train", None)
    data["superlink"]["local"]["options"]["num-supernodes"] = clients
    with open(path, "w") as f:
        toml.dump(data, f)


# ════════════════════════════════════════════════════════════════════
# ÉCRAN 1  — uniquement quand etape == 1
# Les widgets interactifs n'existent PAS dans le DOM aux étapes 2 et 3
# ════════════════════════════════════════════════════════════════════
if st.session_state.etape == 1:

    st.title("🌱 Green Federated Learning Platform")
    st.markdown("### 🛠️ Étape 1 : Configuration")
    st.divider()

    col_m, col_d = st.columns(2)
    with col_m:
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #4CAF50;height:160px;">
            <h4 style="margin-top:0;">🧠 Architecture</h4>
            <p style="font-size:0.85em;color:#555;">Importez votre modèle (.py ou .pt)</p>
        </div>""", unsafe_allow_html=True)
        model_file = st.file_uploader("Fichier", type=["py", "pt"], label_visibility="collapsed")

    with col_d:
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #2196F3;height:160px;">
            <h4 style="margin-top:0;">📂 Données</h4>
            <p style="font-size:0.85em;color:#555;">Choisissez le jeu de données cible</p>
        </div>""", unsafe_allow_html=True)
        dataset = st.selectbox("Dataset", ["CIFAR-10", "CheXpert"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🚀 Stratégie & Hyperparamètres")
    c_s, c_h = st.columns([1, 2])
    with c_s:
        strategie = st.selectbox("Choix Stratégie", ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad"])
    with c_h:
        rounds = st.slider("Rounds", 1, 200, 100)
        e_col, c_col, lr_col = st.columns(3)
        with e_col:
            epochs = st.select_slider("Epochs locales", options=[1, 2, 3])
        with c_col:
            clients = st.number_input("Clients", min_value=1, value=10, step=1)
        with lr_col:
            lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, format="%.4f", step=0.001)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Configuration sélectionnée")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Stratégie", strategie)
    r2.metric("Dataset", dataset)
    r3.metric("Rounds", int(rounds))
    r4.metric("Epochs", int(epochs))
    r5.metric("Clients", int(clients))

    model_name = model_file.name if model_file else "Aucun fichier importé"
    st.caption(f"Modèle sélectionné : {model_name}")

    if st.button("🚀 LANCER L'EXPÉRIENCE", use_container_width=True, type="primary"):  #BUG ICI, RESTE AFFICHE PENDANT L'ETAPE 2
        st.session_state.selected_lr = float(lr)
        st.session_state.selected_strategy = strategie
        st.session_state.selected_dataset = dataset
        st.session_state.selected_rounds = int(rounds)
        st.session_state.selected_epochs = int(epochs)
        st.session_state.selected_clients = int(clients)
        st.session_state.selected_model_name = model_name
        st.session_state.known_csv_files_before_run = get_all_emission_csvs()
        st.session_state.current_run_csv = None
        write_pyproject_with_config(strategie, int(rounds), int(epochs), int(clients), lr)
        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        st.session_state.fl_process = subprocess.Popen(["flwr", "run", "."], cwd=PROJECT_DIR, env=env)
        st.session_state.etape = 2
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# ÉCRAN 2
# ════════════════════════════════════════════════════════════════════
elif st.session_state.etape == 2:
    st.title("🔄 Étape 2 : Entraînement en cours")
    st.divider()

    # Résumé config — lecture seule, pas de widgets
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stratégie", st.session_state.get("selected_strategy", "N/A"))
    c2.metric("Dataset", st.session_state.get("selected_dataset", "N/A"))
    c3.metric("Rounds", st.session_state.get("selected_rounds", "N/A"))
    c4.metric("Epochs", st.session_state.get("selected_epochs", "N/A"))
    c5.metric("Clients", st.session_state.get("selected_clients", "N/A"))
    st.caption(f"Modèle : {st.session_state.get('selected_model_name', 'N/A')}")
    st.divider()

    process_running = False
    if st.session_state.fl_process is not None:
        process_running = st.session_state.fl_process.poll() is None

    if st.session_state.current_run_csv is None:
        csv = get_new_csv_after_run()
        if csv:
            st.session_state.current_run_csv = csv

    df = read_csv_safely(st.session_state.current_run_csv)

    if df is not None and not df.empty:
        last = df.iloc[-1]
        st.subheader("Suivi en direct")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CPU power", safe_value(last.get("cpu_power"), " W"))
        m2.metric("GPU power", safe_value(last.get("gpu_power"), " W"))
        m3.metric("CO₂ émis", safe_value(last.get("emissions"), " kg"))
        m4.metric("Énergie", safe_value(last.get("energy_consumed"), " kWh"))
        if len(df) > 1:
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Énergie consommée")
                st.line_chart(df[["energy_consumed"]])
            with g2:
                st.caption("Émissions CO₂")
                st.line_chart(df[["emissions"]])
        with st.expander("Voir les données en cours"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Le fichier emission.csv du run courant n'est pas encore disponible.")

    st.subheader("État du processus")
    st.write("En cours..." if process_running else "Terminé")

    if not process_running and st.session_state.fl_process is not None:
        st.success("Entraînement terminé.")
        st.session_state.etape = 3
        st.rerun()

    col_nav = st.columns([1, 1, 4])
    if col_nav[0].button("⬅️ Retour"):
        st.session_state.etape = 1
        st.rerun()
    if col_nav[1].button("⏹️ Arrêter", type="secondary"):
        if st.session_state.fl_process is not None:
            st.session_state.fl_process.terminate()
        st.session_state.etape = 3
        st.rerun()

    if process_running:
        time.sleep(2)
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# ÉCRAN 3
# ════════════════════════════════════════════════════════════════════
elif st.session_state.etape == 3:

    st.title("📊 Étape 3 : Résultats finaux")
    st.divider()

    csv_path = st.session_state.current_run_csv or get_latest_csv()
    df_res = read_csv_safely(csv_path)

    if df_res is not None and not df_res.empty:
        last = df_res.iloc[-1]

        st.subheader("Configuration demandée")
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Stratégie", st.session_state.get("selected_strategy", "N/A"))
        d2.metric("Dataset", st.session_state.get("selected_dataset", "N/A"))
        d3.metric("Rounds", st.session_state.get("selected_rounds", "N/A"))
        d4.metric("Epochs", st.session_state.get("selected_epochs", "N/A"))
        d5.metric("Clients", st.session_state.get("selected_clients", "N/A"))
        d6.metric("Learning Rate", st.session_state.get("selected_lr", "N/A"))
        st.caption(f"Modèle demandé : {st.session_state.get('selected_model_name', 'N/A')}")
        st.markdown("---")

        st.subheader("Résumé")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CO₂ émis", safe_value(last.get("emissions"), " kg"))
        c2.metric("Énergie consommée", safe_value(last.get("energy_consumed"), " kWh"))
        c3.metric("Durée", safe_value(last.get("duration"), " s"))
        c4.metric("Taux d'émission", safe_value(last.get("emissions_rate"), " kg/s"))
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("CPU power", safe_value(last.get("cpu_power"), " W"))
        c6.metric("GPU power", safe_value(last.get("gpu_power"), " W"))
        c7.metric("RAM power", safe_value(last.get("ram_power"), " W"))
        c8.metric("Eau consommée", safe_value(last.get("water_consumed"), " L"))
        st.markdown("---")

        st.subheader("Cohérence configuration / CSV")
        csv_proj = str(last.get("project_name", "N/A")).lower()
        sel_strat = str(st.session_state.get("selected_strategy", "N/A")).lower()
        z1, z2, z3 = st.columns(3)
        z1.write(f"**Stratégie demandée** : {st.session_state.get('selected_strategy', 'N/A')}")
        z2.write(f"**Dataset demandé** : {st.session_state.get('selected_dataset', 'N/A')}")
        z3.write(f"**Project name CSV** : {last.get('project_name', 'N/A')}")
        if sel_strat not in csv_proj:
            st.warning(f"CSV d'un run précédent ('{last.get('project_name')}') — ne correspond pas à '{st.session_state.get('selected_strategy')}'.")
        else:
            st.success("✅ Le CSV correspond bien à la stratégie choisie.")
        st.markdown("---")

        left, right = st.columns(2)
        with left:
            st.subheader("Expérience")
            st.markdown(f"""
**Projet** : {safe_value(last.get("project_name"))}  
**Run ID** : {safe_value(last.get("run_id"))}  
**Experiment ID** : {safe_value(last.get("experiment_id"))}  
**Mode** : {safe_value(last.get("tracking_mode"))}  
**PUE** : {safe_value(last.get("pue"))}  
**WUE** : {safe_value(last.get("wue"))}""")
        with right:
            st.subheader("Machine")
            st.markdown(f"""
**OS** : {safe_value(last.get("os"))}  
**Python** : {safe_value(last.get("python_version"))}  
**CPU** : {safe_value(last.get("cpu_model"))}  
**CPU count** : {safe_value(last.get("cpu_count"))}  
**GPU** : {safe_value(last.get("gpu_model"))}  
**GPU count** : {safe_value(last.get("gpu_count"))}""")
        st.markdown("---")

        if len(df_res) > 1:
            st.subheader("Évolution")
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Énergie consommée")
                st.line_chart(df_res[["energy_consumed"]])
            with g2:
                st.caption("Émissions CO₂")
                st.line_chart(df_res[["emissions"]])

        with st.expander("Voir les données brutes du CSV"):
            st.dataframe(df_res, use_container_width=True)

        st.download_button("📥 Télécharger CSV", data=df_res.to_csv(index=False),
                           file_name="emission.csv", mime="text/csv")
    else:
        st.warning("Aucun fichier emission.csv exploitable trouvé.")

    if st.button("🔄 Nouvelle expérience"):
        st.session_state.etape = 1
        st.session_state.current_run_csv = None
        st.session_state.fl_process = None
        st.rerun()
