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

# Initialisation du session_state
for key, default in [
    ("etape", 1),
    ("fl_process", None),
    ("selected_strategy", None),
    ("selected_dataset", None),
    ("selected_rounds", 100),
    ("selected_epochs", 1),
    ("selected_clients", 10),
    ("selected_lr", 0.01),
    ("selected_fraction_train", 1.0),
    ("selected_fraction_eval", 1.0),
    ("selected_model_name", None),
    ("selected_alpha", 0.1),
    ("selected_self_balancing", True),
    ("known_csv_files_before_run", []),
    ("current_run_csv", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Fonctions utilitaires ---
def safe_value(value, unit=""):
    if pd.isna(value): return "N/A"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.6f}{unit}" if abs(value) < 1 else f"{value:.3f}{unit}"
    return str(value)

def get_latest_csv():
    """Récupère le chemin du tout dernier fichier emission.csv créé dans le dossier outputs."""
    csvs = get_all_emission_csvs()
    return csvs[-1] if csvs else None

def get_all_emission_csvs():
    outputs_dir = PROJECT_DIR / "outputs"
    if not outputs_dir.exists(): return []
    return sorted(outputs_dir.glob("**/emission.csv"), key=lambda p: p.stat().st_mtime)

def get_new_csv_after_run():
    known = set(str(p) for p in st.session_state.known_csv_files_before_run)
    new = [p for p in get_all_emission_csvs() if str(p) not in known]
    return new[-1] if new else None

def read_csv_safely(path):
    if path is None: return None
    try: return pd.read_csv(path)
    except: return None

def write_pyproject_with_config(strategy, rounds, epochs, lr, f_train, f_eval, num_clients, extra_opts, alpha, self_balancing):
    # --- 1. Mise à jour de pyproject.toml ---
    pyproject_path = PROJECT_DIR / "pyproject.toml"
    pyproject_data = toml.load(pyproject_path)
    cfg = pyproject_data["tool"]["flwr"]["app"]["config"]
    
    cfg["strategy"] = strategy.lower()
    cfg["num-server-rounds"] = rounds
    cfg["local-epochs"] = epochs
    cfg["learning-rate"] = lr
    cfg["num-supernodes"] = num_clients
    cfg["num-supernodes-training"] = max(1, int(num_clients * f_train))
    cfg["num-supernodes-evaluation"] = max(1, int(num_clients * f_eval))
    cfg["fraction-train"] = f_train
    cfg["fraction-evaluate"] = f_eval
    cfg["alpha"] = alpha
    cfg["self-balancing"] = self_balancing

    for key, val in extra_opts.items():
        cfg[key] = val
        
    with open(pyproject_path, "w") as f:
        toml.dump(pyproject_data, f)

    # --- 2. Mise à jour de C:\Users\darks\.flwr\config.toml ---
    flwr_global_config = Path.home() / ".flwr" / "config.toml"
    
    if flwr_global_config.exists():
        try:
            data_global = toml.load(flwr_global_config)
            
            # On accède à la structure que vous avez vue
            # superlink -> local -> options -> num_supernodes
            if "superlink" in data_global and "local" in data_global["superlink"]:
                local_section = data_global["superlink"]["local"]
                
                # On met à jour le nombre de clients simulés
                if "options" not in local_section:
                    local_section["options"] = {}
                local_section["options"]["num-supernodes"] = num_clients
                
                # SÉCURITÉ : On retire le run_id s'il existe pour éviter le conflit SuperLink
                # .pop(key, None) ne génère pas d'erreur si la clé est absente
                local_section.pop("run_id", None)
                
                with open(flwr_global_config, "w") as f:
                    toml.dump(data_global, f)
                    
        except Exception as e:
            # Si le fichier est verrouillé ou corrompu, on l'efface. 
            # Flower le recréera proprement au lancement.
            st.warning(f"Note : Reset du cache Flower suite à une erreur de lecture.")
            flwr_global_config.unlink()
# ════════════════════════════════════════════════════════════════════
# ÉCRAN 1 : CONFIGURATION
# ════════════════════════════════════════════════════════════════════
if st.session_state.etape == 1:
    with st.container():
        st.title("🌱 Green Federated Learning Platform")
        st.markdown("### 🛠️ Étape 1 : Configuration ")
        st.divider()

        col_m, col_d = st.columns(2)
        with col_m:
            st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #4CAF50;height:160px;"><h4>🧠 Architecture</h4><p>Modèle (.py ou .pt)</p></div>', unsafe_allow_html=True)
            model_file = st.file_uploader("Fichier", type=["py", "pt"], label_visibility="collapsed")

        with col_d:
            st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #2196F3;height:160px;"><h4>📂 Données</h4><p>Jeu de données cible</p></div>', unsafe_allow_html=True)
            dataset = st.selectbox("Dataset", ["CIFAR-10", "CheXpert"], label_visibility="collapsed")

        st.markdown("#### 🚀 Hyperparamètres de base")
        c_s, c_r, c_e, c_l = st.columns(4)
        with c_s:
            strategie = st.selectbox("Stratégie", ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad"])
        with c_r:
            rounds = st.number_input("Rounds", min_value=1, value=1)
        with c_e:
            epochs = st.selectbox("Epochs locales", [1, 2, 3])
        with c_l:
            lr = st.number_input("Learning Rate", min_value=0.0001, value=0.01, format="%.4f")

        st.markdown("#### ⚖️ Sélection des clients")
        clients_number = st.slider("Nombre de clients", 2, 100, st.session_state.selected_clients, help="Nombre de clients")
        default_train = min(8, clients_number)
        default_eval = min(5, clients_number)
        f_train_col, f_eval_col = st.columns(2)
        with f_train_col:
            clients_train = st.slider("Entraînement", 1, max(clients_number, 1), default_train, help="Clients tirés pour l'entraînement")
            frac_train = clients_train / clients_number if clients_number > 0 else 0
        with f_eval_col:
            clients_eval = st.slider("Évaluation", 1, max(clients_number, 1), default_eval, help="Part des clients tirés pour la validation")
            frac_eval = clients_eval / clients_number if clients_number > 0 else 0
        extra_opts = {}
        if strategie in ["FedProx", "FedAdam", "FedYogi", "FedAdagrad"]:
            st.markdown("#### ⚙️ Options spécifiques")
            with st.expander(f"Options spécifiques à {strategie}"):
                if strategie == "FedProx":
                    extra_opts["proximal-mu"] = st.slider("Proximal Mu (μ)", 0.0, 1.0, 0.1)
                else:
                    c1, c2 = st.columns(2)
                    extra_opts["server-learning-rate"] = c1.number_input("Server LR", value=1.0)
                    extra_opts["tau"] = c2.number_input("Tau", value=1e-9, format="%.1e")
                    if strategie != "FedAdagrad":
                        extra_opts["beta-1"] = st.slider("Beta 1", 0.0, 1.0, 0.9)
                        extra_opts["beta-2"] = st.slider("Beta 2", 0.0, 1.0, 0.99)
        st.markdown("#### Options de partitionnement (Dirichlet)")
        alpha = st.slider("Alpha", 0.0, 1.0, 0.1, help="Plus alpha est petit, plus les données sont hétérogènes entre les clients. Avec alpha proche de 0, chaque client aura principalement des exemples d'une seule classe. Avec alpha élevé (ex: 1.0), les données seront plus équilibrées entre les clients. Ceci permet de simuler un label skew.")
        self_balancing = st.checkbox("Équilibrage automatique", value=True, help="Activez cette option pour que chaque client ait une quantité de données similaire, même avec un alpha faible. Le partitionnement restera hétérogène en termes de classes, mais la taille des partitions sera plus équilibrée.")
        st.markdown("#### Options wandb (facultatif)")
        wandb_mode = "disabled" # Valeurs par défaut pour éviter les erreurs si l'utilisateur ne remplit pas ces champs
        wandb_project = ""
        wandb_api_key = ""
        wandb_entity = ""
        with st.expander("Options de tracking (wandb)"):
            wandb_mode = st.selectbox("Mode de synchronisation", ["disabled", "online"])
            if wandb_mode == "online":
                c1, c2 = st.columns(2)
                wandb_api_key = c1.text_input("Clé API W&B", type="password", help="Trouvez votre clé sur wandb.ai/authorize")
                wandb_project = c2.text_input("Nom du Projet", value="FLOWER-advanced-pytorch")
                
                wandb_entity = st.text_input("Entité (Équipe ou Utilisateur)", help="Laissez vide pour votre compte personnel")
                

        if st.button("🚀 LANCER L'EXPÉRIENCE", width="stretch", type="primary"):
            st.session_state.selected_strategy = strategie
            st.session_state.selected_clients = clients_number
            st.session_state.selected_dataset = dataset
            st.session_state.selected_rounds = rounds
            st.session_state.selected_epochs = epochs
            st.session_state.selected_lr = lr
            st.session_state.selected_fraction_train = frac_train
            st.session_state.selected_fraction_eval = frac_eval
            st.session_state.selected_alpha = alpha
            st.session_state.selected_self_balancing = self_balancing
            st.session_state.selected_model_name = model_file.name if model_file else "Défaut"
            st.session_state.known_csv_files_before_run = get_all_emission_csvs()
            
            write_pyproject_with_config(strategie, rounds, epochs, lr, frac_train, frac_eval, clients_number, extra_opts, alpha, self_balancing)
            
            env = os.environ.copy()
            if wandb_project:
                env["WANDB_PROJECT"] = wandb_project
            env["WANDB_MODE"] = wandb_mode
            
            if wandb_api_key:
                env["WANDB_API_KEY"] = wandb_api_key
            if wandb_entity:
                env["WANDB_ENTITY"] = wandb_entity

            # Lancement du processus avec le nouvel environnement
            st.session_state.fl_process = subprocess.Popen(
                ["flwr", "run", "."], 
                cwd=PROJECT_DIR, 
                env=env
            )
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
            st.dataframe(df, width="stretch")
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
            st.dataframe(df_res, width="stretch")

        st.download_button("📥 Télécharger CSV", data=df_res.to_csv(index=False),
                           file_name="emission.csv", mime="text/csv")
    else:
        st.warning("Aucun fichier emission.csv exploitable trouvé.")

    if st.button("🔄 Nouvelle expérience"):
        st.session_state.etape = 1
        st.session_state.current_run_csv = None
        st.session_state.fl_process = None
        st.rerun()
