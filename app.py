import streamlit as st
import torch
import pandas as pd
import numpy as np
import time
import os
import subprocess
import toml
import psutil
import json
import random
from pathlib import Path
import wandb
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
# Détecter si CUDA est disponible
GPU_AVAILABLE = torch.cuda.is_available()
ENERGY_PER_GB = 0.06
def read_accuracy_value(output_path):
    """Lit la dernière précision sauvegardée par le serveur."""
    acc_file = output_path / "last_accuracy.txt"
    if acc_file.exists():
        try:
            with open(acc_file, "r") as f:
                content = f.read().strip()
                return float(content)
        except:
            return None
    return None
def list_outputs_files(directory):
    """Liste tous les fichiers dans le dossier outputs avec leurs métadonnées."""
    file_list = []
    if not directory.exists():
        return file_list
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = Path(root) / file
            # On ignore les fichiers cachés ou temporaires
            if not file.startswith('.'):
                file_list.append({
                    "display_name": f"{path.parent.name}/{file}",
                    "full_path": path,
                    "mtime": path.stat().st_mtime
                })
    # Trier du plus récent au plus ancien
    return sorted(file_list, key=lambda x: x['mtime'], reverse=True)
def generate_pdf_report(df_res, session_state, save_path = None):
    """Génère un rapport PDF lisible à partir des données CSV."""
    buffer = io.BytesIO()
    # Si save_path est fourni, on utilise le chemin, sinon le buffer mémoire
    target = save_path if save_path else buffer
    doc = SimpleDocTemplate(
        target,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    GREEN = colors.HexColor("#2E7D32")
    LIGHT_GREEN = colors.HexColor("#E8F5E9")
    DARK_GRAY = colors.HexColor("#212121")
    MID_GRAY = colors.HexColor("#616161")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")
    BORDER = colors.HexColor("#C8E6C9")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("title", parent=styles["Normal"],
        fontSize=22, textColor=GREEN, fontName="Helvetica-Bold",
        spaceAfter=2*mm, leading=26)
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
        fontSize=10, textColor=MID_GRAY, fontName="Helvetica",
        spaceAfter=6*mm)
    section_style = ParagraphStyle("section", parent=styles["Normal"],
        fontSize=13, textColor=GREEN, fontName="Helvetica-Bold",
        spaceBefore=5*mm, spaceAfter=3*mm, borderPad=2,
        leading=16)
    label_style = ParagraphStyle("label", parent=styles["Normal"],
        fontSize=8, textColor=MID_GRAY, fontName="Helvetica", leading=10)
    value_style = ParagraphStyle("value", parent=styles["Normal"],
        fontSize=12, textColor=DARK_GRAY, fontName="Helvetica-Bold", leading=14)
    note_style = ParagraphStyle("note", parent=styles["Normal"],
        fontSize=8, textColor=MID_GRAY, fontName="Helvetica-Oblique",
        spaceAfter=4*mm)

    story = []

    # ── Header ──
    story.append(Paragraph("📊 Rapport Green FL", title_style))
    from datetime import datetime
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=5*mm))

    def metric_table(rows_data):
        """rows_data: list of (label, value) tuples, displayed in a 4-col grid."""
        col_w = (A4[0] - 40*mm) / 4
        cells = []
        row_labels = []
        row_values = []
        for i, (lbl, val) in enumerate(rows_data):
            row_labels.append(Paragraph(lbl, label_style))
            row_values.append(Paragraph(str(val), value_style))
            if (i + 1) % 4 == 0 or i == len(rows_data) - 1:
                # Pad to 4 cols
                while len(row_labels) < 4:
                    row_labels.append(Paragraph("", label_style))
                    row_values.append(Paragraph("", value_style))
                cells.append(row_labels)
                cells.append(row_values)
                row_labels = []
                row_values = []

        t = Table(cells, colWidths=[col_w]*4, hAlign="LEFT")
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
            ("ROWBACKGROUND", (0, 0), (-1, -1), [LIGHT_GRAY, LIGHT_GREEN]),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        # Alternate bg per pair of rows
        for r in range(0, len(cells), 2):
            bg = LIGHT_GREEN if (r // 2) % 2 == 0 else LIGHT_GRAY
            style_cmds.append(("BACKGROUND", (0, r), (-1, r+1), bg))
        t.setStyle(TableStyle(style_cmds))
        return t

    last = df_res.iloc[-1]

    # ── Section 1 : Configuration ──
    story.append(Paragraph("⚙️ Configuration demandée", section_style))
    config_data = [
        ("Stratégie", session_state.get("selected_strategy", "N/A")),
        ("Dataset", session_state.get("selected_dataset", "N/A")),
        ("Rounds", session_state.get("selected_rounds", "N/A")),
        ("Epochs", session_state.get("selected_epochs", "N/A")),
        ("Clients", session_state.get("selected_clients", "N/A")),
        ("Learning Rate", session_state.get("selected_lr", "N/A")),
        ("Modèle", session_state.get("selected_model_name", "Défaut")),
    ]
    story.append(metric_table(config_data))
    story.append(Spacer(1, 5*mm))
# --- Section 2 : Coût Réseau ---
    path_comp = get_latest_csv("em_comparison.json")
    if path_comp and path_comp.exists():
        with open(path_comp, "r") as f:
            comp_data = json.load(f) 
            
        story.append(Paragraph("🌐 Analyse du Coût Réseau", section_style))
        story.append(Paragraph(f"Estimation basée sur l'intensité du transport de données ({ENERGY_PER_GB} kWh/GB).", note_style))
        network_metrics = [
            ("Volume (GB)", f"{comp_data.get('network_transfer_gb', 0):.4f} GB"),
            ("Énergie Réseau", f"{comp_data.get('network_energy_kwh', 0):.6f} kWh"),
            ("Total Combiné", f"{comp_data.get('total_combined_energy_kwh', 0):.6f} kWh"),
            ("Source", "Shift Project / IEA"),
        ]
    
        story.append(metric_table(network_metrics))
        story.append(Spacer(1, 5*mm))

    # ── Section 3 : Résumé Environnemental ──
    story.append(Paragraph("🌿 Résumé Environnemental", section_style))
    env_data = [
        ("CO<sub rise='2' size='7'>2</sub> émis (kg)", safe_value(last.get("emissions"), "")),
        ("Énergie consommée (kWh)", safe_value(last.get("energy_consumed"), "")),
        ("Durée (s)", safe_value(last.get("duration"), "")),
        ("Taux d'émission (kg/s)", safe_value(last.get("emissions_rate"), "")),
        ("CPU power (W)", safe_value(last.get("cpu_power"), "")),
        ("GPU power (W)", safe_value(last.get("gpu_power"), "")),
        ("RAM power (W)", safe_value(last.get("ram_power"), "")),
        ("Eau consommée (L)", safe_value(last.get("water_consumed"), "")),
    ]
    story.append(metric_table(env_data))
    story.append(Spacer(1, 5*mm))

    # ── Section 4 : Expérience & Machine ──
    story.append(Paragraph("🧪 Expérience & Machine", section_style))
    exp_data = [
        ("Projet", safe_value(last.get("project_name"))),
        ("Run ID", safe_value(last.get("run_id"))),
        ("Experiment ID", safe_value(last.get("experiment_id"))),
        ("Mode", safe_value(last.get("tracking_mode"))),
        ("PUE", safe_value(last.get("pue"))),
        ("WUE", safe_value(last.get("wue"))),
        ("OS", safe_value(last.get("os"))),
        ("Python", safe_value(last.get("python_version"))),
        ("CPU", safe_value(last.get("cpu_model"))),
        ("CPU count", safe_value(last.get("cpu_count"))),
        ("GPU", safe_value(last.get("gpu_model"))),
        ("GPU count", safe_value(last.get("gpu_count"))),
    ]
    story.append(metric_table(exp_data))
    story.append(Spacer(1, 5*mm))

    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Paragraph("Green FL Platform — Rapport généré automatiquement", note_style))
    doc.build(story)
    
    if save_path:
        return None # Le fichier est déjà écrit sur le disque
    else:
        buffer.seek(0)
        return buffer.read()
    return buffer.read()

OPTIM_TIPS = {
    "FedAvg": " **Optimisation Green** : Idéal pour limiter la communication. Activez l'Early Stopping local pour économiser du CPU sur les clients qui convergent vite.",
    "FedProx": " **Optimisation Green** : Ajustez le **Mu**. Un Mu faible réduit la charge de calcul locale mais peut ralentir la convergence globale.",
    "FedAdam": " **Optimisation Green** : Très efficace mais gourmand en RAM. Réduisez la **Fraction Entraînement** pour économiser l'énergie globale du réseau.",
    "FedYogi": " **Optimisation Green** : Similaire à Adam, mais plus stable. Utilisez un **Learning Rate** plus élevé pour atteindre la précision cible en moins de rounds.",
    "FedAdagrad": " **Optimisation Green** : Moins de paramètres à synchroniser. Excellent pour les connexions instables afin d'éviter les ré-émissions de paquets réseau."
}
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
    ("selected_blur_config", {}),
    ("known_csv_files_before_run", []),
    ("current_run_csv", None),
    ("known_csv_files_before_run_2",[]),
    ("known_csv_files_before_run_3",[]),
    ("selected_small_clients", 0),
    ("selected_medium_clients", 0),
    ("selected_big_clients", 0),
    ("use_gpu", True)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Fonctions utilitaires ---
@st.cache_resource
def get_global_state():
    # Cet objet sera partagé par TOUS les utilisateurs et TOUS les rafraîchissements
    return {"running": False, "process": None, "config": {}}

def safe_value(value, unit="", precision=4):
    try:
        # On force la conversion en float au cas où c'est du texte
        num_value = float(str(value).replace(',', '.'))
        
        if pd.isna(num_value): 
            return "N/A"
            
        # Formatage strict avec la précision choisie
        formatted = f"{num_value:.{precision}f}".replace('.', ',')
        return f"{formatted}{unit}"
    except (ValueError, TypeError):
        # Si vraiment ce n'est pas un chiffre (ex: "N/A"), on renvoie tel quel
        return str(value)

def get_latest_csv(data):
    """Récupère le chemin du tout dernier fichier emission.csv créé dans le dossier outputs."""
    csvs = get_all_emission_csvs(data)
    return csvs[-1] if csvs else None

def get_all_emission_csvs(data):
    outputs_dir = PROJECT_DIR / "outputs"
    if not outputs_dir.exists(): return []
    return sorted(outputs_dir.glob(f"**/{data}"), key=lambda p: p.stat().st_mtime)

def get_new_csv_after_run(data):
    known = set(str(p) for p in st.session_state.known_csv_files_before_run)
    known.update(str(p) for p in st.session_state.known_csv_files_before_run_2)
    known.update(str(p) for p in st.session_state.known_csv_files_before_run_3)
    new = [p for p in get_all_emission_csvs(data) if str(p) not in known]
    return new[-1] if new else None

def read_csv_safely(path):
    if path is None: return None
    try: return pd.read_csv(path, sep=';')
    except: return None

def write_pyproject_with_config(batch_size,num_gpus,strategy, rounds, epochs, lr, f_train, f_eval, num_clients, extra_opts, alpha, self_balancing, small_c, medium_c, big_c, dataset_name, img_size, num_channels, num_classes, blur_config=None):
    # --- 1. Mise à jour de pyproject.toml ---
    pyproject_path = PROJECT_DIR / "pyproject.toml"
    pyproject_data = toml.load(pyproject_path)
    cfg = pyproject_data["tool"]["flwr"]["app"]["config"]
    cfg["img-size"] = img_size
    cfg["num-channels"] = num_channels
    cfg["num-classes"] = num_classes
    cfg["strategy"] = strategy.lower()
    cfg["num-server-rounds"] = rounds
    cfg["dataset-name"] = dataset_name
    cfg["local-epochs"] = epochs
    cfg["learning-rate"] = lr
    cfg["num-supernodes"] = num_clients
    cfg["num-supernodes-training"] = max(1, int(num_clients * f_train))
    cfg["num-supernodes-evaluation"] = max(1, int(num_clients * f_eval))
    cfg["fraction-train"] = f_train
    cfg["fraction-evaluate"] = f_eval
    cfg["alpha"] = alpha
    cfg["self-balancing"] = self_balancing
    cfg["small-clients"] = small_c
    cfg["medium-clients"] = medium_c
    cfg["big-clients"] = big_c
    cfg["num-gpus"] = num_gpus
    cfg["batch-size"] = batch_size
    
    # blur_config : dict {client_id -> blur_percent}, sérialisé en JSON string pour pyproject.toml
    cfg["blur-config"] = json.dumps({str(k): v for k, v in (blur_config or {}).items()})

    for key, val in extra_opts.items():
        cfg[key] = val
        
    with open(pyproject_path, "w") as f:
        toml.dump(pyproject_data, f)

    # --- 2. Mise à jour de \.flwr\config.toml ---
    flwr_global_config = Path.home() / ".flwr" / "config.toml"
    
    if flwr_global_config.exists():
        try:
            data_global = toml.load(flwr_global_config)
            
            # On accède à la structure
            # superlink -> local -> options -> num_supernodes
            if "superlink" in data_global and "local" in data_global["superlink"]:
                local_section = data_global["superlink"]["local"]
                
                # On met à jour le nombre de clients simulés
                if "options" not in local_section:
                    local_section["options"] = {}
                local_section["options"]["num-supernodes"] = num_clients
                local_section2 = data_global["superlink"]["local"]["options"]["backend"]
                if "client-resources" not in local_section2:
                    local_section2["client-resources"] = {}
                local_section2["client-resources"]["num-gpus"] = num_gpus            
                with open(flwr_global_config, "w") as f:
                    toml.dump(data_global, f)
                    
        except Exception as e:
            # Si le fichier est verrouillé ou corrompu, on l'efface. 
            # Flower le recréera proprement au lancement.
            st.warning(f"Note : Reset du cache Flower suite à une erreur de lecture.")
            flwr_global_config.unlink()
# --- Logique de récupération après rafraîchissement ---
global_status = get_global_state()
# --- Barre latérale : Explorateur de fichiers ---
with st.sidebar:
    st.header("📂 Historique des Runs")
    st.write("Retrouvez vos anciens rapports et données.")

    outputs_dir = PROJECT_DIR / "outputs"

    if outputs_dir.exists():

        # ==============================
        # 1. DOSSIERS PRINCIPAUX
        # ==============================
        first_level_dirs = sorted(
            [d for d in outputs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if first_level_dirs:

            selected_main_dir = st.selectbox(
                "📁 Choisir une date",
                first_level_dirs,
                format_func=lambda x: x.name
            )

            # ==============================
            # 2. SOUS-DOSSIERS (HEURES)
            # ==============================
            second_level_dirs = sorted(
                [d for d in selected_main_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if second_level_dirs:

                selected_run_dir = st.selectbox(
                    "🕒 Choisir une heure",
                    second_level_dirs,
                    format_func=lambda x: x.name
                )

                # ==============================
                # 3. FILTRE EXTENSIONS
                # ==============================
                ext_filter = st.multiselect(
                    "Filtrer par type",
                    options=[".csv", ".json", ".pdf", ".png"],
                    default=[".csv", ".json", ".pdf", ".png"]
                )

                # ==============================
                # 4. FICHIERS
                # ==============================
                files_in_folder = []

                for file in selected_run_dir.iterdir():

                    if file.is_file():

                        if file.name.startswith("."):
                            continue

                        if file.suffix not in ext_filter:
                            continue

                        files_in_folder.append({
                            "name": file.name,
                            "path": file,
                            "mtime": file.stat().st_mtime
                        })

                files_in_folder = sorted(
                    files_in_folder,
                    key=lambda x: x["mtime"],
                    reverse=True
                )

                if files_in_folder:

                    selected_file_idx = st.selectbox(
                        "📄 Choisir un fichier",
                        range(len(files_in_folder)),
                        format_func=lambda i: files_in_folder[i]["name"]
                    )

                    f_info = files_in_folder[selected_file_idx]
                    f_path = f_info["path"]

                    st.divider()

                    st.markdown(f"**Fichier :** `{f_path.name}`")
                    st.caption(
                        f"Dernière modification : {time.ctime(f_info['mtime'])}"
                    )

                    # Téléchargement
                    with open(f_path, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger",
                            data=f,
                            file_name=f_path.name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )

                    # Aperçu CSV
                    if f_path.suffix == ".csv":

                        if st.button(
                            "👁️ Aperçu CSV",
                            use_container_width=True
                        ):
                            try:
                                df_preview = pd.read_csv(f_path, sep=';')
                                st.dataframe(df_preview.head(5))

                            except Exception as e:
                                st.error(
                                    f"Impossible de lire le CSV : {e}"
                                )

                else:
                    st.info("Aucun fichier trouvé.")

            else:
                st.info("Aucun sous-dossier trouvé.")

        else:
            st.info("Aucun dossier dans outputs.")

    else:
        st.info("Le dossier outputs n'existe pas. Il n'y a pas encore eu de runs sur cet appareil.")
if global_status["running"]:
    st.session_state.etape = 2
    st.session_state.fl_process = global_status["process"]
    
    # ON RE-REMPLIT LE SESSION STATE LOCAL DEPUIS LE GLOBAL
    conf = global_status["config"]
    st.session_state.selected_strategy = conf.get("strategy")
    st.session_state.selected_dataset = conf.get("dataset")
    st.session_state.selected_rounds = conf.get("rounds")
    st.session_state.selected_epochs = conf.get("epochs")
    st.session_state.selected_clients = conf.get("clients")
    st.session_state.selected_model_name = conf.get("model_name")
if st.session_state.etape == 1:
    with st.container():
        st.title("🌱 Green Federated Learning Platform")
        st.markdown("### 🛠️ Étape 1 : Configuration")
        # Logique de grisement
        if not GPU_AVAILABLE:
            st.checkbox(
                "Activer l'accélération GPU", 
                value=False, 
                disabled=True, 
                help="Option désactivée : Aucun GPU compatible CUDA n'a été détecté sur cette machine."
            )
            option_gpu = False
            num_gpus = 0.0
        else:
            option_gpu = st.checkbox(
                "Activer l'accélération GPU", 
                value=st.session_state.get("use_gpu", True),
                help="Utilise les cœurs CUDA pour l'entraînement local des clients."
            )
            num_gpus = 1.0 if option_gpu else 0.0

        # Optionnel : Un petit message d'alerte visuel si pas de GPU
        if not GPU_AVAILABLE:
            st.info("ℹ️ **Note :** L'entraînement se fera exclusivement sur CPU en raison de l'absence de GPU.")
        st.divider()

        col_m, col_d, col_p = st.columns(3)
        with col_m:
            st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #4CAF50;height:160px;"><h4>🧠 Architecture</h4><p>Modèle (.py)</p></div>', unsafe_allow_html=True)
            model_file = st.file_uploader("Fichier", type=["py"], label_visibility="collapsed")
        if model_file is not None:
            # Définition du chemin cible
            target_path = PROJECT_DIR / "pytorchexample" / "user_model.py"
            
            # Création du dossier s'il n'existe pas
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Écriture du contenu
            with open(target_path, "wb") as f:
                f.write(model_file.getbuffer())
            st.success(f"Modèle sauvegardé sous : {target_path.name}")       

        with col_d:
            st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #2196F3;height:160px;"><h4>📂 Données</h4><p>Jeu de données cible</p></div>', unsafe_allow_html=True)
            
            # 1. Sélection du type
            dataset_selection = st.selectbox("Dataset", ["uoft-cs/cifar10", "danjacobellis/chexpert", "custom (EXPERIMENTAL)"], label_visibility="collapsed")
            
            # 2. Assignation des valeurs (Logique robuste)
            if dataset_selection == "uoft-cs/cifar10":
                dataset = "uoft-cs/cifar10"
                
            elif dataset_selection == "danjacobellis/chexpert":
                dataset = "danjacobellis/chexpert"
                
            else: # Cas "custom"
                st.info("🛠️ **Configuration Modèle & Dataset Custom**")
                st.markdown("""
                <div style="font-size: 0.85rem; line-height: 1.4;">
                ⚠️ <b>Conditions de succès :</b>
                <ul>
                    <li><b>Labels : </b> Fonctionne uniquement avec du mono-labeling (càd un seul label par image)
                    <li><b>Dataset :</b> Doit être sur HuggingFace (Public ou avec Token).</li>
                    <li><b>Mapping :</b> Colonnes nommées <code>image</code> (ou <code>img</code>) et <code>label</code>.</li>
                    <li><b>Cohérence :</b> Assurez-vous que la <b>Taille Image</b> et les <b>Canaux</b> saisis correspondent strictement à l'architecture de votre modèle.</li>
                    <li><b>Sortie :</b> Le modèle doit retourner un objet avec un attribut <code>.logits</code> ou être compatible avec votre boucle d'entraînement.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                dataset = st.text_input("Chemin HuggingFace", value="", help="ex: user/my-dataset")
        st.markdown("#### Paramètres du modèle")        
        c_img, c_chan, c_cls = st.columns(3)
        img_size = c_img.number_input("Taille Image", value=32, min_value=1)
        num_channels = c_chan.selectbox("Canaux", [1, 3], index=1)
        num_classes = c_cls.number_input("Nombre de classes", value=10, min_value=2)
        with col_p:
            st.markdown('<div style="background-color:#f0f2f6;padding:20px;border-radius:15px;border-left:5px solid #4CAF50;height:160px;"><h4>🧠 Poids</h4><p>Modèle (.pt ou .pth)</p></div>', unsafe_allow_html=True)
            model_weights = st.file_uploader("Fichier", type=["pt", "pth"], label_visibility="collapsed")
        if model_weights is not None:
            # Définition du chemin cible
            target_path2 = PROJECT_DIR / "final_model.pt"
            
            # Création du dossier s'il n'existe pas
            target_path2.parent.mkdir(parents=True, exist_ok=True)
            
            # Écriture du contenu
            with open(target_path2, "wb") as f:
                f.write(model_weights.getbuffer())
            st.success(f"Modèle sauvegardé sous : {target_path2.name}")
        with st.expander("📖 Consulter le guide des stratégies (État de l'Art)", expanded=False):
            st.markdown("""
| Stratégie | Points Forts | Points Faibles | Impact Énergie |
| :--- | :--- | :--- | :--- |
| **FedAvg** | Simplicité / Baseline | Très mauvais si données hétérogènes | 🟢 Faible |
| **FedProx** | Robustesse (Clients lents) | Dépend fortement du choix de μ | 🟡 Moyen |
| **FedAdagrad**| **Données éparses / Adaptatif** | Moins efficace au fil des rounds | 🟡 Moyen |
| **FedAdam/Yogi**| Convergence ultra-rapide | Dépend de nombreux hyperparamètres | 🔴 Élevé |
            """)
            st.info("💡 **Focus FedAdagrad** : Idéal si vos clients ont des données très spécifiques (ex: imagerie médicale rare) car elle adapte le pas d'apprentissage à la rareté des features.")
        st.write("")      
        st.markdown("#### 🚀 Hyperparamètres de base")
        c_s, c_r, c_e, c_l, c_br = st.columns(5)
        with c_s:
            strategie = st.selectbox("Stratégie", ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad"])
        with c_r:
            rounds = st.number_input("Rounds", min_value=1, value=1)
        with c_e:
            epochs = st.selectbox("Epochs locales", [1, 2, 3])
        with c_l:
            lr = st.number_input("Learning Rate", min_value=0.0001, value=0.01, format="%.4f")
        with c_br:
            batch_size = st.number_input("Batch Size", min_value = 2, value = 32)
        st.info(OPTIM_TIPS.get(strategie, "Sélectionnez une stratégie."))
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
        alpha = st.slider("Alpha", 0.0, 100.0, 0.1, help="Plus alpha est petit, plus les données sont hétérogènes entre les clients. Avec alpha proche de 0, chaque client aura principalement des exemples d'une seule classe. Avec alpha élevé, les données seront plus équilibrées entre les clients. Ceci permet de simuler un label skew.")
        self_balancing = st.checkbox("Équilibrage automatique", value=True, help="Activez cette option pour que chaque client ait une quantité de données similaire, même avec un alpha faible. Le partitionnement restera hétérogène en termes de classes, mais la taille des partitions sera plus équilibrée.")
        
        st.markdown("#### 🌫️ Dégradation des images par client (Feature Skew)")
        st.caption("Simule des clients avec des capteurs de qualité variable — certains clients reçoivent des images floues, d'autres non.")

        blur_mode = st.radio(
            "Mode de partition du flou",
            ["✅ Aucun flou", "🎲 Partition automatique", "🎛️ Partition arbitraire"],
            horizontal=True,
            help="Automatique : vous choisissez quels clients sont floutés, le niveau est tiré aléatoirement. Arbitraire : vous définissez le niveau de flou de chaque client manuellement."
        )

        blur_config = {}  # vide = pas de flou pour personne

        if blur_mode == "🎲 Partition automatique":
            st.markdown("**Sélectionnez les clients à dégrader :**")
            affected_clients = []
            cols_per_row = 10
            client_ids = list(range(clients_number))
            rows = [client_ids[i:i+cols_per_row] for i in range(0, len(client_ids), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for col, cid in zip(cols, row):
                    checked = col.checkbox(f"C{cid}", key=f"auto_blur_client_{cid}", value=False)
                    if checked:
                        affected_clients.append(cid)

            if affected_clients:
                max_blur = st.slider(
                    "Niveau de flou maximum (les clients sélectionnés auront un % aléatoire entre 10% et ce max)",
                    min_value=10, max_value=100, value=60, format="%d%%"
                )
                rng = random.Random(42)  # Seed fixe pour reproductibilité
                for cid in affected_clients:
                    blur_config[cid] = rng.randint(10, max_blur)
                st.markdown("**Aperçu de la partition générée :**")
                preview_cols = st.columns(min(len(affected_clients), 5))
                for i, cid in enumerate(affected_clients):
                    col = preview_cols[i % 5]
                    pct = blur_config[cid]
                    emoji = "🟡" if pct <= 30 else ("🟠" if pct <= 70 else "🔴")
                    col.metric(f"Client {cid}", f"{emoji} {pct}%")
            else:
                st.info("Aucun client sélectionné — aucun flou ne sera appliqué.")

        elif blur_mode == "🎛️ Partition arbitraire":
            st.markdown("**Définissez le niveau de flou pour chaque client :**")
            st.caption("0% = image nette, 100% = flou maximal. Laissez à 0 pour les clients non dégradés.")
            cols_per_row = 5
            client_ids = list(range(clients_number))
            rows = [client_ids[i:i+cols_per_row] for i in range(0, len(client_ids), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for col, cid in zip(cols, row):
                    val = col.number_input(
                        f"C{cid} (%)",
                        min_value=0, max_value=100, value=0, step=5,
                        key=f"arb_blur_client_{cid}"
                    )
                    if val > 0:
                        blur_config[cid] = val
            if blur_config:
                st.markdown("**Récapitulatif :**")
                summary_cols = st.columns(min(len(blur_config), 5))
                for i, (cid, pct) in enumerate(blur_config.items()):
                    emoji = "🟡" if pct <= 30 else ("🟠" if pct <= 70 else "🔴")
                    summary_cols[i % 5].metric(f"Client {cid}", f"{emoji} {pct}%")
            else:
                st.info("Tous les clients à 0% — aucun flou ne sera appliqué.")

        st.markdown("#### ⚖️ Hétérogénéité des clients (Quantity Skew)")

        c1, c2, c3 = st.columns(3)

        with c1:
            small_c = st.number_input(
                "Clients Small (10%)",
                min_value=0,
                max_value=clients_number,
                value=st.session_state.selected_small_clients
            )

        with c2:
            medium_c = st.number_input(
                "Clients Medium (50%)",
                min_value=0,
                max_value=clients_number,
                value=st.session_state.selected_medium_clients
            )

        with c3:
            big_c = st.number_input(
                "Clients Big (100%)",
                min_value=0,
                max_value=clients_number,
                value=st.session_state.selected_big_clients
            )

        total = small_c + medium_c + big_c

        if total != clients_number:
            st.error(f"⚠️ Total clients ({total}) ≠ nombre total ({clients_number})")
        else:
            st.success("✔ Répartition des clients valide")
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
            if wandb_mode == "online":
                if not wandb_api_key:
                    st.error("❌ Erreur : Clé API manquante.")
                    st.stop()
                
                with st.spinner("Test de connexion WandB en cours..."):
                    try:
                        # 1. On force l'authentification avec la clé saisie
                        wandb.login(key=wandb_api_key, relogin=True, force=True)
                        
                        # 2. On lance un run de test très court
                        test_run = wandb.init(
                            project=wandb_project if wandb_project else "test-connection",
                            entity=wandb_entity if wandb_entity else None,
                            name="connection-check",
                            job_type="test",
                            reinit=True # Permet de réinitialiser si un run existait déjà
                        )
                        
                        
                        # 3. On ferme immédiatement le run
                        test_run.finish()
                        #Clean up les restes éventuels
                        if hasattr(wandb, "setup"):
                            wandb.setup()._teardown()
                        # --- NETTOYAGE CRUCIAL ---
                        # On réinitialise les variables d'environnement pour que le 
                        # prochain run (dans le sous-processus) reparte de zéro
                        if "WANDB_API_KEY" in os.environ:
                            del os.environ["WANDB_API_KEY"]
                        
                        st.success("✅ Connexion validée !")
                        
                    except Exception as e:
                        # Si le 401 arrive ici, on capture et on bloque
                        st.error("❌ Échec de l'authentification WandB.")
                        st.error(f"Détails : {str(e)}")
                        
                        # On tente de fermer si c'est resté ouvert
                        try: wandb.finish()
                        except: pass
                        
                        st.stop() # LE LANCEMENT EST ANNULÉ ICI
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
            st.session_state.selected_blur_config = blur_config
            st.session_state.selected_model_name = model_file.name if model_file else "Défaut"
            st.session_state.known_csv_files_before_run = get_all_emission_csvs("emission.csv")
            st.session_state.known_csv_files_before_run_2 = get_all_emission_csvs("EXCEL_emissions_history.csv")
            st.session_state.known_csv_files_before_run_3 = get_all_emission_csvs("EXCEL_eval_emissions_history.csv")
            st.session_state.selected_small_clients = small_c
            st.session_state.selected_medium_clients = medium_c
            st.session_state.selected_big_clients = big_c
            
            write_pyproject_with_config(batch_size, num_gpus,strategie, rounds, epochs, lr, frac_train, frac_eval, clients_number, extra_opts, alpha, self_balancing, small_c, medium_c, big_c, dataset, img_size, num_channels, num_classes, blur_config)
            
            env = os.environ.copy()
            if wandb_project:
                env["WANDB_PROJECT"] = wandb_project
            env["WANDB_MODE"] = wandb_mode
            
            if wandb_api_key:
                env["WANDB_API_KEY"] = wandb_api_key
            if wandb_entity:
                env["WANDB_ENTITY"] = wandb_entity

            # Lancement du processus avec le nouvel environnement
            """st.session_state.fl_process = subprocess.Popen(
                ["flwr", "run", "."], 
                cwd=PROJECT_DIR, 
                env=env
            )
            global_status["running"] = True
            global_status["process"] = st.session_state.fl_process
            global_status["config"] = {
                "strategy": strategie,
                "dataset": dataset,
                "rounds": rounds,
                "epochs": epochs,
                "clients": clients_number,
                "model_name": model_file.name if model_file else "Défaut"
            }
            st.session_state.etape = 2
            st.rerun()"""

            # --- BLOC DE LANCEMENT SÉCURISÉ ---
            log_file_path = PROJECT_DIR / "federated_log.txt"
            
            # 1. On force l'encodage UTF-8 pour Python
            env_with_utf8 = env.copy()
            env_with_utf8["PYTHONIOENCODING"] = "utf-8"

            # 2. On ouvre le fichier log en UTF-8 explicitement
            log_f = open(log_file_path, "w", encoding="utf-8")
            
            st.session_state.fl_process = subprocess.Popen(
                ["flwr", "run", "."], 
                cwd=PROJECT_DIR, 
                env=env_with_utf8, 
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            global_status["running"] = True
            global_status["process"] = st.session_state.fl_process
            global_status["config"] = {
                "strategy": strategie,
                "dataset": dataset,
                "rounds": rounds,
                "epochs": epochs,
                "clients": clients_number,
                "model_name": model_file.name if model_file else "Défaut"
            }
            st.session_state.etape = 2
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# ÉCRAN 2 : ENTRAÎNEMENT EN COURS
# ════════════════════════════════════════════════════════════════════
elif st.session_state.etape == 2:
    st.title("🔄 Étape 2 : Entraînement en cours")
    st.divider()

    # Résumé config — lecture seule
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stratégie", st.session_state.get("selected_strategy", "N/A"))
    c2.metric("Dataset", st.session_state.get("selected_dataset", "N/A"))
    c3.metric("Rounds", st.session_state.get("selected_rounds", "N/A"))
    c4.metric("Epochs", st.session_state.get("selected_epochs", "N/A"))
    c5.metric("Clients", st.session_state.get("selected_clients", "N/A"))
    st.caption(f"Modèle : {st.session_state.get('selected_model_name', 'N/A')}")
    st.divider()

    # --- AFFICHAGE DES LOGS ---
    st.subheader("💻 Console d'exécution en temps réel")
    log_container = st.empty()
    log_file_path = PROJECT_DIR / "federated_log.txt"

    if log_file_path.exists():
        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                last_logs = "".join(lines[-15:]) # Affiche les 15 dernières lignes
                log_container.code(last_logs, language="bash")
        except Exception:
            log_container.info("Synchronisation des logs...")
    else:
        log_container.info("Initialisation du serveur Flower...")

    # --- ÉTAT DU PROCESSUS ---
    process_running = False
    if st.session_state.fl_process is not None:
        process_running = st.session_state.fl_process.poll() is None

    # Récupération du CSV de résultats si disponible
    if st.session_state.current_run_csv is None:
        csv = get_new_csv_after_run("emission.csv")
        if csv:
            st.session_state.current_run_csv = csv

    st.subheader("État du processus")
    st.write("🏃 En cours..." if process_running else "🏁 Terminé")

    # Si terminé, on passe à l'étape 3
    if not process_running and st.session_state.fl_process is not None:
        st.success("Entraînement terminé avec succès.")
        global_status["running"] = False 
        global_status["process"] = None
        st.session_state.etape = 3
        time.sleep(1) # Petit délai pour laisser le temps au fichier CSV d'être finalisé
        st.rerun()

    # --- BARRE DE NAVIGATION / ACTIONS ---
    col_nav = st.columns([1, 1, 4])
    if col_nav[0].button("⏹️ Arrêter", type="secondary"):
        if st.session_state.fl_process is not None:
            try:
                parent = psutil.Process(st.session_state.fl_process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                st.info("Processus arrêté par l'utilisateur.")
            except psutil.NoSuchProcess:
                st.warning("Le processus était déjà terminé.")
            except Exception as e:
                st.error(f"Erreur lors de l'arrêt : {e}")
        
        global_status["running"] = False 
        global_status["process"] = None
        st.session_state.etape = 1
        st.rerun()

    # Boucle de rafraîchissement automatique
    if process_running:
        time.sleep(1)
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# ÉCRAN 3
# ════════════════════════════════════════════════════════════════════
elif st.session_state.etape == 3:

    st.title("📊 Étape 3 : Résultats finaux")
    csv_path = st.session_state.get('current_run_csv') or get_latest_csv("emission.csv")
    
    if csv_path:
        df_res = read_csv_safely(csv_path)
        if df_res is not None:
            pdf_path = Path(csv_path).parent / "rapport_green_fl_auto.pdf"
            
            # Forcer la génération même s'il existe (pour le test)
            try:
                # On passe un maximum d'infos pour être sûr
                generate_pdf_report(df_res, st.session_state, save_path=str(pdf_path))
                st.success(f"PDF créé : {pdf_path}")
            except Exception as e:
                st.error(f"⚠️ ÉCHEC GÉNÉRATION PDF : {str(e)}")
                import traceback
                st.code(traceback.format_exc()) # Affiche la pile d'erreur complète
    output_dir = Path(csv_path).parent if csv_path else None
    df_res = read_csv_safely(csv_path)
    global_accuracy = None
    if output_dir:
        global_accuracy = read_accuracy_value(output_dir)
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
        water_consumed = last.get("water_consumed")
        c8.metric("Eau consommée", safe_value(water_consumed, " L"))
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
        path_comp = get_latest_csv("em_comparison.json")
        if path_comp and path_comp.exists():
            with open(path_comp, "r") as f:
                comp_data = json.load(f)
            
            st.subheader("🌐 Impact Réseau & Global")
            
            # Création de 3 colonnes pour le réseau
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("Volume Transféré", f"{comp_data.get('network_transfer_gb', 0):.4f} GB")
            with c2:
                st.metric("Énergie Réseau", f"{comp_data.get('network_energy_kwh', 0):.6f} kWh", help="Basé sur 0.06 kWh/GB")
            with c3:
                # On affiche le total combiné (Matériel + Réseau)
                total_kwh = comp_data.get('total_combined_energy_kwh', 0)
                st.metric("Consommation Totale", f"{total_kwh:.6f} kWh")
                intensity_avg = comp_data.get('intensity_avg_gco2_kwh',0)
            total_combined_CO2 = (total_kwh * intensity_avg) #Formule
        water_consumed = float(
            str(last.get("water_consumed", 0)).replace(",", ".")
        )

        st.metric(
            "Score Green",
            f"{global_accuracy / (total_combined_CO2 + water_consumed):.6f}"
        )

        with st.expander("Voir les données brutes du CSV"):
            st.dataframe(df_res, width="stretch")
        csv_1, csv_2, csv_3, csv_4, csv_5, fichier_txt = st.columns(6)
        with csv_1:
            st.download_button("📥 Télécharger CSV", data=df_res.to_csv(index=False, sep =';').encode('utf-8'),
                               file_name="emission.csv", mime="text/csv")
        with csv_2:
            pdf_bytes = generate_pdf_report(df_res, st.session_state)
            st.download_button("📄 Télécharger PDF", data=pdf_bytes,
                               file_name="rapport_green_fl.pdf", mime="application/pdf")
        with csv_3:
            path_hist = get_latest_csv("EXCEL_emissions_history.csv")
            if path_hist:
                df_hist = pd.read_csv(path_hist, sep=';')
                st.download_button("📥 Historique Global", 
                                   data=df_hist.to_csv(index=False, sep=';').encode('utf-8'),
                                   file_name="history.csv", mime="text/csv")
            else:
                st.info("Historique non dispo.")

        # 3. Le rapport d'évaluation
        with csv_4:
            path_eval = get_latest_csv("EXCEL_eval_emissions_history.csv")
            if path_eval:
                df_eval = pd.read_csv(path_eval, sep=';')
                st.download_button("📥 Rapport Évaluation", 
                                   data=df_eval.to_csv(index=False, sep=';').encode('utf-8'),
                                   file_name="evaluation.csv", mime="text/csv")
            else:
                st.info("Éval non dispo.")
        with csv_5:
            file_path = PROJECT_DIR /"Etat_de_l_art_green_FL.pdf"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    st.download_button(
                        "📘 État de l'art",
                        data=f,
                        file_name="Etat_de_l_art_green_FL.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("Fichier État de l'art non trouvé.")
        with fichier_txt:
            st.markdown("##### 📄 Données JSON")
            
            # --- 1. FICHIER GRID CONTEXT (Initial) ---
            path_grid = get_latest_csv("grid_context.json")
            if path_grid and path_grid.exists():
                with open(path_grid, "r", encoding="utf-8") as f:
                    grid_data = f.read()
                st.download_button(
                    label="📥 Grid Context (Initial)",
                    data=grid_data,
                    file_name="grid_context_initial.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # --- 2. FICHIER BILAN COMPLET (Comparaison + Réseau) ---
            path_comp = get_latest_csv("em_comparison.json")
            if path_comp and path_comp.exists():
                with open(path_comp, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                
                # Petit rappel visuel des métriques réseau avant le bouton
                st.info(f"🌐 Réseau : {comp_data.get('network_transfer_gb', 0):.4f} GB")
                
                json_string = json.dumps(comp_data, indent=2)
                st.download_button(
                    label="📊 Bilan Complet (Final)", 
                    data=json_string,
                    file_name=f"bilan_global_{st.session_state.get('strategy', 'fl')}.json", 
                    mime="application/json",
                    use_container_width=True
                )
            
            if not path_grid and not path_comp:
                st.info("Aucune donnée JSON disponible.")
            

    else:
        st.warning("Aucun fichier emission.csv exploitable trouvé.")

    if st.button("🔄 Nouvelle expérience"):
        st.session_state.etape = 1
        st.session_state.current_run_csv = None
        st.session_state.fl_process = None
        st.rerun()
