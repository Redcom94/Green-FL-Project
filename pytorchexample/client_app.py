"""pytorchexample: A Flower / PyTorch app."""

import torch
import pandas as pd
import csv
import psutil
from pathlib import Path
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from codecarbon import EmissionsTracker

from pytorchexample.model import Net
from pytorchexample.task import load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# --- Fonction Utilitaire pour corriger le format CSV (Point vers Virgule) ---
def harmoniser_csv_format(file_path: Path):
    """
    Lit un CSV au format US (virgule et point décimal) 
    et le réécrit au format FR (point-virgule et virgule décimale).
    """
    try:
        if file_path.exists():
#           On lit l'original (toujours en US)
            df = pd.read_csv(file_path, sep=',', decimal='.')
            
            # On sauvegarde dans un NOUVEAU fichier pour Excel
            excel_path = file_path.with_name(f"EXCEL_{file_path.name}")
            df.to_csv(excel_path, sep=';', decimal=',', index=False)
            print(f"✅ Version Excel générée : {excel_path.name}")
    except Exception as e:
        print(f"⚠️ Impossible de convertir {file_path.name} : {e}")

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_info = psutil.virtual_memory()
    partition_id = context.node_config["partition-id"]
    
    print(f"[monitoring client {partition_id}]")
    print(f"Charge CPU : {cpu_usage}%")
    print(f"RAM utilisée : {ram_info.percent}% ({ram_info.used / 1024**3:.2f} GB)")

    # 1. Chargement du modèle
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    current_round = msg.content["config"].get("server_round", 0)

    # 2. Chargement des données
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.1)
    self_balancing = context.run_config.get("self_balancing", True)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, alpha, self_balancing)

    # 3. Chemins de sauvegarde
    strategy_name = context.run_config.get("strategy", "unknown")
    run_id = context.run_config.get("run_id", "1")
    save_path_str = msg.content.get("config", {}).get("save_path", ".")
    save_path = Path(save_path_str)
    save_path.mkdir(parents=True, exist_ok=True)

    # 4. Log des ressources (CPU/RAM)
    log_file = save_path / "client_stats.csv"
    file_exists = log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';') # On garde le point-virgule ici
        if not file_exists:
            writer.writerow(["Execution_No", "Strategy", "Round", "id_client", "POURCENTAGE_CPU", "POURCENTAGE_RAM"])
        writer.writerow([run_id, strategy_name, msg.metadata.group_id, partition_id, cpu_usage, ram_info.percent])

    # 5. Entraînement avec CodeCarbon
    emissions_file = "emissions_history.csv"
    tracker = EmissionsTracker(
        project_name=f"client_{partition_id}_round_{current_round}_train",
        output_dir=str(save_path),
        output_file=emissions_file, 
        on_csv_write="append",
        measure_power_secs=1
    )
    
    tracker.start()
    try:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
    finally:        
        tracker.stop()
        # Correction immédiate du fichier CodeCarbon pour Excel FR
        harmoniser_csv_format(save_path / emissions_file)

    # 6. Retour des résultats
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "client_cpu": cpu_usage,
        "client_ram": ram_info.percent,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # 1. Récupération des paramètres
    partition_id = context.node_config["partition-id"]
    current_round = msg.content["config"].get("server_round", 0)
    
    # Chemins de sauvegarde (Logique identique au train)
    save_path_str = msg.content.get("config", {}).get("save_path", ".")
    save_path = Path(save_path_str)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. Préparation Modèle / Device
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Données
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # 4. CodeCarbon - Logique de fichier identique au train
    eval_emissions_file = "eval_emissions_history.csv"
    tracker = EmissionsTracker(
        project_name=f"client_{partition_id}_round_{current_round}_eval",
        output_dir=str(save_path),
        output_file=eval_emissions_file, 
        on_csv_write="append",
        measure_power_secs=1
    )
    
    tracker.start()
    try:
        eval_loss, eval_acc = test_fn(model, valloader, device)
    finally:
        tracker.stop()
        # Appel de la fonction utilitaire (Identique au train)
        harmoniser_csv_format(save_path / eval_emissions_file)

    # 5. Retour des métriques
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    return Message(content=RecordDict({"metrics": metric_record}), reply_to=msg)
