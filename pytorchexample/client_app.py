"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.model import Net
from pytorchexample.task import load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
import csv
from pathlib import Path
import psutil
from codecarbon import EmissionsTracker

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_info = psutil.virtual_memory()
    partition_id = context.node_config["partition-id"]
    
    print(f"[monitoring client{partition_id}]")
    print(f"charge de cpu est : {cpu_usage}")
    print(f"La RAM utilisée  : {ram_info.percent}% ({ram_info.used / 1024**3:.2f} GB)")

    # Load the model
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    current_round = msg.content["config"].get("server_round", 0)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.1)
    self_balancing = context.run_config.get("self_balancing", True)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, alpha, self_balancing)

    # --- Logique de chemin du PREMIER CODE ---
    strategy_name = context.run_config.get("strategy", "unknown")
    run_id = context.run_config.get("run_id", "1")
    save_path_str = msg.content.get("config", {}).get("save_path", ".")
    save_path = Path(save_path_str)
    save_path.mkdir(parents=True, exist_ok=True)

    # Fichier CSV classique (Ressources)
    log_file = save_path / "client_stats.csv"
    file_exists = log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow(["Execution_No", "Strategy", "Round", "id_client", "POURCENTAGE_CPU", "POURCENTAGE_RAM"])
        writer.writerow([run_id, strategy_name, msg.metadata.group_id, partition_id, cpu_usage, ram_info.percent])

    # --- Intégration CodeCarbon (utilise save_path du premier code) ---
    tracker = EmissionsTracker(
        project_name=f"client_{partition_id}_round_{current_round}_train",
        output_dir=str(save_path),
        output_file="emissions_history.csv", 
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

    # Construct and return reply Message
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
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # On récupère le chemin de sauvegarde pour l'évaluation aussi
    save_path_str = msg.content.get("config", {}).get("save_path", ".")
    save_path = Path(save_path_str)
    save_path.mkdir(parents=True, exist_ok=True)
    current_round = msg.content["config"].get("server_round", 0)

    # --- CodeCarbon pour Evaluate ---
    tracker = EmissionsTracker(
        project_name=f"client_{partition_id}_round_{current_round}_eval",
        output_dir=str(save_path),
        output_file="emissions_eval_history.csv", 
        on_csv_write="append",
        measure_power_secs=1
    )
    
    tracker.start()
    try:
        eval_loss, eval_acc = test_fn(model, valloader, device)
    finally:
        tracker.stop()

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
