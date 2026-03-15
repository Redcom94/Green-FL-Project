"""pytorchexample: A Flower / PyTorch app."""

import torch
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from .custom_strategy import (
    CustomFedAvg, 
    CustomFedAdam, 
    CustomFedYogi, 
    CustomFedProx, 
    CustomFedAdagrad
)

from pytorchexample.task import load_centralized_dataset, test
try:
    from .user_model import Net  # Importe le modèle fourni par l'utilisateur
except ImportError:
    from .model import Net       # Modèle par défaut si aucun n'est fourni

# Dictionnaire de correspondance
STRATEGIES = {
    "fedavg": CustomFedAvg,
    "fedadam": CustomFedAdam,
    "fedyogi": CustomFedYogi,
    "fedprox": CustomFedProx,
    "fedadagrad": CustomFedAdagrad,
}

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # 1. Configuration with safer defaults
    config = context.run_config
    strategy_name = config.get("strategy", "fedavg").lower()
    num_rounds = config.get("num-server-rounds", 10)
    lr = config.get("learning-rate", 0.01)
    
    # Handle the typo gracefully
    fraction_train = config.get("fraction-train", config.get("franction-train", 0.1))
    fraction_evaluate = config.get("fraction-evaluate", 0.1)

    # 2. Model Initialization
    global_model = Net()
    model_path = Path("final_model.pt")
    if model_path.exists():
        global_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Strategy Factory
    strategy_kwargs = {
        "fraction_train": fraction_train, # Reviens à fraction_train
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": 8,            # Reviens à min_train_nodes
        "min_evaluate_nodes": 5,         # Reviens à min_evaluate_nodes
        "min_available_nodes": 10,       # Reviens à min_available_nodes
    }
    if strategy_name == "fedprox":
        strategy_kwargs["proximal_mu"] = config.get("proximal-mu", 0.1)
    elif strategy_name in ["fedadam", "fedyogi"]:
        strategy_kwargs.update({
            "eta": config.get("server-learning-rate", 1.0),
            "eta_l": lr,
            "beta_1": config.get("beta-1", 0.9),
        })

    strategy = STRATEGIES[strategy_name](**strategy_kwargs)

    # 4. Préparation du dossier de sortie
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    strategy.set_save_path(save_path)

    # 5. Lancement de l'entraînement
    print(f"🚀 Starting Federated Learning with strategy: {strategy_name.upper()}")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # 6. Sauvegarde du modèle final
    print("\n💾 Saving final model to disk...")
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, save_path / "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})