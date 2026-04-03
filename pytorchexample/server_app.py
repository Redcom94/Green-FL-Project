"""pytorchexample: A Flower / PyTorch app."""

import torch
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorchexample.custom_strategy import (
    CustomFedAvg, 
    CustomFedAdam, 
    CustomFedYogi, 
    CustomFedProx, 
    CustomFedAdagrad
)
import matplotlib.pyplot as plt
import pandas as pd
from pytorchexample.task import load_centralized_dataset, test
try:
    from pytorchexample.user_model import Net  # Importe le modèle fourni par l'utilisateur
except ImportError:
    from pytorchexample.model import Net       # Modèle par défaut si aucun n'est fourni
from codecarbon import EmissionsTracker
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
    

    fraction_train = config.get("fraction-train", 0.1)
    fraction_evaluate = config.get("fraction-evaluate", 0.1)
    min_available_nodes = config.get("num-supernodes", 10)
    min_train = config.get("num-supernodes-training", 8)
    min_eval = config.get("num-supernodes-evaluation", 5)
    # 2. Model Initialization
    global_model = Net()
    model_path = Path("final_model.pt")
    if model_path.exists():
        global_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Strategy Factory
    strategy_kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": min_train,          
        "min_evaluate_nodes": min_eval,
        "min_available_nodes": min_available_nodes,
    }
    if strategy_name == "fedprox":
        strategy_kwargs["proximal_mu"] = config.get("proximal-mu", 0.1)
    elif strategy_name in ["fedadam", "fedyogi"]:
        strategy_kwargs.update({
            "eta": config.get("server-learning-rate", 1.0),  # server lr
            "eta_l": lr,                                     # client lr
            "beta_1": config.get("beta-1", 0.9),
            "beta_2": config.get("beta-2", 0.99),
            "tau": config.get("tau", 1e-9),  # epsilon / stabilité numérique
        })
    elif strategy_name == "fedadagrad":
        strategy_kwargs.update({
            "eta": config.get("server-learning-rate", 1.0),  # server lr
            "eta_l": lr,                                     # client lr
            "tau": config.get("tau", 1e-9),  # epsilon / stabilité numérique
        })
    

    strategy = STRATEGIES[strategy_name](**strategy_kwargs)

    # 4. Préparation du dossier de sortie
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    strategy.set_save_path(save_path)

    # 5. Lancement de l'entraînement
    tracker=EmissionsTracker(
        project_name=config.get("strategy", "fedavg").lower(),
        output_dir=str(save_path),
        output_file="emission.csv",
        measure_power_secs=15
        )
    tracker.start()
    try:
        print(f"🚀 Starting Federated Learning with strategy: {strategy_name.upper()}")
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr,
                                       "save_path":str(save_path),
                                       "strategy":strategy_name,
                                       "run_id":"01"}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )
    finally:
        tracker.stop()
        """ le ficheir intel_power_gadget est un fichier cree directement par le drive intel, code carbone ne fzit que de l'ouvrir pour extraire les puissance 
        ici l'idee est d'utiliser une boucle afin de forcer toute les fichier csv cree de passer en point de virgul quelque soit qui l'a cree"""
        toute_fichier_csv=list(save_path.glob("*csv"))
        for csv_path in toute_fichier_csv:
            try:
                df_temp=pd.read_csv(csv_path,sep=None,engine='python')
                df_temp.to_csv(csv_path,sep=';',index=False)
                print(f" fichier{csv_path.name}")
            
            except Exception as e :
                print(f"impossible de convertir {csv_path.name} : {e}")
        generate_emission_chart(save_path,strategy_name)

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

def generate_emission_chart(output_path: Path,strategy_name:str):
    """Génère un graphique à partir du fichier CSV de CodeCarbon."""
    csv_file = output_path / "emission.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file, sep = ';')
        
        # Exemple de graphe : Énergie consommée par composant
        components = ['cpu_energy', 'gpu_energy', 'ram_energy']
        values = [df[c].sum() for c in components if c in df.columns]
        labels = [c.replace('_energy', '').upper() for c in components if c in df.columns]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=['#4CAF50', '#2196F3', '#FF9800'])
        plt.ylabel('Energy Consumed (kWh)')
        plt.title(f'Energy Consumption by Hardware Component,{strategy_name.upper()}')
        plt.savefig(output_path / f"{strategy_name.lower()}_energy_breakdown.png")
        plt.close()