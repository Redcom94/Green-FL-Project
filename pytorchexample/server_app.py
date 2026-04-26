"""pytorchexample: A Flower / PyTorch app."""

import torch
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import csv

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorchexample.custom_strategy import (
    CustomFedAvg, 
    CustomFedAdam, 
    CustomFedYogi, 
    CustomFedProx, 
    CustomFedAdagrad
)
from pytorchexample.task import load_centralized_dataset, test

try:
    from pytorchexample.user_model import Net  # Modèle utilisateur
except ImportError:
    from pytorchexample.model import Net       # Modèle par défaut

from codecarbon import EmissionsTracker

# Dictionnaire de correspondance des stratégies
STRATEGIES = {
    "fedavg": CustomFedAvg,
    "fedadam": CustomFedAdam,
    "fedyogi": CustomFedYogi,
    "fedprox": CustomFedProx,
    "fedadagrad": CustomFedAdagrad,
}

app = ServerApp()

import requests

def get_carbon_intensity_realtime(zone: str = "BE") -> dict:
    """cette fonction sert a recuperer l'intensite en temps reel via electricite maps 
    pour cette partie j'ai travaillr avec cloud ia"""
    try:
        r1 = requests.get(
            f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}",
            headers={"auth-token": "x3Sd4MfSmzaXK6ppYtTd"},
            timeout=5
        )
        r2 = requests.get(
            f"https://api.electricitymap.org/v3/power-breakdown/latest?zone={zone}",
            headers={"auth-token": "x3Sd4MfSmzaXK6ppYtTd"},
            timeout=5
        )


        d1=r1.json()
        d2=r2.json()




        
        return {
            "zone": zone,
            "realtime_carbon_intensity": d1.get("carbonIntensity"),
            "datetime": d1.get("datetime"),
            "fossil_free_percentage": d2.get("fossilFreePercentage"),
            # NOUVEAUX CHAMPS
            "renewable_percentage":    d2.get("renewablePercentage"),
            "nuclear_mw":   d2.get("powerConsumptionBreakdown", {}).get("nuclear"),
            "wind_mw":      d2.get("powerConsumptionBreakdown", {}).get("wind"),
            "solar_mw":     d2.get("powerConsumptionBreakdown", {}).get("solar"),
            "gas_mw":       d2.get("powerConsumptionBreakdown", {}).get("gas"),
            "coal_mw":      d2.get("powerConsumptionBreakdown", {}).get("coal"),
        }
    except Exception as e:
        print(f"⚠️ Electricity Maps indisponible : {e}")
        return {"realtime_carbon_intensity": None}





@app.main()
def main(grid: Grid, context: Context) -> None:

    """ j'ajoute ce petite script afin de forcer la detection du gpu , car dans certaines ordinateur comme le mien on trouve des gpu intégre avec cpu et pas isoler 
    exemple : intel R iris R xe graphics . ce qui  explique aussi  pourquoi avant on avait tjr 0 watts comme consommation de gpu (alorque que sa consommation reel etait compte comme cpu) mais des qu'on a inclu les cluster on a arrivé a separer le gpu et cpu 
    """
    if torch.cuda.is_available():
        device_type="GPU(NVIDIA)"
        gpu_name=torch.cuda.get_device_name(0)
        print(f"le gpu detecté est : {gpu_name}")
    else:
        print("aucun gpu nvidia n'est detecté. il y a juste la presence de cpu et gpu integré ")
        print("la consommation du gpu sera incluse dans ' cpu_energy'  ")
    # 1. Configuration
    config = context.run_config
    dataset_name = config.get("dataset-name", "uoft-cs/cifar10")
    n_cls = context.run_config.get("num-classes", 10)
    img_size = config.get("img-size", 32)
    num_channels = config.get("num-channels", 3)
    strategy_name = config.get("strategy", "fedavg").lower()
    num_rounds = config.get("num-server-rounds", 10)
    lr = config.get("learning-rate", 0.01)

    fraction_train = config.get("fraction-train", 0.1)
    fraction_evaluate = config.get("fraction-evaluate", 0.1)
    min_available_nodes = config.get("num-supernodes", 10)
    min_train = config.get("num-supernodes-training", 8)
    min_eval = config.get("num-supernodes-evaluation", 5)

    # 2. Initialisation du modèle
    global_model = Net()
    model_path = Path("final_model.pt")
    if model_path.exists():
        global_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Configuration de la stratégie
    strategy_kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": min_train,          
        "min_evaluate_nodes": min_eval,
        "min_available_nodes": min_available_nodes,
    }

    if strategy_name == "fedprox":
        strategy_kwargs["proximal_mu"] = config.get("proximal-mu", 0.1)
    elif strategy_name in ["fedadam", "fedyogi", "fedadagrad"]:
        strategy_kwargs.update({
            "eta": config.get("server-learning-rate", 1.0),
            "eta_l": lr,
            "tau": config.get("tau", 1e-9),
        })
        if strategy_name != "fedadagrad":
            strategy_kwargs.update({
                "beta_1": config.get("beta-1", 0.9),
                "beta_2": config.get("beta-2", 0.99),
            })

    strategy = STRATEGIES[strategy_name](**strategy_kwargs)

    # 4. Dossier de sortie
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    strategy.set_save_path(save_path)
    def evaluate_callable(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return global_evaluate(server_round, arrays, dataset_name, n_cls, img_size, num_channels)
    # 5. Lancement de l'entraînement avec suivi d'émissions
    tracker = EmissionsTracker(
        project_name=strategy_name,
        output_dir=str(save_path),
        output_file="emission.csv",
        measure_power_secs=15
    )

    # ⚡ Snapshot Electricity Maps avant l'entraînement
    em_before = get_carbon_intensity_realtime(zone="BE")
    if em_before["realtime_carbon_intensity"]:
        print(f"⚡ Intensité carbone réseau : {em_before['realtime_carbon_intensity']} gCO2eq/kWh")
        print(f"🌿 Énergie fossile-free : {em_before.get('fossil_free_percentage', 'N/A')}%")
    
    import json
    with open(save_path / "grid_context.json", "w") as f:
        json.dump(em_before, f, indent=2)


    tracker.start()
    try:
        print(f"🚀 Starting Federated Learning: {strategy_name.upper()}")
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({
                "lr": lr,
                "save_path": str(save_path),
                "strategy": strategy_name,
                "run_id": "01"
            }),
            evaluate_config=ConfigRecord({
                "save_path": str(save_path),
            }),
            num_rounds=num_rounds,
            evaluate_fn=evaluate_callable,
        )
    finally:
        tracker.stop()
        #  Snapshot Electricity Maps après l'entraînement + comparaison
        em_after = get_carbon_intensity_realtime(zone="BE")
        
        if em_before["realtime_carbon_intensity"] and em_after["realtime_carbon_intensity"]:
            intensity_avg = (
                em_before["realtime_carbon_intensity"] + 
                em_after["realtime_carbon_intensity"]
            ) / 2
            
            # Lire l'énergie totale consommée depuis le CSV CodeCarbon
            emission_csv = save_path / "emission.csv"
            if emission_csv.exists():
                df_em = pd.read_csv(emission_csv)
                if "energy_consumed" in df_em.columns:
                    energy_kwh = pd.to_numeric(
                        df_em["energy_consumed"].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    ).sum()
                    
                    codecarbon_co2  = pd.to_numeric(
                        df_em["emissions"].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    ).sum()
                    
                    realtime_co2 = energy_kwh * (intensity_avg / 1000)  # kg CO2
                    
                    diff_pct = (
                        abs(realtime_co2 - codecarbon_co2) / codecarbon_co2 * 100
                        if codecarbon_co2 > 0 else 0
                    )
                    
                    print(f"\n📊 Comparaison émissions CO2 :")
                    print(f"   CodeCarbon  : {codecarbon_co2:.6f} kg CO2")
                    print(f"   ElectricityMaps : {realtime_co2:.6f} kg CO2")
                    print(f"   Écart       : {diff_pct:.1f}%")
                    
                    # Sauvegarder la comparaison
                    comparison = {
                        "strategy": strategy_name,
                        "zone": "BE",
                        "intensity_before_gco2_kwh": em_before["realtime_carbon_intensity"],
                        "intensity_after_gco2_kwh":  em_after["realtime_carbon_intensity"],
                        "intensity_avg_gco2_kwh":    intensity_avg,
                        "energy_consumed_kwh":        energy_kwh,
                        "codecarbon_kg_co2":          codecarbon_co2,
                        "electricitymaps_kg_co2":     realtime_co2,
                        "difference_percent":         diff_pct,
                        "datetime_start":             em_before.get("datetime"),
                        "datetime_end":               em_after.get("datetime"),
                    }
                    with open(save_path / "em_comparison.json", "w") as f:
                        json.dump(comparison, f, indent=2)
                    print(f"   💾 Sauvegardé dans : em_comparison.json")









        # --- CORRECTION DES DÉCIMALES ET FORMATS CSV ---
        print("\n🔄 Harmonisation des fichiers CSV pour Excel FR...")
        fichiers_csv = list(save_path.glob("*.csv"))
        
        for csv_path in fichiers_csv:
            try:
                # 1. On lit le fichier sans se soucier du séparateur
                # On utilise engine='python' pour détecter si c'est déjà du ';' ou du ','
                df_temp = pd.read_csv(csv_path, sep=None, engine='python')
                
                # 2. On s'assure que les colonnes numériques sont bien des nombres
                # Si Excel a déjà mis des virgules, on les remet en points pour que Pandas comprenne
                cols_numeriques = ['emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 
                                 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy', 'energy_consumed']
                
                for col in cols_numeriques:
                    if col in df_temp.columns:
                        # On force la conversion en numérique, au cas où c'est lu comme du texte
                        df_temp[col] = pd.to_numeric(df_temp[col].astype(str).str.replace(',', '.'), errors='coerce')

                # 3. On sauvegarde proprement pour Excel FR
                # POINT-VIRGULE pour les colonnes, VIRGULE pour les décimales
                df_temp.to_csv(csv_path, sep=';', decimal=',', index=False)
                print(f" ✅ Nettoyage terminé pour : {csv_path.name}")
            except Exception as e:
                print(f" ⚠️ Erreur sur {csv_path.name} : {e}")

        # Génération du graphique après conversion
        generate_emission_chart(save_path, strategy_name)

    # 6. Sauvegarde du modèle final
    print("\n💾 Sauvegarde du modèle final...")
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, save_path / "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord, dataset_name: str, n_cls:int, img_size:int, num_channels:int) -> MetricRecord:
    """Évaluation globale sur le dataset centralisé."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset(dataset_name = dataset_name, img_size = img_size, num_channels= num_channels)
    test_loss, test_acc, test_f1 = test(model, test_dataloader, device, n_cls)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss, "f1_score":test_f1})


def generate_emission_chart(output_path: Path, strategy_name: str):
    """Génère un graphique à partir du fichier converti (format FR)."""
    csv_file = output_path / "emission.csv"
    if csv_file.exists():
        try:
            # On lit avec le nouveau format (point-virgule et virgule)
            df = pd.read_csv(csv_file, sep=';', decimal=',')
            
            components = ['cpu_energy', 'gpu_energy', 'ram_energy']
            # Filtrer les colonnes présentes
            cols_to_plot = [c for c in components if c in df.columns]
            
            if not cols_to_plot:
                print(" ⚠️ Aucune colonne d'énergie trouvée.")
                return

            values = [df[c].sum() for c in cols_to_plot]
            labels = [c.replace('_energy', '').upper() for c in cols_to_plot]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=['#4CAF50', '#2196F3', '#FF9800'])
            plt.ylabel('Energy Consumed (kWh)')
            plt.title(f'Energy Consumption Breakdown - {strategy_name.upper()}')
            
            # Ajouter les valeurs au-dessus des barres
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', va='bottom', ha='center')

            plt.tight_layout()
            plt.savefig(output_path / f"{strategy_name.lower()}_energy_breakdown.png")
            plt.close()
            print(f" 📊 Graphique sauvegardé dans : {output_path}")
        except Exception as e:
            print(f" ⚠️ Erreur graphique : {e}")
