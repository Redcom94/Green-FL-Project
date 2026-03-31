import pandas as pd
import os

class GreenOptimizationExpert:
    def __init__(self, emissions_file="emissions.csv"):
        self.emissions_file = emissions_file

    def analyze_and_report(self, final_accuracy, strategy_name, mu_val=None):
        if not os.path.exists(self.emissions_file):
            return "Erreur : Fichier d'émissions introuvable."

        # Lecture des données CodeCarbon
        df = pd.read_csv(self.emissions_file)
        last_run = df.iloc[-1]
        
        energy_kwh = last_run['energy_consumed']
        co2_g = last_run['emissions'] * 1000 # Conversion en grammes
        cpu_watt = last_run['cpu_power']
        ram_watt = last_run['ram_power']

        report_name = f"RAPPORT_BONNE_CONDUITE_{strategy_name}.txt"
        
        with open(report_name, "w", encoding="utf-8") as f:
            f.write(f"=== ANALYSE D'OPTIMISATION GREEN-FL ({strategy_name}) ===\n")
            f.write(f"Précision finale : {final_accuracy:.2%}\n")
            f.write(f"Consommation totale : {energy_kwh:.6f} kWh\n")
            f.write(f"Empreinte Carbone : {co2_g:.4f} g CO2\n\n")
            
            f.write("--- RÈGLES DE BONNE CONDUITE APPLIQUÉES ---\n")
            
            # Règle 1 : Efficacité énergétique
            if energy_kwh > 0.001:
                f.write("[CONSEIL] Consommation élevée détectée. Réduisez le nombre d'époques locales (E) pour soulager le CPU.\n")
            else:
                f.write("[BRAVO] Sobriété numérique respectée. Votre modèle est léger.\n")

            # Règle 2 : Analyse Hardware (Barre orange de la RAM dans ta vidéo)
            if ram_watt > cpu_watt:
                f.write(f"[ALERTE] La RAM ({ram_watt:.1f}W) consomme plus que le CPU. Évitez de charger des jeux de données trop lourds en mémoire d'un coup.\n")
            
            # Règle 3 : Optimisation FedProx
            if strategy_name == "FedProx":
                if final_accuracy < 0.70:
                    f.write(f"[AJUSTEMENT] Précision faible avec FedProx. Essayez de diminuer mu (actuellement {mu_val}) pour laisser plus de liberté aux clients.\n")
                else:
                    f.write(f"[SUCCÈS] Le paramètre mu ({mu_val}) stabilise bien l'apprentissage sans surconsommation.\n")
            
            # Règle 4 : Arrêt Précoce (Early Stopping)
            f.write("[INFO] Règle d'or : Si la précision ne gagne plus 1% après 3 rounds, stoppez l'expérience pour économiser du CO2.\n")

        return report_name