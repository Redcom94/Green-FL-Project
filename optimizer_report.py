import pandas as pd

def generate_green_report(csv_file, accuracy):
    df = pd.read_csv(csv_file)
    total_energy = df['energy_consumed'].sum()
    
    with open("CONSEILS_OPTIMISATION.txt", "w") as f:
        f.write("=== RÈGLES DE BONNE CONDUITE ÉCO-IA ===\n\n")
        
        # Règle 1: Analyse de la consommation
        f.write(f"- Consommation totale : {total_energy:.4f} kWh\n")
        if total_energy > 0.005: # Seuil à ajuster
            f.write("-> CONSEIL : Votre consommation est élevée. Réduisez le nombre de 'local epochs'.\n")
        
        # Règle 2: Équilibre Précision/Énergie
        f.write(f"- Précision atteinte : {accuracy:.2%}\n")
        if accuracy > 0.85:
            f.write("-> SUCCÈS : Bonne précision. Vous pouvez essayer de réduire le nombre de Rounds pour économiser du CO2.\n")
            
        # Règle 3: Stabilité (Lien avec FedProx)
        f.write("- Stabilité : Si les courbes oscillent, augmentez le paramètre mu (u) de FedProx.\n")