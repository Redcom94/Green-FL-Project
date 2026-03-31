import flwr as fl
from optimizer_expert import GreenOptimizationExpert

# 1. Paramètres d'optimisation
STRATEGY_NAME = "FedProx"
MU_PROXIMAL = 0.1  # Ton levier d'optimisation principal

# 2. Configuration de la stratégie optimisée
strategy = fl.server.strategy.FedProx(
    proximal_mu=MU_PROXIMAL,
    fraction_fit=1.0,      # On entraîne tous les clients pour la stabilité
    min_fit_clients=2,
    min_available_clients=2,
)

# 3. Lancement du serveur
print(f"Lancement de la course avec {STRATEGY_NAME} (mu={MU_PROXIMAL})...")
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# 4. GÉNÉRATION AUTOMATIQUE DU RAPPORT (Ta partie Hakim)
# On récupère la précision du dernier round
final_acc = history.metrics_distributed["accuracy"][-1][1] if history.metrics_distributed else 0.0

expert = GreenOptimizationExpert(emissions_file="emissions.csv")
rapport = expert.analyze_and_report(final_acc, STRATEGY_NAME, MU_PROXIMAL)

print(f"\n--- Simulation terminée ---")
print(f"Le rapport de bonne conduite a été généré : {rapport}")