#!/bin/bash

# 1. Configuration simulée
PROJECT_DIR=$(pwd)
ENV_NAME="Federated"
STRATEGY="fedavg"
NB_CLIENTS=${1:-2}  # On teste avec 2 clients pour commencer

# Créer le dossier de sortie
mkdir -p outputs

echo "--- 🛠️ DÉMARRAGE DU TEST LOCAL ---"

# 2. Simulation de l'activation de l'environnement
source "$ENV_NAME/bin/activate"
# 3. Simulation deaster
export MASTER_IP="127.0.0.1"

# 4. Lancement du SERVEUR (en arrière-plan)
echo "🚀 [SERVEUR] Lancement sur $MASTER_IP..."
flower-superlink --insecure > outputs/server_log.txt 2>&1 &
SERVER_PID=$!

# Attente pour l'initialisation
sleep 5

# 5. Lancement des CLIENTS (en arrière-plan)
echo "Initialisation des clients"
for i in $(seq 1 $NB_CLIENTS)
do
    # Calcul d'un port unique pour chaque client (ex: 9094, 9095, 9096...)
    CLIENT_PORT=$((9094 + i))
    
    # Correction de l'ID de partition (i-1 car seq commence à 1 et partition-id à 0)
    PART_ID=$((i - 1))

    echo "👥 [CLIENT $i] Connexion au serveur sur le port local $CLIENT_PORT..."

    # Commande correcte (en utilisant flwr clientapp qui gère tout)
    flower-supernode\
        --insecure \
        --superlink $MASTER_IP:9092 \
        --clientappio-api-address 127.0.0.1:$CLIENT_PORT \
        --node-config "partition-id=$PART_ID num-partitions=$NB_CLIENTS" \
        > outputs/client_${i}_log.txt 2>&1 &
done
flwr run . local-deployment --stream

echo "🧹 Le serveur a fini. Nettoyage des clients restants..."

# On tue tous les processus lancés par ce script (SuperNodes et ClientApps)
# 'jobs -p' liste les IDs des processus lancés en arrière-plan
kill $(jobs -p) 2>/dev/null

echo "--- ✅ TEST TERMINÉ ---"
echo "Vérifie outputs/server_log.txt pour voir si l'agrégation a réussi."
