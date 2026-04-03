#!/bin/bash
#SBATCH --job-name=FL_Distributed
#SBATCH --nodes=3                # 1 Serveur + 2 Clients
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=debug
#SBATCH --output=outputs/multi_node_%j.log
# --- 1. PRÉPARATION ---
ABS_PATH="/home_nfs/deleaul/Green-FL-Project"
mkdir -p $ABS_PATH/outputs
source /home_nfs/deleaul/FederatedVenv/bin/activate

# Nettoyage préventif : on tue les processus Flower qui pourraient traîner
srun --nodes=$SLURM_JOB_NUM_NODES pkill -u $USER -f flower-superlink || true
srun --nodes=$SLURM_JOB_NUM_NODES pkill -u $USER -f flower-supernode || true

# Récupération des IPs
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_NODE=${NODES[0]}
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w $MASTER_NODE hostname -I | awk '{print $1}')

echo "--- 🌐 CONFIGURATION ---"
echo "MASTER: $MASTER_NODE à l'adresse $MASTER_IP"

# --- 2. LANCER LE SERVEUR (SUPERLINK) ---
echo "📡 Démarrage du SuperLink..."
srun --nodes=1 --ntasks=1 -w $MASTER_NODE \
    flower-superlink --insecure > $ABS_PATH/outputs/server_log.txt 2>&1 &

sleep 10

# --- 3. LANCER LES CLIENTS (SUPERNODES) ---
CLIENT_NODES=(${NODES[@]:1})
NUM_CLIENTS=${#CLIENT_NODES[@]}

for i in "${!CLIENT_NODES[@]}"
do
    NODE=${CLIENT_NODES[$i]}
    PART_ID=$i
    echo "👥 Client $PART_ID sur $NODE"

    srun --nodes=1 --ntasks=1 -w $NODE \
        flower-supernode \
        --insecure \
        --superlink $MASTER_IP:9092 \
        --node-config "partition-id=$PART_ID num-partitions=$NUM_CLIENTS" \
        > $ABS_PATH/outputs/client_${PART_ID}_log.txt 2>&1 &
done

sleep 10

# --- 4. LANCER LA LOGIQUE (SERVERAPP) ---
echo "🚀 Lancement de la ServerApp..."
cd $ABS_PATH
# On pointe vers l'adresse Fleet API du SuperLink
flwr run . local-deployment --stream

# --- 5. NETTOYAGE FINAL ---
echo "✅ Terminé. Nettoyage..."
scancel $SLURM_JOB_ID
