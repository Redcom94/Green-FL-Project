#!/bin/bash
# local_test_run.sh

if [ -z "$1" ]; then
    echo "Usage: ./launch.sh [nombre_de_noeuds]"
    exit 1
fi

NB_NODES=$1
NB_TOTAL_NODES=$((NB_NODES + 1))

echo "⏳ Soumission d'un job FL avec $NB_NODES + 1 nœuds..."
JOB_ID=$(sbatch --parsable -p debug --nodes=$NB_TOTAL_NODES script_slurm.slurm)

echo "Job lancé : $JOB_ID. En attente"
sleep 10
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%t")
    echo "État du job $JOB_ID : $STATE..."
    sleep 10 # Vérifie toutes les 10 secondes
done
scontrol wait_job $JOB_ID
