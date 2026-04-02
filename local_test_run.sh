#!/bin/bash
# local_test_run.sh

if [ -z "$1" ]; then
    echo "Usage: ./launch.sh [nombre_de_noeuds]"
    exit 1
fi

NB_NODES=$1

echo "⏳ Soumission d'un job FL avec $NB_NODES nœuds..."
sbatch --nodes=$NB_NODES run_fl.slurm