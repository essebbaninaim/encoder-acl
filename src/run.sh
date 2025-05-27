#!/bin/bash
#SBATCH --job-name=both
#SBATCH --output=both_%j.out

#SBATCH --nodes=1
#SBATCH --partition=quad_rtx_8000
#SBATCH --cpus-per-task=8

#SBATCH --ntasks=2
#SBATCH --gres=gpu:4

if [ -z "$1" ]; then
    echo "Error: No argument provided for the script."
    echo "Usage: sbatch both.sh chemin_vers_config"
    exit 1
fi

echo "job start: $(date)"

echo "Running both.py with argument: $1"
source ../../../../venv_llama2/bin/activate
python run.py $1
deactivate

sleep 5
echo "job end: $(date)"
