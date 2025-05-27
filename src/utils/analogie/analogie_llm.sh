#!/bin/bash
#SBATCH --job-name=analogie_llm
#SBATCH --output=analogie_llm.out

#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00


echo "job start: $(date)"

source ../../../../venv/bin/activate

python analogie_llm.py --k 0  | tee res_llama33_0.txt
#python analogie_llm.py --k 5  | tee res_llama33_5.txt

sleep 5
echo "job end: $(date)"
