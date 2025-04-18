#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH -o logs/output_%j.log

module load cuda/11.8  # or whatever your cluster needs
source activate gpu_bert_ft # conda environment with JAX

python base-BERT-ft.py --train_all
