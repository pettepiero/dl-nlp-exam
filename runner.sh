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

echo "Running experiment with SLURM job ID: $SLURM_JOB_ID"

python base-BERT-ft.py --suffix "$SLURM_JOB_ID" --save_params --save_params_path "outputs/job_$SLURM_JOB_ID/params_$SLURM_JOB_ID.npz" --learning_rate 1e-4
