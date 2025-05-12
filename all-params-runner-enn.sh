#!/bin/sh
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH -o logs/output_%j.log

module load cuda/11.8  # or whatever your cluster needs
conda activate gpu_bert_ft # conda environment with JAX

echo "Running experiment with SLURM job ID: $SLURM_JOB_ID"

python experiment_sst-2-version6.py --train_all --suffix "$SLURM_JOB_ID" --save_params --save_params_path "outputs/job_$SLURM_JOB_ID/all-params_$SLURM_JOB_ID.npz" --priority "$1" --learning_rate 3e-6
