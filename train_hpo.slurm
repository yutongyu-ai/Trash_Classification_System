#!/bin/bash --login
#SBATCH --job-name=trashnet-hpo
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuL
#SBATCH --qos=gpu-fse-ugpgt01
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs main.py: a short Optuna hyperparameter search (25 trials x 6 epochs)
# over lr / weight_decay / batch_size, followed by a full 30-epoch training
# run using the best trial's hyperparameters. Saves the best checkpoint to
# checkpoints/best_resnet18_trashnet.pth (same path backend/inference.py
# already loads from) plus training curves / confusion matrix in outputs/.
#
# Submit from the project root with: sbatch train_hpo.slurm
# Adjust --partition/--gres if gpuL is unavailable or oversubscribed
# (alternatives on this cluster: gpuA40GB, gpuA, gpuH); --qos stays
# gpu-fse-ugpgt01 regardless of partition (that's this account's GPU QOS).

set -euo pipefail

module load miniforge/24.11.2
module load cuda/12.6.2
conda activate trashnet

cd "$SLURM_SUBMIT_DIR"

python main.py
