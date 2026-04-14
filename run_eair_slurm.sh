#!/bin/bash
#SBATCH --job-name=eair_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --nodelist=devbox

cd /mnt/raid5/leejg/code/Reasoning-Editing/EAIR-ACL2026
source .venv/bin/activate
mkdir -p logs output

echo "=== EAIR Experiment ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
cat config.json
echo "======================="

uv run python main.py

echo "=== Finished: $(date) ==="
