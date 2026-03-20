#!/bin/bash
#SBATCH --job-name=UrbanFloodNet
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/UrbanFloodNet/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/UrbanFloodNet/logs/slurm_%j.err

# Usage:
#   sbatch slurm/submit_inference_slurm.sh
#       # default: infer both models from checkpoints/latest, select by val_loss
#
#   sbatch slurm/submit_inference_slurm.sh --select best
#       # use Model_X_best.pt from checkpoints/latest
#
#   sbatch slurm/submit_inference_slurm.sh --model1-dir checkpoints/Model_1_20260307_123456 \
#                                           --model2-dir checkpoints/Model_2_20260308_113327
#       # use specific run dirs; pick best checkpoint from each
#
#   sbatch slurm/submit_inference_slurm.sh \
#       --model2-ckpt checkpoints/Model_2_20260313_201543/Model_2_epoch_048.pt
#       # use an exact checkpoint file for Model_2; Model_1 defaults to checkpoints/latest
#
#   sbatch slurm/submit_inference_slurm.sh \
#       --model1-ckpt checkpoints/Model_1_.../Model_1_epoch_032.pt \
#       --model2-ckpt checkpoints/Model_2_.../Model_2_epoch_048.pt
#       # exact checkpoint files for both models

# Ensure log directory exists (must happen before SLURM tries to write output files,
# so run `mkdir -p /users/admarkowitz/UrbanFloodNet/logs` once before your first sbatch)
mkdir -p /users/admarkowitz/UrbanFloodNet/logs

# Activate conda environment
source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate urbanfloodnet

# Configure CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Move to project root
cd /users/admarkowitz/UrbanFloodNet

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo ""

bash run/pipeline_inference.sh auto all "$@"

echo ""
echo "Finished: $(date)"
