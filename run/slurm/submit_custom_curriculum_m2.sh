#!/bin/bash
#SBATCH --job-name=UrbanFloodNet-m2-curriculum
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=/users/admarkowitz/UrbanFloodNet/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/UrbanFloodNet/logs/slurm_%j.err

# Resume Model_2 from a checkpoint with a custom horizon curriculum.
# Args:
#   $1 — checkpoint path (required; use "scratch" to train from scratch)
#   $2 — curriculum string, e.g. "8:2,16:4,24:4,32:4,48:4,64:4,128:4" (required)
#   Remaining args (any order):
#     <number>    — first numeric arg is LR (default 1e-3), second is clip norm
#     no-mirror   — skip mirroring to checkpoints/latest/
#     all-data    — train on train+val+test with no validation
#     keep-short  — retain short events at their max available horizon
#
# Examples:
#   sbatch slurm/submit_custom_curriculum.sh checkpoints/Model_2_best.pt "8:2,16:4,32:4,64:4,128:4"
#   sbatch slurm/submit_custom_curriculum.sh checkpoints/Model_2_best.pt "8:2,16:4,32:4,64:4" 1e-3 no-mirror
#   sbatch slurm/submit_custom_curriculum.sh ckpt.pt "192:6" keep-short
#   sbatch slurm/submit_custom_curriculum.sh ckpt.pt "192:6" 5e-4 keep-short no-mirror

CKPT="${1}"
CURRICULUM="${2}"

if [ -z "$CKPT" ] || [ -z "$CURRICULUM" ]; then
    echo "Usage: sbatch submit_custom_curriculum.sh <checkpoint> <curriculum> [lr] [flags...]"
    echo "  Flags: no-mirror, all-data, keep-short"
    exit 1
fi

if [ "$CKPT" = "scratch" ]; then
    RESUME_FLAG=""
else
    RESUME_FLAG="--resume ${CKPT}"
fi

# Parse remaining args: keyword flags + optional LR / clip-norm
LR="1e-3"
NO_MIRROR=""
ALL_DATA=""
CLIP_NORM=""
KEEP_SHORT=""
shift 2
for arg in "$@"; do
    case "$arg" in
        no-mirror)   NO_MIRROR="yes" ;;
        all-data)    ALL_DATA="yes" ;;
        keep-short)  KEEP_SHORT="yes" ;;
        *)
            # First numeric-looking arg is LR, second is clip norm
            if [ "$LR" = "1e-3" ] && echo "$arg" | grep -qE '^[0-9]'; then
                LR="$arg"
            else
                CLIP_NORM="$arg"
            fi
            ;;
    esac
done

mkdir -p /users/admarkowitz/UrbanFloodNet/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate urbanfloodnet

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/UrbanFloodNet

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     UrbanFloodNet Model_2 Custom Curriculum                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Started:     $(date)"
echo "Checkpoint:  $CKPT"
echo "Curriculum:  $CURRICULUM"
echo "LR:          $LR"
echo "Clip norm:   ${CLIP_NORM:-1.0}"
echo ""

PROJECT_DIR="/users/admarkowitz/UrbanFloodNet"
PYTHON="python3"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

ALLDATA_FLAG=""
if [ "$ALL_DATA" = "yes" ]; then
    ALLDATA_FLAG="--train-split all --no-val"
    echo "NOTE: Training on train+val combined, no validation"
fi

MIRROR_FLAG=""
if [ "$NO_MIRROR" = "yes" ]; then
    MIRROR_FLAG="--no-mirror-latest"
    echo "NOTE: checkpoints/latest/ will NOT be touched (--no-mirror-latest)"
fi

CLIP_FLAG=""
if [ -n "$CLIP_NORM" ]; then
    CLIP_FLAG="--clip-norm ${CLIP_NORM}"
    echo "NOTE: Gradient clip norm = $CLIP_NORM"
fi

KEEP_SHORT_FLAG=""
if [ "$KEEP_SHORT" = "yes" ]; then
    KEEP_SHORT_FLAG="--keep-short-events"
    echo "NOTE: Keeping short events (--keep-short-events)"
fi
echo ""

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"Model_2\" ${PYTHON} -u src/train.py \
    ${RESUME_FLAG} \
    --mixed-precision \
    --learning-rate ${LR} \
    --curriculum \"${CURRICULUM}\" \
    ${MIRROR_FLAG} \
    ${ALLDATA_FLAG} \
    ${CLIP_FLAG} \
    ${KEEP_SHORT_FLAG}"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "Training complete."
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "Training failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    Finished successfully                           ║"
else
    echo "║                    Failed                                          ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
