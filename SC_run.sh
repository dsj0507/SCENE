# =================================
# SCENE LD50K training entry
# outputs are stored inside this project directory
# =================================

# use up to 4 GPUs; fallback to available GPU count
export CUDA_VISIBLE_DEVICES=0,1
export SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT:-./SegEarth-OV-3}"
GPUS=$(printf "%s" "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# =================================
# ViT-B LD50K model
# =================================
sh run.sh \
  configs/vitb_384_LD50K_v2.yaml \
  "$GPUS" \
  "$PROJECT_ROOT/output_vitb_384_LD50K_v2"

# =================================
# ViT-L LD50K model
# =================================
# sh run.sh \
#   configs/vitl_336_LD50K_v2.yaml \
#   "$GPUS" \
#   "$PROJECT_ROOT/output_vitl_336_LD50K_v2"
