# =================================
# SCENE LD50K cross-dataset evaluation
# evaluate the LD50K-trained models on four GSNet test datasets
# =================================

# use up to 4 GPUs; fallback to available GPU count
GPUS=1
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# =================================
# ViT-B LD50K model
# =================================
bash eval.sh \
  configs/vitb_384_LD50K_v2.yaml \
  "$GPUS" \
  "$PROJECT_ROOT/output_vitb_384_LD50K_v2/latest" \
  MODEL.WEIGHTS "$PROJECT_ROOT/output_vitb_384_LD50K_v2/latest/model_final.pth"

# =================================
# ViT-L LD50K model
# =================================

# bash eval.sh \
#   configs/vitl_336_LD50K_v2.yaml \
#   "$GPUS" \
#   "$PROJECT_ROOT/output_vitl_336_LD50K_v2/latest" \
#   MODEL.WEIGHTS "$PROJECT_ROOT/output_vitl_336_LD50K_v2/latest/model_final.pth"
