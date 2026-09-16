GPUS=4
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 跑 2 个 SCENE LD50K 训练模型
sh run.sh configs/vitl_336_LD50K_v2.yaml "$GPUS" "$PROJECT_ROOT/output_vitl_336_LD50K_v2/"
sh run.sh configs/vitb_384_LD50K_v2.yaml "$GPUS" "$PROJECT_ROOT/output_vitb_384_LD50K_v2/"
