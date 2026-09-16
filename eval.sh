#!/bin/sh

export SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT:-./SegEarth-OV-3}"
config=$1
gpus=$2
output=$3
PYTHON_CMD=${PYTHON_CMD:-"conda run -n sg_ovrss_py39 python"}

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            echo "$count"
            return
        fi
    fi
    echo 1
}

resolve_gpu_count() {
    requested=$1
    available=$(detect_gpu_count)

    if [ -z "$requested" ] || [ "$requested" -le 0 ] 2>/dev/null; then
        requested=4
    fi

    if [ "$requested" -gt 4 ] 2>/dev/null; then
        requested=4
    fi

    if [ "$available" -lt "$requested" ] 2>/dev/null; then
        echo "$available"
    else
        echo "$requested"
    fi
}

if [ -z "$config" ]; then
    echo 'Run with: bash eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]'
    exit 1
fi

if [ -z "$gpus" ]; then
    echo 'Run with: bash eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]'
    exit 1
fi

if [ -z "$output" ]; then
    echo 'Run with: bash eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]'
    exit 1
fi

shift 3
opts="$*"
gpus=$(resolve_gpu_count "$gpus")

echo "Using ${gpus} GPU(s) for evaluation"

run_eval() {
    dataset_name=$1
    class_json=$2
    tag=$3

    if [ -n "$opts" ]; then
        # shellcheck disable=SC2086
        $PYTHON_CMD train_net.py --config-file "$config" \
            --num-gpus "$gpus" \
            --dist-url "auto" \
            --eval-only \
            OUTPUT_DIR "$output/eval-${tag}_out" \
            MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$class_json" \
            DATASETS.TEST "(\"${dataset_name}\",)" \
            TEST.SLIDING_WINDOW False \
            MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
            MODEL.WEIGHTS "$output/model_final.pth" \
            $opts
    else
        $PYTHON_CMD train_net.py --config-file "$config" \
            --num-gpus "$gpus" \
            --dist-url "auto" \
            --eval-only \
            OUTPUT_DIR "$output/eval-${tag}_out" \
            MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$class_json" \
            DATASETS.TEST "(\"${dataset_name}\",)" \
            TEST.SLIDING_WINDOW False \
            MODEL.SEM_SEG_HEAD.POOLING_SIZES "[2,2]" \
            MODEL.WEIGHTS "$output/model_final.pth"
    fi
}

run_eval "Potsdam_all_sem_seg" "datasets/potsdam.json" "Potsdam"
run_eval "FloodNet" "datasets/floodnet.json" "FloodNet"
run_eval "FLAIR_test" "datasets/flair.json" "FLAIR"
run_eval "FAST_val" "datasets/fast.json" "FAST"

for tag in Potsdam FloodNet FLAIR FAST; do
    if [ -f "$output/eval-${tag}_out/log.txt" ]; then
        grep copypaste "$output/eval-${tag}_out/log.txt"
    fi
done
