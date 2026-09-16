#!/bin/sh

set -e
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

prepare_output_dir() {
    requested_output=$1

    if [ -f "$requested_output/last_checkpoint" ] || [ -f "$requested_output/config.yaml" ]; then
        mkdir -p "$requested_output"
        echo "$requested_output"
        return
    fi

    mkdir -p "$requested_output/runs"
    run_id=$(date +%Y%m%d_%H%M%S)
    run_output="$requested_output/runs/$run_id"
    mkdir -p "$run_output"
    ln -sfn "$run_output" "$requested_output/latest"
    echo "$run_output"
}

if [ -z $config ]
then
    echo "No config file found! Run with bash run.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with bash run.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with bash run.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi

shift 3
gpus=$(resolve_gpu_count "$gpus")
output=$(prepare_output_dir "$output")

echo "Using ${gpus} GPU(s) for training"
echo "Training output: ${output}"

if [ "$#" -gt 0 ]; then
    $PYTHON_CMD train_net.py --config "$config" \
        --num-gpus "$gpus" \
        --dist-url "auto" \
        --resume \
        OUTPUT_DIR "$output" \
        "$@"
else
    $PYTHON_CMD train_net.py --config "$config" \
        --num-gpus "$gpus" \
        --dist-url "auto" \
        --resume \
        OUTPUT_DIR "$output"
fi

if [ ! -f "$output/model_final.pth" ]; then
    echo "Training finished without producing $output/model_final.pth; skipping evaluation." >&2
    exit 1
fi

sh eval.sh "$config" "$gpus" "$output" "$@"
