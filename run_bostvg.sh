#!/bin/bash
set -e

MODE=${MODE:-train}
GPUS=${GPUS:-32}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${CONFIG:-experiments/omnistvg.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-model_output}
MODEL_WEIGHT=${MODEL_WEIGHT:-${OUTPUT_DIR}/model_final.pth}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ "$MODE" = "train" ]; then
    SCRIPT=scripts/train_net.py
    EXTRA_OPTS=(OUTPUT_DIR "$OUTPUT_DIR" TENSORBOARD_DIR "${OUTPUT_DIR}/tensorboard")
elif [ "$MODE" = "test" ]; then
    SCRIPT=scripts/test_net.py
    EXTRA_OPTS=(OUTPUT_DIR "$OUTPUT_DIR" MODEL.WEIGHT "$MODEL_WEIGHT")
else
    echo "Unsupported MODE: ${MODE}. Use MODE=train or MODE=test."
    exit 1
fi

if [ "$GPUS" -gt 1 ]; then
    python3 -m torch.distributed.launch \
        --nproc_per_node "$GPUS" \
        --master_port "$MASTER_PORT" \
        "$SCRIPT" \
        --config-file "$CONFIG" \
        "${EXTRA_OPTS[@]}"
else
    python3 "$SCRIPT" \
        --config-file "$CONFIG" \
        "${EXTRA_OPTS[@]}"
fi
