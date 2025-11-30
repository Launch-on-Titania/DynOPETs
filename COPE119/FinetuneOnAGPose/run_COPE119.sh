#!/bin/bash

# COPE119 Evaluation Script
# Usage: bash run_COPE119.sh <category_id> <scene_id>
# Example: bash run_COPE119.sh bottle 00

category_id=$1
scene_id=$2

# Configuration - Update these paths
CHECKPOINT="log/COPE119/ckpt/epoch_30.pt"
DATA_DIR="/path/to/COPE119_Data"
SAVE_PATH="results/${category_id}_${scene_id}"

# Validate inputs
if [ -z "$category_id" ] || [ -z "$scene_id" ]; then
    echo "Usage: bash run_COPE119.sh <category_id> <scene_id>"
    echo "Example: bash run_COPE119.sh bottle 00"
    echo ""
    echo "Available categories: bottle, bowl, camera, can, laptop, mug"
    exit 1
fi

# Run evaluation
python eval_on_COPE119.py \
    --checkpoint ${CHECKPOINT} \
    --data_dir ${DATA_DIR} \
    --category_id ${category_id} \
    --scene_id ${scene_id} \
    --save_path ${SAVE_PATH} \
    --gpus 0

echo ""
echo "Evaluation completed!"
echo "Results saved to: ${SAVE_PATH}"