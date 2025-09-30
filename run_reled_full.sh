#!/bin/bash

# RelEdit Full Experiment Script
# This script runs RelEdit on the full CounterFact dataset

# Set experiment parameters
ALG_NAME="RelEdit"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_FILE="Llama3-8B.json"
DATASET="mcf"
DATASET_SIZE=2000
NUM_EDITS=100
DIR_NAME="RelEdit"

echo "========================================"
echo "Running RelEdit Full Experiment"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Dataset Size: $DATASET_SIZE"
echo "Number of Edits: $NUM_EDITS"
echo "========================================"

# Run the experiment
python3 -m experiments.evaluate \
    --alg_name=$ALG_NAME \
    --model_name=$MODEL_NAME \
    --hparams_fname=$HPARAMS_FILE \
    --ds_name=$DATASET \
    --dataset_size_limit=$DATASET_SIZE \
    --num_edits=$NUM_EDITS \
    --downstream_eval_steps=5 \
    --dir_name=$DIR_NAME

echo "========================================"
echo "RelEdit Full Experiment Completed"
echo "========================================"