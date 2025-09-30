#!/bin/bash

# RelEdit Demo Script
# This script runs RelEdit on a small subset of CounterFact dataset

# Set experiment parameters
ALG_NAME="RelEdit"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_FILE="Llama3-8B.json"
DATASET="mcf"
DATASET_SIZE=100
NUM_EDITS=10
DIR_NAME="RelEdit"

echo "========================================"
echo "Running RelEdit Demo"
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
echo "RelEdit Demo Completed"
echo "========================================"