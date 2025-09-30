#!/bin/bash

# RelEdit vs AlphaEdit Comparison Script
# Runs both methods on the same dataset for direct comparison

# Set experiment parameters
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_FILE="Llama3-8B.json"
DATASET="mcf"
DATASET_SIZE=1000
NUM_EDITS=50

echo "========================================"
echo "Running RelEdit vs AlphaEdit Comparison"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Dataset Size: $DATASET_SIZE"
echo "Number of Edits: $NUM_EDITS"
echo "========================================"

# Run AlphaEdit
echo ""
echo "----------------------------------------"
echo "Running AlphaEdit..."
echo "----------------------------------------"
python3 -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=$MODEL_NAME \
    --hparams_fname=$HPARAMS_FILE \
    --ds_name=$DATASET \
    --dataset_size_limit=$DATASET_SIZE \
    --num_edits=$NUM_EDITS \
    --downstream_eval_steps=5 \
    --dir_name=AlphaEdit_comparison

# Run RelEdit
echo ""
echo "----------------------------------------"
echo "Running RelEdit..."
echo "----------------------------------------"
python3 -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=$MODEL_NAME \
    --hparams_fname=$HPARAMS_FILE \
    --ds_name=$DATASET \
    --dataset_size_limit=$DATASET_SIZE \
    --num_edits=$NUM_EDITS \
    --downstream_eval_steps=5 \
    --dir_name=RelEdit_comparison

echo "========================================"
echo "Comparison Completed"
echo "========================================"
echo "Results are in:"
echo "  - results/AlphaEdit_comparison/"
echo "  - results/RelEdit_comparison/"
echo ""
echo "To summarize results, use:"
echo "python summarize.py --dir_name=AlphaEdit_comparison --runs=run_<id>"
echo "python summarize.py --dir_name=RelEdit_comparison --runs=run_<id>"