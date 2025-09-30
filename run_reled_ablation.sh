#!/bin/bash

# RelEdit Ablation Study Script
# Tests different alpha values to analyze the effect of KG-based projection

# Set base parameters
ALG_NAME="RelEdit"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_FILE="Llama3-8B.json"
DATASET="mcf"
DATASET_SIZE=500
NUM_EDITS=50
DIR_NAME="RelEdit_ablation"

# Alpha values to test: 0.0 (pure null-space), 0.25, 0.5, 0.75, 1.0
ALPHAS=(0.0 0.25 0.5 0.75 1.0)

echo "========================================"
echo "Running RelEdit Ablation Study"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Testing alpha values: ${ALPHAS[*]}"
echo "========================================"

for alpha in "${ALPHAS[@]}"
do
    echo ""
    echo "----------------------------------------"
    echo "Running with alpha=$alpha"
    echo "----------------------------------------"

    # Create temporary hparams file with specific alpha
    TEMP_HPARAMS="Llama3-8B_alpha${alpha}.json"

    # Copy base hparams and modify alpha value
    jq ".alpha = $alpha" "hparams/RelEdit/$HPARAMS_FILE" > "hparams/RelEdit/$TEMP_HPARAMS"

    # Run experiment
    python3 -m experiments.evaluate \
        --alg_name=$ALG_NAME \
        --model_name=$MODEL_NAME \
        --hparams_fname=$TEMP_HPARAMS \
        --ds_name=$DATASET \
        --dataset_size_limit=$DATASET_SIZE \
        --num_edits=$NUM_EDITS \
        --downstream_eval_steps=5 \
        --dir_name="${DIR_NAME}_alpha${alpha}"

    echo "Completed alpha=$alpha"
done

echo "========================================"
echo "RelEdit Ablation Study Completed"
echo "========================================"
echo "To compare results, use:"
echo "python summarize.py --dir_name=RelEdit_ablation --runs=run_<id1>,run_<id2>,..."