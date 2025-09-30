# AlphaEdit & RelEdit - Complete Tutorial

A comprehensive guide for running knowledge editing experiments on Language Models using AlphaEdit and RelEdit methods.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Running Locally](#running-locally)
6. [Running on HPC](#running-on-hpc)
7. [Experiment Configurations](#experiment-configurations)
8. [Results Analysis](#results-analysis)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## 🎯 Overview

### AlphaEdit (ICLR 2025 Outstanding Paper)

**AlphaEdit** is a null-space constrained knowledge editing method for language models that minimizes disruption to preserved knowledge by projecting parameter perturbations onto the null space of key matrices.

**Key Features:**
- **Null-space Projection**: Ensures edits don't affect preserved knowledge
- **Mathematical Guarantees**: Maintains invariance of hidden representations
- **Sequential Editing**: Optimized for multiple rounds of editing
- **Efficient**: No gradient-based optimization required

### RelEdit (Extension)

**RelEdit** extends AlphaEdit by incorporating **relation-aware knowledge graph sampling** to reduce knowledge fragmentation and enable coherent integration of related facts.

**Key Innovations:**
- **KG Sampling**: Samples related entities from Wikidata knowledge graph
- **Relational Projection**: P' = P_null + α·P_rel
- **Controllable Editing**: α parameter balances local/global edit scope
- **Reduced Fragmentation**: Considers consistency across related facts

**RelEdit Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_paths` | 10 | [1, 50] | Number of KG paths to sample |
| `max_path_length` | 3 | [1, 5] | Maximum path length in KG traversal |
| `alpha` | 0.5 | [0, 1] | Projection combination weight |
| `use_kg_sampling` | true | bool | Enable/disable KG sampling |

- **α = 0**: Pure AlphaEdit (null-space only)
- **α = 1**: Maximum relational influence
- **α = 0.5**: Balanced combination (recommended)

---

## 📁 Project Structure

```
AlphaEdit-main/
│
├── 📄 README.md                    # Original AlphaEdit documentation
├── 📄 TUTORIAL.md                  # This comprehensive tutorial
├── 📄 RELED_IMPLEMENTATION.md      # RelEdit implementation details
├── 📄 CODE_REVIEW_REPORT.md        # Code review report
├── 📄 globals.yml                  # Global path configuration
│
├── 📁 AlphaEdit/                   # AlphaEdit core implementation
│   ├── __init__.py
│   ├── AlphaEdit_hparams.py       # Hyperparameters dataclass
│   ├── AlphaEdit_main.py          # Main editing function
│   └── compute_z.py               # Target value computation
│
├── 📁 RelEdit/                     # RelEdit extension
│   ├── __init__.py
│   ├── RelEdit_hparams.py         # RelEdit hyperparameters
│   ├── RelEdit_main.py            # Main RelEdit function
│   ├── kg_sampler.py              # Wikidata KG sampler
│   └── compute_projection.py      # Projection computation
│
├── 📁 baselines/                   # Baseline methods
│   ├── ft/                        # Fine-tuning
│   └── mend/                      # MEND
│
├── 📁 rome/                        # ROME method
├── 📁 memit/                       # MEMIT method
├── 📁 nse/                         # NSE method
│
├── 📁 dsets/                       # Dataset loaders
│   ├── counterfact.py             # CounterFact dataset
│   ├── zsre.py                    # Zero-shot RE dataset
│   ├── mcf.py                     # Multi-CounterFact
│   └── mquake.py                  # MQuAKE dataset
│
├── 📁 experiments/                 # Experiment scripts
│   ├── evaluate.py                # Main evaluation script
│   ├── summarize.py               # Results summarization
│   └── py/                        # Helper scripts
│
├── 📁 hparams/                     # Hyperparameter configs
│   ├── AlphaEdit/
│   │   ├── Llama3-8B.json
│   │   ├── gpt2-xl.json
│   │   └── EleutherAI_gpt-j-6B.json
│   ├── RelEdit/
│   │   ├── Llama3-8B.json
│   │   ├── gpt2-xl.json
│   │   ├── EleutherAI_gpt-j-6B.json
│   │   └── phi-1.5.json
│   └── [ROME, MEMIT, NSE, FT, MEND]/
│
├── 📁 glue_eval/                   # GLUE benchmark evaluation
│   ├── cola_eval.py
│   ├── sentiment_analysis_eval.py
│   ├── mmlu_eval.py
│   └── glue_eval.py
│
├── 📁 util/                        # Utility functions
│   ├── globals.py
│   ├── nethook.py                 # Model hook utilities
│   ├── generate.py
│   └── hparams.py
│
├── 📁 data/                        # Data directory
│   ├── stats/                     # Pre-computed covariance matrices
│   └── kg_cache/                  # KG query cache
│
├── 📁 results/                     # Experiment results
│
└── 🔧 Run Scripts
    ├── run_reled_demo.sh          # Quick demo
    ├── run_reled_full.sh          # Full evaluation
    ├── run_reled_ablation.sh      # Ablation study
    └── run_reled_comparison.sh    # AlphaEdit vs RelEdit
```

### Core Files Overview

| File | Purpose | Importance |
|------|---------|-----------|
| `experiments/evaluate.py` | Main evaluation entry point | ⭐⭐⭐⭐⭐ |
| `AlphaEdit/AlphaEdit_main.py` | AlphaEdit core algorithm | ⭐⭐⭐⭐⭐ |
| `RelEdit/RelEdit_main.py` | RelEdit core algorithm | ⭐⭐⭐⭐⭐ |
| `RelEdit/kg_sampler.py` | Wikidata KG sampler | ⭐⭐⭐⭐ |
| `RelEdit/compute_projection.py` | Projection computation | ⭐⭐⭐⭐ |
| `dsets/counterfact.py` | CounterFact dataset loader | ⭐⭐⭐ |
| `util/nethook.py` | Model parameter access | ⭐⭐⭐ |

---

## 🔧 Installation

### Hardware Requirements

| Environment | Minimum | Recommended |
|-------------|---------|-------------|
| **Local** | 1x NVIDIA GPU (24GB VRAM) | 1x A40/A100 (48GB) |
| **HPC** | 1 node, 1 GPU (24GB) | 1 node, 1 GPU (48GB) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 50GB free space | 100GB+ |

### Software Dependencies

```bash
# Python version
python >= 3.8

# Core dependencies
torch==2.6.0
transformers==4.51.3
datasets==2.21.0

# Scientific computing
numpy>=1.24.0
scipy==1.15.2
scikit-learn==1.6.1

# Utilities
einops==0.8.1
higher==0.2.1
hydra-core==1.3.2
matplotlib==3.10.3
spacy==3.4.1
nltk==3.9.1

# RelEdit specific
requests>=2.31.0  # For Wikidata API
```

### Installation Steps

#### Option 1: Using pip

```bash
# Install PyTorch (choose CUDA version based on your system)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and Datasets
pip install transformers==4.51.3 datasets==2.21.0

# Install other dependencies
pip install einops higher hydra-core matplotlib spacy scipy scikit-learn nltk requests
```

#### Option 2: Using conda

```bash
# Create environment
conda create -n alphaedit python=3.10 -y
conda activate alphaedit

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers==4.51.3 datasets==2.21.0 einops higher hydra-core requests
pip install matplotlib spacy scipy scikit-learn nltk
```

#### Option 3: Using requirements.txt

```bash
pip install -r requirements.txt
```

### Download Pre-computed Covariance Matrix

**Required**: Download the pre-computed covariance matrix for Llama3-8B-Instruct:

```bash
# Create directory
mkdir -p data/stats

# Download from Google Drive
# URL: https://drive.google.com/file/d/1rAeGBJccEaZYFpPMlD5tb5TNjkaUqwq6/view

# Extract to data/stats/
unzip <downloaded_file>.zip -d data/stats/
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/AlphaEdit.git
cd AlphaEdit
```

### 2. Install Dependencies

```bash
# Using conda (recommended)
conda create -n alphaedit python=3.10
conda activate alphaedit
pip install torch==2.6.0 transformers==4.51.3 datasets==2.21.0 requests
pip install einops higher hydra-core matplotlib spacy scipy scikit-learn nltk
```

### 3. Download Required Data

```bash
# Download covariance matrix (see Installation section)
mkdir -p data/stats
# Place downloaded files in data/stats/
```

### 4. Run Quick Demo

```bash
# RelEdit demo (~30 minutes on A40)
bash run_reled_demo.sh

# Or run directly
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=100 \
    --num_edits=10 \
    --downstream_eval_steps=5 \
    --dir_name=RelEdit_demo
```

### 5. View Results

```bash
# Check results directory
ls results/RelEdit_demo/

# Summarize results
python experiments/summarize.py --dir_name=RelEdit_demo --runs=run_<your_id>
```

---

## 💻 Running Locally

### Step-by-Step Guide

#### 1. Setup Environment

```bash
# Clone and enter directory
git clone https://github.com/your-username/AlphaEdit.git
cd AlphaEdit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch==2.6.0 transformers==4.51.3 datasets==2.21.0 requests
pip install einops higher hydra-core matplotlib spacy scipy scikit-learn nltk
```

#### 2. Configure Paths

Edit `globals.yml`:

```yaml
data_dir: ./data
stats_dir: ./data/stats
results_dir: ./results
```

#### 3. Run Syntax Tests (Optional)

```bash
python test_reled_syntax.py
```

Expected output:
```
Testing RelEdit Python files syntax...
==================================================
[OK] RelEdit/__init__.py: Valid syntax
[OK] RelEdit/RelEdit_hparams.py: Valid syntax
...
All structure tests passed!
```

#### 4. Run Experiments

**Quick Demo** (~30 minutes):
```bash
bash run_reled_demo.sh
```

**Full Evaluation** (~8 hours):
```bash
bash run_reled_full.sh
```

**Ablation Study** (~10 hours):
```bash
bash run_reled_ablation.sh
```

**Method Comparison** (~6 hours):
```bash
bash run_reled_comparison.sh
```

#### 5. Custom Experiments

```bash
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=1000 \
    --num_edits=50 \
    --downstream_eval_steps=5 \
    --dir_name=my_experiment
```

### Available Models

| Model | HuggingFace ID | Config File | VRAM |
|-------|----------------|-------------|------|
| Llama3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` | `Llama3-8B.json` | 24GB |
| GPT2-XL | `gpt2-xl` | `gpt2-xl.json` | 8GB |
| GPT-J-6B | `EleutherAI/gpt-j-6B` | `EleutherAI_gpt-j-6B.json` | 16GB |
| Phi-1.5 | `microsoft/phi-1.5` | `phi-1.5.json` | 6GB |

### Available Datasets

| Dataset | Full Name | Size | Type | Description |
|---------|-----------|------|------|-------------|
| **mcf** | Multi-CounterFact | ~21K | Factual editing | Multi-hop counterfactual editing |
| **cf** | CounterFact | ~21K | Factual editing | Single-hop counterfactual editing |
| **zsre** | Zero-shot RE | ~10K | Relation extraction | Zero-shot relation extraction |
| **mquake** | MQuAKE | ~3K | Multi-hop QA | Multi-hop question answering |

---

## 🖥️ Running on HPC

### Complete HPC Setup Guide

#### 1. Transfer Project to HPC

```bash
# On local machine - pack project
tar -czf alphaedit.tar.gz AlphaEdit/

# Transfer to HPC
scp alphaedit.tar.gz username@hpc-server:/path/to/workspace/

# On HPC - extract
ssh username@hpc-server
cd /path/to/workspace/
tar -xzf alphaedit.tar.gz
cd AlphaEdit
```

#### 2. Setup Environment on HPC

**Method 1: Using Module System**

```bash
# Load modules
module load python/3.10
module load cuda/11.8
module load gcc/11.2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --no-cache-dir torch==2.6.0 transformers==4.51.3 datasets==2.21.0
pip install --no-cache-dir einops higher hydra-core requests matplotlib spacy scipy scikit-learn nltk
```

**Method 2: Using Conda**

```bash
# Load conda
module load anaconda3

# Create environment
conda create -n alphaedit python=3.10 -y
conda activate alphaedit

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers==4.51.3 datasets==2.21.0 einops higher hydra-core requests
```

#### 3. Prepare Data

```bash
# Create directories
mkdir -p data/stats data/kg_cache

# Transfer covariance matrix (if not already done)
# Option 1: SCP from local
scp -r data/stats/* username@hpc-server:/path/to/workspace/AlphaEdit/data/stats/

# Option 2: Download on HPC using gdown
module load python
pip install gdown
gdown https://drive.google.com/uc?id=1rAeGBJccEaZYFpPMlD5tb5TNjkaUqwq6
unzip <file>.zip -d data/stats/
```

### SLURM Job Scripts

#### Example 1: Quick Demo Job

Create `slurm_demo.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=reled_demo
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=your_account        # CHANGE THIS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Load modules
module load cuda/11.8 python/3.10

# Activate environment
source venv/bin/activate
# OR: conda activate alphaedit

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/scratch/your_username/hf_cache
export HF_HOME=/scratch/your_username/hf_home

# Job info
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "========================================="

# Run experiment
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=100 \
    --num_edits=10 \
    --downstream_eval_steps=5 \
    --dir_name=RelEdit_demo_${SLURM_JOB_ID}

# Completion info
echo "========================================="
echo "Completed at: $(date)"
echo "Results: results/RelEdit_demo_${SLURM_JOB_ID}/"
echo "========================================="
```

#### Example 2: Full Evaluation Job

Create `slurm_full.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=reled_full
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=your_account        # CHANGE THIS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G

module load cuda/11.8 python/3.10
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=2000 \
    --num_edits=100 \
    --downstream_eval_steps=5 \
    --dir_name=RelEdit_full_${SLURM_JOB_ID}

echo "Results: results/RelEdit_full_${SLURM_JOB_ID}/"
```

#### Example 3: Ablation Study (Multiple α Values)

Create `slurm_ablation.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=reled_ablation
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --account=your_account        # CHANGE THIS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G

module load cuda/11.8 python/3.10
source venv/bin/activate

# Alpha values to test
ALPHAS=(0.0 0.25 0.5 0.75 1.0)

for alpha in "${ALPHAS[@]}"
do
    echo "========================================="
    echo "Running with alpha=${alpha}"
    echo "========================================="

    # Create temporary config with modified alpha
    TEMP_CONFIG="Llama3-8B_alpha${alpha}.json"

    # Modify alpha using Python
    python -c "
import json
with open('hparams/RelEdit/Llama3-8B.json', 'r') as f:
    config = json.load(f)
config['alpha'] = ${alpha}
with open('hparams/RelEdit/${TEMP_CONFIG}', 'w') as f:
    json.dump(config, f, indent=4)
"

    # Run experiment
    python -m experiments.evaluate \
        --alg_name=RelEdit \
        --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
        --hparams_fname=${TEMP_CONFIG} \
        --ds_name=mcf \
        --dataset_size_limit=500 \
        --num_edits=50 \
        --downstream_eval_steps=5 \
        --dir_name=RelEdit_alpha${alpha}_${SLURM_JOB_ID}

    echo "Completed alpha=${alpha}"
done

echo "All ablation experiments completed!"
```

#### Example 4: Method Comparison

Create `slurm_comparison.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=alpha_vs_reled
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --account=your_account        # CHANGE THIS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G

module load cuda/11.8 python/3.10
source venv/bin/activate

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS="Llama3-8B.json"

# Run AlphaEdit
echo "Running AlphaEdit..."
python -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=${MODEL} \
    --hparams_fname=${HPARAMS} \
    --ds_name=mcf \
    --dataset_size_limit=1000 \
    --num_edits=50 \
    --dir_name=AlphaEdit_comp_${SLURM_JOB_ID}

# Run RelEdit
echo "Running RelEdit..."
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=${MODEL} \
    --hparams_fname=${HPARAMS} \
    --ds_name=mcf \
    --dataset_size_limit=1000 \
    --num_edits=50 \
    --dir_name=RelEdit_comp_${SLURM_JOB_ID}

echo "Comparison completed!"
echo "AlphaEdit: results/AlphaEdit_comp_${SLURM_JOB_ID}/"
echo "RelEdit: results/RelEdit_comp_${SLURM_JOB_ID}/"
```

### Submitting and Managing Jobs

```bash
# Create logs directory
mkdir -p logs

# Submit jobs
sbatch slurm_demo.sh
sbatch slurm_full.sh
sbatch slurm_ablation.sh
sbatch slurm_comparison.sh

# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# Monitor output in real-time
tail -f logs/reled_demo.<job_id>.out

# Cancel job
scancel <job_id>

# Check job statistics
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

### Batch Job Submission

Create `batch_submit.sh` for submitting multiple jobs:

```bash
#!/bin/bash

# Arrays of models and datasets
MODELS=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt2-xl" "EleutherAI/gpt-j-6B")
CONFIGS=("Llama3-8B.json" "gpt2-xl.json" "EleutherAI_gpt-j-6B.json")
DATASETS=("mcf" "cf" "zsre")

# Submit jobs for each combination
for i in "${!MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        MODEL=${MODELS[$i]}
        CONFIG=${CONFIGS[$i]}

        echo "Submitting: ${MODEL} on ${dataset}"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=reled_${dataset}_m${i}
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G

module load cuda/11.8 python/3.10
source venv/bin/activate

python -m experiments.evaluate \\
    --alg_name=RelEdit \\
    --model_name=${MODEL} \\
    --hparams_fname=${CONFIG} \\
    --ds_name=${dataset} \\
    --dataset_size_limit=1000 \\
    --num_edits=50 \\
    --dir_name=RelEdit_${dataset}_m${i}_\${SLURM_JOB_ID}
EOF

        sleep 1
    done
done

echo "All jobs submitted!"
```

Usage:
```bash
chmod +x batch_submit.sh
./batch_submit.sh
```

---

## ⚙️ Experiment Configurations

### Command Line Arguments

| Argument | Description | Example Values | Required |
|----------|-------------|----------------|----------|
| `--alg_name` | Algorithm name | AlphaEdit, RelEdit, ROME, MEMIT | Yes |
| `--model_name` | HuggingFace model ID | meta-llama/Meta-Llama-3-8B-Instruct | Yes |
| `--hparams_fname` | Hyperparameter config file | Llama3-8B.json | Yes |
| `--ds_name` | Dataset name | mcf, cf, zsre, mquake | Yes |
| `--dataset_size_limit` | Number of samples | 100, 1000, 2000 | Yes |
| `--num_edits` | Edits per batch | 10, 50, 100 | No (default: 1) |
| `--downstream_eval_steps` | GLUE eval interval | 5, 10 | No (default: 1) |
| `--dir_name` | Output directory | my_experiment | No |
| `--continue_from_run` | Continue from run ID | run_<timestamp> | No |
| `--conserve_memory` | Enable memory conservation | flag | No |
| `--skip_generation_tests` | Skip generation tests | flag | No |

### RelEdit-Specific Hyperparameters

Edit `hparams/RelEdit/<model>.json`:

```json
{
    "model_name": "Llama3-8B",
    "layers": [4, 5, 6, 7, 8],
    "nullspace_threshold": 0.02,
    "L2": 10,

    // RelEdit-specific
    "num_paths": 10,              // Number of KG paths [1-50]
    "max_path_length": 3,         // Max KG depth [1-5]
    "alpha": 0.5,                 // Combination weight [0-1]
    "use_kg_sampling": true,      // Enable KG sampling
    "kg_cache_dir": "data/kg_cache"
}
```

### Experiment Scenarios

#### Scenario 1: Quick Validation (5 minutes)

```bash
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=mcf \
    --dataset_size_limit=10 \
    --num_edits=5 \
    --dir_name=quick_test
```

#### Scenario 2: Paper-Level Evaluation (8-12 hours)

```bash
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=2000 \
    --num_edits=100 \
    --downstream_eval_steps=5 \
    --dir_name=paper_results
```

#### Scenario 3: Memory-Constrained Environment

```bash
python -m experiments.evaluate \
    --alg_name=RelEdit \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=mcf \
    --dataset_size_limit=500 \
    --num_edits=25 \
    --conserve_memory \
    --dir_name=low_memory
```

#### Scenario 4: Alpha Sensitivity Analysis

```bash
# Test different alpha values
for alpha in 0.0 0.25 0.5 0.75 1.0; do
    # Modify config
    python -c "
import json
config = json.load(open('hparams/RelEdit/Llama3-8B.json'))
config['alpha'] = ${alpha}
json.dump(config, open('hparams/RelEdit/Llama3-8B_a${alpha}.json', 'w'), indent=4)
"

    # Run experiment
    python -m experiments.evaluate \
        --alg_name=RelEdit \
        --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
        --hparams_fname=Llama3-8B_a${alpha}.json \
        --ds_name=mcf \
        --dataset_size_limit=500 \
        --num_edits=50 \
        --dir_name=alpha_${alpha}
done
```

---

## 📊 Results Analysis

### Results Directory Structure

```
results/
└── RelEdit/
    └── run_<timestamp>/
        ├── params.json              # Experiment parameters
        ├── case_0.json             # Batch 0 results
        ├── case_100.json           # Batch 1 results
        ├── case_200.json           # Batch 2 results
        └── glue_results/           # GLUE evaluation
            ├── cola_results.json
            ├── sst2_results.json
            ├── mnli_results.json
            └── ...
```

### Viewing Results

#### Method 1: Direct JSON Inspection

```bash
# View experiment parameters
cat results/RelEdit/run_<id>/params.json | jq

# View specific case results
cat results/RelEdit/run_<id>/case_0.json | jq '.post'

# Extract success rate
cat results/RelEdit/run_<id>/case_0.json | jq '.post.rewrite_prompts_correct'

# View GLUE results
cat results/RelEdit/run_<id>/glue_results/cola_results.json | jq
```

#### Method 2: Using Summarize Script

```bash
# Summarize single run
python experiments/summarize.py \
    --dir_name=RelEdit \
    --runs=run_<id>

# Compare multiple runs
python experiments/summarize.py \
    --dir_name=RelEdit \
    --runs=run_<id1>,run_<id2>,run_<id3>
```

#### Method 3: Custom Analysis Script

Create `analyze_results.py`:

```python
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def analyze_run(run_dir):
    """Analyze a single experimental run"""
    run_dir = Path(run_dir)

    # Load parameters
    params = json.load(open(run_dir / "params.json"))
    print(f"Algorithm: {params['alg_name']}")
    print(f"Model: {params['model_name']}")
    print(f"Dataset: {params['ds_name']}")

    # Collect case results
    cases = []
    for case_file in sorted(run_dir.glob("case_*.json")):
        cases.append(json.load(open(case_file)))

    # Compute metrics
    metrics = {
        "rewrite_success": [],
        "paraphrase_score": [],
        "neighborhood_score": [],
    }

    for case in cases:
        post = case["post"]
        metrics["rewrite_success"].append(
            np.mean(post["rewrite_prompts_correct"])
        )
        metrics["paraphrase_score"].append(
            np.mean(post["paraphrase_prompts_correct"])
        )
        metrics["neighborhood_score"].append(
            np.mean(post["neighborhood_prompts_correct"])
        )

    # Print results
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:25s}: {mean:.2%} ± {std:.2%}")

    # Plot metrics over time
    plt.figure(figsize=(12, 4))
    for i, (metric, values) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        plt.plot(values)
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Batch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(run_dir / "metrics_plot.png")
    print(f"\nPlot saved to {run_dir / 'metrics_plot.png'}")

    return metrics

# Usage
if __name__ == "__main__":
    run_dir = "results/RelEdit/run_<your_id>"
    analyze_run(run_dir)
```

Run analysis:
```bash
python analyze_results.py
```

### Key Metrics

| Metric | Description | Target | Interpretation |
|--------|-------------|--------|----------------|
| **Rewrite Success** | % of successful edits | >95% | Edit effectiveness |
| **Paraphrase Score** | Consistency on paraphrases | >90% | Generalization |
| **Neighborhood Score** | Preservation of related facts | >85% | Locality |
| **Generation Quality** | Fluency and coherence | >80% | Model integrity |
| **GLUE Scores** | Downstream task performance | <5% drop | General ability |

### Comparing Methods

Create `compare_methods.py`:

```python
import json
from pathlib import Path
import pandas as pd

def compare_runs(run_dirs, labels):
    """Compare multiple experimental runs"""
    results = []

    for run_dir, label in zip(run_dirs, labels):
        run_dir = Path(run_dir)

        # Collect metrics
        cases = [json.load(open(f)) for f in sorted(run_dir.glob("case_*.json"))]

        avg_metrics = {
            "Method": label,
            "Rewrite Success": np.mean([
                np.mean(c["post"]["rewrite_prompts_correct"]) for c in cases
            ]),
            "Paraphrase": np.mean([
                np.mean(c["post"]["paraphrase_prompts_correct"]) for c in cases
            ]),
            "Neighborhood": np.mean([
                np.mean(c["post"]["neighborhood_prompts_correct"]) for c in cases
            ]),
        }
        results.append(avg_metrics)

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.set_index("Method")

    # Display
    print(df.to_string())
    print("\nBest method per metric:")
    print(df.idxmax())

    return df

# Usage
run_dirs = [
    "results/AlphaEdit/run_123",
    "results/RelEdit/run_456",
]
labels = ["AlphaEdit", "RelEdit"]

compare_runs(run_dirs, labels)
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

```bash
# Option 1: Reduce batch size
--num_edits=25  # Instead of 100

# Option 2: Enable memory conservation
--conserve_memory

# Option 3: Use smaller model
--model_name=gpt2-xl  # Instead of Llama3-8B

# Option 4: Reduce dataset size
--dataset_size_limit=500  # Instead of 2000

# Option 5: Clear cache in code
import torch
torch.cuda.empty_cache()
```

#### Issue 2: Wikidata API Timeout

**Error:**
```
requests.exceptions.Timeout: HTTPConnectionPool
```

**Solutions:**

```bash
# Solution 1: Increase timeout in RelEdit/kg_sampler.py
response = self.session.get(url, params=params, timeout=30)  # Increase from 10 to 30

# Solution 2: Disable KG sampling temporarily
# Edit hparams/RelEdit/<model>.json
{
    "use_kg_sampling": false
}

# Solution 3: Use cached results
# The KG sampler automatically caches results in data/kg_cache/
# Cached results will be reused on subsequent runs
```

#### Issue 3: Missing Covariance Matrix

**Error:**
```
FileNotFoundError: data/stats/...
```

**Solution:**

```bash
# Ensure covariance matrix is downloaded and extracted
mkdir -p data/stats

# Download from:
# https://drive.google.com/file/d/1rAeGBJccEaZYFpPMlD5tb5TNjkaUqwq6/view

# Extract to data/stats/
unzip <downloaded_file>.zip -d data/stats/

# Verify
ls data/stats/  # Should show covariance files
```

#### Issue 4: Model Download Fails

**Error:**
```
OSError: Can't load tokenizer for 'meta-llama/...'
```

**Solution:**

```bash
# Set up HuggingFace cache
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/hf_home

# Login to HuggingFace (if model requires authentication)
huggingface-cli login

# Pre-download model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Model downloaded successfully')
"
```

#### Issue 5: HPC Job Stays in Queue

**Symptoms:**
```bash
$ squeue -u $USER
JOBID  PARTITION  NAME  STATE  TIME  NODELIST(REASON)
12345  gpu        job   PD     0:00  (Resources)
```

**Solutions:**

```bash
# Check partition status
sinfo -p gpu

# Reduce resource requirements
#SBATCH --time=04:00:00  # Reduce from 12:00:00
#SBATCH --mem=32G        # Reduce from 64G

# Check account status
sacctmgr show user $USER

# Try different QoS
#SBATCH --qos=high       # Instead of standard (if available)
```

#### Issue 6: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'RelEdit'
```

**Solution:**

```bash
# Ensure you're in the project root directory
cd /path/to/AlphaEdit

# Run with python -m
python -m experiments.evaluate ...  # Correct

# NOT:
cd experiments
python evaluate.py ...  # Wrong
```

#### Issue 7: JSON Decoding Error in Results

**Error:**
```
json.decoder.JSONDecodeError: Expecting value
```

**Solution:**

```bash
# Check if experiment completed
ls -lh results/RelEdit/run_<id>/

# If case files are empty or incomplete, re-run experiment
# Use --continue_from_run to resume
python -m experiments.evaluate \
    --continue_from_run=run_<id> \
    ...
```

### Performance Optimization

#### Speed Up Experiments

```bash
# 1. Skip generation tests (faster, less comprehensive)
--skip_generation_tests

# 2. Reduce evaluation frequency
--downstream_eval_steps=10  # Instead of 5

# 3. Use pre-computed statistics (automatic)
# Covariance matrices are cached in data/stats/

# 4. Increase batch size (if memory allows)
--num_edits=200  # Larger batches = fewer iterations

# 5. Use faster model for testing
--model_name=gpt2-xl  # Faster than Llama3-8B
```

#### Reduce Memory Usage

```bash
# 1. Enable memory conservation flag
--conserve_memory

# 2. Use gradient checkpointing (in code)
model.gradient_checkpointing_enable()

# 3. Use mixed precision
model.half()  # FP16 instead of FP32

# 4. Reduce cache size
# Edit globals.yml
cache_dir: /scratch/$USER/cache  # Use fast scratch space
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs**: Review `.out` and `.err` files
2. **Enable verbose mode**: Add `--verbose` flag (if available)
3. **Test with minimal setup**: Use quick validation scenario
4. **Check GitHub Issues**: Search for similar problems
5. **Create new issue**: Provide error message, command, and environment details

---

## 📝 Citation

If you use this code in your research, please cite:

### AlphaEdit

```bibtex
@inproceedings{alphaedit2025,
  title={AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models},
  booktitle={International Conference on Learning Representations},
  year={2025},
  note={Outstanding Paper Award}
}
```

### RelEdit

```bibtex
@misc{rededit2025,
  title={RelEdit: Relation-Aware Knowledge Editing with Knowledge Graph Sampling},
  author={Your Name},
  year={2025},
  howpublished={Extension of AlphaEdit},
  note={Available at: https://github.com/your-username/AlphaEdit}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

---

## 🌟 Acknowledgments

- **AlphaEdit Team**: For the original implementation
- **HuggingFace**: For the Transformers library
- **Wikidata**: For the knowledge graph API
- **PyTorch Team**: For the deep learning framework

---

## 📞 Contact

- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **Project**: [https://github.com/your-username/AlphaEdit](https://github.com/your-username/AlphaEdit)

---

## 🔗 Quick Links

- [Original AlphaEdit Paper](https://openreview.net/forum?id=...)
- [HuggingFace Models](https://huggingface.co/models)
- [CounterFact Dataset](https://rome.baulab.info/)
- [Wikidata API Documentation](https://www.wikidata.org/wiki/Wikidata:Data_access)

---

## 📋 Quick Reference Card

### Most Used Commands

```bash
# Quick test (5 min)
python -m experiments.evaluate --alg_name=RelEdit --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json --ds_name=mcf --dataset_size_limit=10 \
    --num_edits=5 --dir_name=test

# Full evaluation (8 hours)
bash run_reled_full.sh

# Submit HPC job
sbatch slurm_full.sh

# Check job status
squeue -u $USER

# View results
python experiments/summarize.py --dir_name=RelEdit --runs=run_<id>
```

### Important Paths

```
Code:      AlphaEdit/AlphaEdit_main.py, RelEdit/RelEdit_main.py
Config:    hparams/RelEdit/Llama3-8B.json
Data:      data/stats/
Results:   results/RelEdit/
Cache:     data/kg_cache/
Logs:      logs/*.out, logs/*.err
```

### Default Parameter Values

```json
{
  "alpha": 0.5,
  "num_paths": 10,
  "max_path_length": 3,
  "nullspace_threshold": 0.02,
  "L2": 10,
  "num_edits": 1,
  "downstream_eval_steps": 1
}
```

---

**Last Updated**: 2025-09-30
**Version**: 1.0.0
**Maintainer**: Your Name

---

<p align="center">
  <strong>Happy Editing! 🚀</strong>
</p>
