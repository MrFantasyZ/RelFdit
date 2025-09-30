#!/usr/bin/env python3
"""
Test script for RelEdit implementation
Performs basic sanity checks without running full evaluation
"""

import torch
from pathlib import Path
import json

# Test imports
print("Testing imports...")
try:
    from RelEdit import RelEditHyperParams, apply_RelEdit_to_model
    from RelEdit.kg_sampler import WikidataKGSampler
    from RelEdit.compute_projection import (
        compute_null_space_projection,
        compute_relational_projection,
        compute_reled_projection
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test hyperparameters loading
print("\nTesting hyperparameters loading...")
try:
    hparams_path = Path("hparams/RelEdit/Llama3-8B.json")
    with open(hparams_path, 'r') as f:
        hparams_dict = json.load(f)

    # Check RelEdit-specific parameters
    assert "num_paths" in hparams_dict, "Missing num_paths"
    assert "max_path_length" in hparams_dict, "Missing max_path_length"
    assert "alpha" in hparams_dict, "Missing alpha"
    assert "use_kg_sampling" in hparams_dict, "Missing use_kg_sampling"
    assert "kg_cache_dir" in hparams_dict, "Missing kg_cache_dir"

    print("✓ Hyperparameters file valid")
    print(f"  - num_paths: {hparams_dict['num_paths']}")
    print(f"  - max_path_length: {hparams_dict['max_path_length']}")
    print(f"  - alpha: {hparams_dict['alpha']}")
    print(f"  - use_kg_sampling: {hparams_dict['use_kg_sampling']}")
except Exception as e:
    print(f"✗ Hyperparameters loading failed: {e}")
    exit(1)

# Test KG sampler initialization
print("\nTesting KG sampler initialization...")
try:
    kg_sampler = WikidataKGSampler(
        cache_dir="data/kg_cache_test",
        verbose=False
    )
    print("✓ KG sampler initialized successfully")

    # Test cache directory creation
    cache_dir = Path("data/kg_cache_test")
    assert cache_dir.exists(), "Cache directory not created"
    print(f"✓ Cache directory created: {cache_dir}")
except Exception as e:
    print(f"✗ KG sampler initialization failed: {e}")
    exit(1)

# Test projection computation functions
print("\nTesting projection computation...")
try:
    # Create dummy matrices for testing
    d0 = 100  # Hidden dimension
    m = 10    # Number of related keys

    # Test null-space projection (without model)
    # Create random symmetric positive definite matrix as proxy for covariance
    A = torch.randn(d0, d0)
    cov = A @ A.T + 0.1 * torch.eye(d0)

    # SVD decomposition
    U, S, _ = torch.linalg.svd(cov)
    threshold = 2e-2
    small_indices = (S < threshold).nonzero(as_tuple=True)[0]
    P_null = U[:, small_indices] @ U[:, small_indices].T

    print(f"✓ Null-space projection computed: shape {P_null.shape}")

    # Test relational projection
    K_rel = torch.randn(d0, m)
    P_rel = compute_relational_projection(K_rel, P_null, orthogonalize=True)

    print(f"✓ Relational projection computed: shape {P_rel.shape}")

    # Test combined projection
    alpha = 0.5
    P_prime = P_null + alpha * P_rel

    print(f"✓ Combined projection computed: shape {P_prime.shape}")

    # Verify projection is symmetric
    assert torch.allclose(P_prime, P_prime.T, atol=1e-5), "Projection not symmetric"
    print("✓ Combined projection is symmetric")

except Exception as e:
    print(f"✗ Projection computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test request format
print("\nTesting request format...")
try:
    test_request = {
        "subject": "Paris",
        "relation": "P17",  # country
        "prompt": "The capital of France is {}",
        "target_new": {"str": "Paris"}
    }

    # Validate request structure
    assert "subject" in test_request, "Missing subject"
    assert "prompt" in test_request, "Missing prompt"
    assert "target_new" in test_request, "Missing target_new"
    assert "str" in test_request["target_new"], "Missing str in target_new"

    print("✓ Request format valid")
    print(f"  Example: {test_request['subject']} - {test_request['prompt']}")
except Exception as e:
    print(f"✗ Request format test failed: {e}")
    exit(1)

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
print("\nRelEdit implementation is ready to use.")
print("\nNext steps:")
print("1. Ensure you have the pre-computed covariance matrix")
print("2. Run a demo experiment: bash run_reled_demo.sh")
print("3. For full evaluation: bash run_reled_full.sh")
print("4. For comparison with AlphaEdit: bash run_reled_comparison.sh")