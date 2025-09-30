import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from AlphaEdit.compute_z import compute_z, get_module_input_output_at_words
from AlphaEdit.AlphaEdit_main import get_context_templates, get_cov
from .RelEdit_hparams import RelEditHyperParams
from .kg_sampler import WikidataKGSampler
from .compute_projection import compute_reled_projection

# Cache for deltas and residuals
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_RelEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: RelEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: str = None,
    cache_c=None,
    P=None,  # Projection matrices for all layers
    kg_sampler: WikidataKGSampler = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Apply RelEdit to model.

    Args:
        model: The language model to edit
        tok: Tokenizer
        requests: List of editing requests, each containing:
            - subject: str
            - relation: str (optional, for KG sampling)
            - prompt: str
            - target_new: Dict with 'str' key
        hparams: RelEdit hyperparameters
        copy: Whether to copy the model
        return_orig_weights: Whether to return original weights
        cache_template: Template for caching k/v pairs
        cache_c: Accumulated covariance matrix for sequential editing
        P: Pre-computed projection matrices
        kg_sampler: KG sampler instance (initialized if None)

    Returns:
        Tuple of (edited_model, updated_cache_c or orig_weights)
    """
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # Initialize KG sampler if needed
    if kg_sampler is None and hparams.use_kg_sampling:
        kg_sampler = WikidataKGSampler(
            cache_dir=hparams.kg_cache_dir,
            verbose=True
        )
        print(f"Initialized KG sampler with cache dir: {hparams.kg_cache_dir}")

    # Get context templates (cached)
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = get_context_templates(model, tok)
    context_templates = CONTEXT_TEMPLATES_CACHE

    # Compute z (target values) for each request
    deltas = []
    for request in requests:
        if "target_new" not in request:
            raise ValueError("Request must contain 'target_new' field")

        # Use the last layer for computing z
        layer = hparams.layers[-1]
        print(f"\nComputing target value for request: {request.get('prompt', 'N/A')}")

        # Compute target representation z*
        z = compute_z(
            model, tok, request, hparams, layer, context_templates
        )

        deltas.append(z)

    # Convert to matrix V1: [d1, u]
    V1 = torch.stack(deltas, dim=1)

    # Compute keys K1 for updated knowledge
    # Extract subjects and prompts
    subjects = [r["subject"] for r in requests]
    prompts = [r["prompt"] for r in requests]

    # Get context templates (use first template for each request)
    context_templates_for_keys = [
        [template[0] for template in context_templates]
        for _ in requests
    ]

    # Compute input keys K1
    cur_k, cur_v = get_module_input_output_at_words(
        model,
        tok,
        hparams.layers[-1],
        context_templates=context_templates_for_keys,
        words=subjects,
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    K1 = cur_k.T  # [d0, u]

    # For sequential editing: load previous knowledge
    if cache_c is not None:
        Kp = cache_c  # Simplified: use accumulated matrix
    else:
        Kp = None

    # Initialize projection matrices if not provided
    if P is None:
        print("\nComputing null-space projection matrices...")
        P = {}
        for i, layer in enumerate(hparams.layers):
            # Compute base covariance
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
            ).cpu()

            # Compute null-space projection
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            threshold = hparams.nullspace_threshold
            small_indices = (S < threshold).nonzero(as_tuple=True)[0]
            P[layer] = U[:, small_indices] @ U[:, small_indices].T

        print(f"Computed projection matrices for layers: {list(P.keys())}")

    # Sample related knowledge and compute RelEdit projections
    print("\n=== RelEdit: Sampling related knowledge from KG ===")
    P_prime = {}

    for i, layer in enumerate(hparams.layers):
        K_rel_list = []

        if hparams.use_kg_sampling and kg_sampler is not None:
            # Sample related keys for each request
            for request in requests:
                subject = request.get("subject", "")
                relation = request.get("relation", "")

                # For demo: use simplified entity-to-key mapping
                # In real implementation, maintain proper entity mapping
                entity_to_key_map = {}

                # Sample related keys from K0 (preserved knowledge)
                # Here we use K1 as a proxy for K0 for demonstration
                K_rel = kg_sampler.sample_related_keys(
                    subject=subject,
                    relation=relation,
                    K0=K1,  # In practice, use actual K0
                    entity_to_key_map=entity_to_key_map,
                    k=hparams.num_paths,
                    max_path_length=hparams.max_path_length
                )

                if K_rel is not None:
                    K_rel_list.append(K_rel)

        # Combine related keys
        if K_rel_list:
            K_rel_combined = torch.cat(K_rel_list, dim=1)
            print(f"Layer {layer}: Sampled {K_rel_combined.shape[1]} related keys")
        else:
            K_rel_combined = None
            print(f"Layer {layer}: No related keys sampled, using pure null-space")

        # Compute RelEdit projection
        P_prime[layer] = compute_reled_projection(
            model=model,
            tok=tok,
            layer=layer,
            hparams=hparams,
            K_rel=K_rel_combined,
            alpha=hparams.alpha
        )

    # Compute weight updates for each layer
    print("\n=== Computing weight updates ===")
    with torch.no_grad():
        for i, layer in enumerate(hparams.layers):
            print(f"\nLayer {layer}:")

            # Get weight matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            W = nethook.get_parameter(model, weight_name)

            # Compute residual R = V1 - WK1
            R = V1.to(W.device) - torch.matmul(W, K1.to(W.device))

            # Get projection matrix
            P_layer = P_prime[layer].to(W.device, dtype=W.dtype)

            # Project new and cached keys into the constrained subspace
            K1_device = K1.to(W.device, dtype=W.dtype)
            K1_P = torch.matmul(K1_device.T, P_layer)

            mat_to_inv = torch.matmul(K1_P.T, K1_P)

            if Kp is not None and Kp.numel() > 0:
                Kp_device = Kp.to(W.device, dtype=W.dtype)
                Kp_P = torch.matmul(Kp_device.T, P_layer)
                mat_to_inv = mat_to_inv + torch.matmul(Kp_P.T, Kp_P)

            mat_to_inv = mat_to_inv + hparams.L2 * torch.eye(
                P_layer.shape[1], device=W.device, dtype=W.dtype
            )
            # Solve for delta
            try:
                delta = torch.matmul(
                    torch.matmul(R, K1_P),
                    torch.linalg.inv(mat_to_inv)
                )
                delta = torch.matmul(delta, P_layer.T)
            except Exception as e:
                print(f"Warning: Matrix inversion failed: {e}")
                print("Using pseudo-inverse instead")
                delta = torch.matmul(
                    torch.matmul(R, K1_P),
                    torch.linalg.pinv(mat_to_inv)
                )
                delta = torch.matmul(delta, P_layer.T)

            # Store original weights if requested
            if return_orig_weights and weight_name not in weights_copy:
                weights_copy[weight_name] = W.detach().clone()

            # Apply update
            W[...] += delta

            print(f"Update norm: {delta.norm().item():.4f}")
            print(f"Relative update: {(delta.norm() / W.norm()).item():.4f}")

    # Update cache_c for sequential editing
    if cache_c is not None:
        # Accumulate K1 into cache
        cache_c = torch.cat([cache_c, K1.cpu()], dim=1) if cache_c.numel() > 0 else K1.cpu()
    else:
        cache_c = K1.cpu()

    print(f"\n{'='*60}")
    print(f"RelEdit completed for {len(requests)} requests")
    print(f"{'='*60}\n")

    if return_orig_weights:
        return model, weights_copy
    else:
        return model, cache_c


def get_cov_cached(model, tok, layer_name, mom2_dataset, mom2_n_samples, mom2_dtype):
    """Cached version of get_cov"""
    cache_key = f"{layer_name}_{mom2_dataset}_{mom2_n_samples}_{mom2_dtype}"

    if cache_key not in COV_CACHE:
        COV_CACHE[cache_key] = get_cov(
            model, tok, layer_name, mom2_dataset, mom2_n_samples, mom2_dtype
        )

    return COV_CACHE[cache_key]
