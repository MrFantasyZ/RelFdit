import torch
from typing import Optional
from AlphaEdit.AlphaEdit_main import get_cov
def compute_null_space_projection(model, tok, layer, hparams):
    """
    Compute null-space projection matrix (same as AlphaEdit)
    Args:
        model: The language model
        tok: Tokenizer
        layer: Layer number
        hparams: Hyperparameters
    Returns:
        P_null: Null-space projection matrix [d0, d0]
    """
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        inv=False,
        force_recompute=force_recompute,
    ).cpu()
    # SVD decomposition
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    # Filter small singular values to get null space
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    if small_singular_indices.numel() == 0:
        # No singular values under threshold; fall back to identity projector
        print(
            f"Layer {layer}: Null space empty for threshold {threshold:.2e}; using identity projector"
        )
        dim = U.shape[0]
        return torch.eye(dim, dtype=U.dtype, device=U.device)
    print(f"Layer {layer}: Found {len(small_singular_indices)} dimensions in null space")
    basis = U[:, small_singular_indices]
    P_null = basis @ basis.T
    return P_null
def compute_relational_projection(K_rel: torch.Tensor,
                                  P_null: torch.Tensor,
                                  orthogonalize: bool = True) -> torch.Tensor:
    """
    Compute projection matrix for related knowledge
    Args:
        K_rel: Related knowledge keys [d0, m]
        P_null: Null-space projection matrix [d0, d0]
        orthogonalize: Whether to orthogonalize K_rel against null space
    Returns:
        P_rel: Projection matrix onto span(K_rel) [d0, d0]
    """
    if K_rel is None or K_rel.shape[1] == 0:
        # Return zero projection if no related keys
        return torch.zeros_like(P_null)
    # Ensure K_rel is on the same device and dtype as P_null
    K_rel = K_rel.to(P_null.device).to(P_null.dtype)
    if orthogonalize:
        # Orthogonalize K_rel against null space
        # K_rel_orth = K_rel - P_null @ K_rel
        K_rel_orth = K_rel - torch.matmul(P_null, K_rel)
        # Check if K_rel_orth has non-zero columns
        norms = torch.norm(K_rel_orth, dim=0)
        valid_cols = norms > 1e-6
        if valid_cols.sum() == 0:
            print("Warning: All K_rel columns are in null space, using original K_rel")
            K_rel_orth = K_rel
        else:
            K_rel_orth = K_rel_orth[:, valid_cols]
    else:
        K_rel_orth = K_rel
    # QR decomposition to get orthonormal basis
    try:
        Q, R = torch.linalg.qr(K_rel_orth)
    except Exception as e:
        print(f"QR decomposition failed: {e}, falling back to SVD")
        # Fallback to SVD if QR fails
        U, S, Vh = torch.linalg.svd(K_rel_orth, full_matrices=False)
        # Keep only non-zero singular values
        valid_dims = S > 1e-6
        Q = U[:, valid_dims]
    # Projection matrix onto span(K_rel)
    P_rel = torch.matmul(Q, Q.T)
    return P_rel
def compute_reled_projection(model,
                             tok,
                             layer,
                             hparams,
                             K_rel: Optional[torch.Tensor] = None,
                             alpha: float = 0.5) -> torch.Tensor:
    """
    Compute RelEdit combined projection matrix
    Args:
        model: The language model
        tok: Tokenizer
        layer: Layer number
        hparams: Hyperparameters
        K_rel: Related knowledge keys [d0, m], None for AlphaEdit mode
        alpha: Weight parameter in [0,1]
    Returns:
        P_prime: Combined projection matrix P' = P_null + alpha * P_rel
    """
    # Compute null-space projection
    P_null = compute_null_space_projection(model, tok, layer, hparams)
    # If no related knowledge, fall back to AlphaEdit
    if K_rel is None or not hparams.use_kg_sampling:
        print(f"Layer {layer}: Using pure null-space projection (AlphaEdit mode)")
        return P_null
    # Compute relational projection
    P_rel = compute_relational_projection(K_rel, P_null, orthogonalize=True)
    # Combine projections
    P_prime = P_null + alpha * P_rel
    print(f"Layer {layer}: Combined projection with alpha={alpha}, P_rel rank {torch.linalg.matrix_rank(P_rel).item()}")
    return P_prime
