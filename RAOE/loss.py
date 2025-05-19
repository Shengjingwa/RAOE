import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def rank_margin_loss(similarity: torch.Tensor, embeddings: torch.Tensor, tau: float = 20.0, interval: float = 16.0) -> torch.Tensor:
    """
    Calculate rank margin loss based on similarity scores and embeddings with interval-based ranking.
    
    Args:
        similarity: Tensor of similarity scores (shape: [n])
        embeddings: Tensor of embeddings (shape: [2n, d])
        tau: Temperature parameter to scale the loss
        interval: Minimum rank difference to consider as positive
        
    Returns:
        A scalar tensor containing the loss value
    """
    # Normalize embeddings and compute cosine similarity between pairs
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    pair_similarities = torch.sum(norm_embeddings[::2] * norm_embeddings[1::2], dim=1)

    # Get ranking of samples based on their similarity scores
    ranks = torch.argsort(torch.argsort(similarity))
    
    # Create mask for pairs whose rank difference exceeds the interval threshold
    rank_diff_mask = (ranks.unsqueeze(1) - ranks) < -interval
    
    # Calculate pairwise differences between similarities
    similarity_differences = pair_similarities[:, None] - pair_similarities[None, :]
    
    # Add zero and apply temperature scaling
    device = similarity_differences.device
    zero = torch.zeros(1, device=device)
    scaled_differences = similarity_differences[rank_diff_mask] * tau
    combined = torch.cat((zero, scaled_differences), dim=0)
    
    # Return the log-sum-exp of the combined tensor
    return torch.logsumexp(combined, dim=0)


def gated_angular_loss(labels: torch.Tensor, embeddings: torch.Tensor, similarity_scores: Optional[torch.Tensor]=None, tau: float = 20.0):
    """
    Calculate gated angular loss using cosine similarity and angular distance.
    
    Args:
        labels: Ground truth labels for pairs (shape: [n])
        embeddings: Normalized embeddings (shape: [2n, d])
        similarity_scores: Similarity scores between pairs (shape: [n])
        tau: Temperature parameter to scale the loss
        
    Returns:
        Scalar tensor containing the loss value
    """
    # Normalize embeddings and compute cosine similarity between pairs
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    pair_cosine_sim = torch.sum(norm_embeddings[::2] * norm_embeddings[1::2], dim=1)
    
    # Convert cosine similarity to angle (π - arccos(cos_sim))
    # Clamp values to avoid numerical instability
    pair_angles = torch.pi - torch.acos(torch.clamp(pair_cosine_sim, -1.0, 0.999999))
    
    # Create mask for valid pairs based on label and similarity conditions
    if similarity_scores is not None:
        valid_pairs_mask = (labels[:, None] > labels[None, :]) & (similarity_scores[:, None] < similarity_scores[None, :])
    else:
        valid_pairs_mask = (labels[:, None] > labels[None, :])
    # Calculate pairwise differences between angles
    angle_differences = pair_angles[:, None] - pair_angles[None, :]
    
    # Get device of input tensors
    device = embeddings.device
    
    # Apply mask and temperature scaling
    scaled_differences = angle_differences[valid_pairs_mask] * tau
    
    # Add zero and compute log-sum-exp
    zero = torch.zeros(1, device=device)
    combined = torch.cat((zero, scaled_differences), dim=0)
    
    return torch.logsumexp(combined, dim=0)


