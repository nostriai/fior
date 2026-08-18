"""Trust layer implementation for FIOR.

Implements:
- TrustTable: per-peer, per-group p values
- Corroboration: one-peer-one-vote with MAD scale
- Loewner cap: bound claimed precision
"""

import numpy as np
from typing import Optional
from .types import Eta, Site


class TrustTable:
    """Per-peer, per-group inclusion probabilities.
    
    Stores p values for each peer and parameter group.
    """
    
    def __init__(self):
        # p[peer_pubkey][group_name] = inclusion probability
        self._p: dict[str, dict[str, float]] = {}
    
    def get_p(self, peer: str, group: str, default: float = 0.5) -> float:
        """Get inclusion probability for a peer and group."""
        return self._p.get(peer, {}).get(group, default)
    
    def set_p(self, peer: str, group: str, p: float) -> None:
        """Set inclusion probability for a peer and group."""
        if peer not in self._p:
            self._p[peer] = {}
        self._p[peer][group] = np.clip(p, 1e-10, 1 - 1e-10)
    
    def get_all_p(self, peer: str, groups: list[str]) -> dict[str, float]:
        """Get all p values for a peer across groups."""
        return {group: self.get_p(peer, group) for group in groups}
    
    def peers(self) -> list[str]:
        """List all peers with p values."""
        return list(self._p.keys())
    
    def reset(self, peer: Optional[str] = None, group: Optional[str] = None) -> None:
        """Reset p values to default (0.5)."""
        if peer is None:
            self._p.clear()
        elif group is None:
            self._p.pop(peer, None)
        else:
            if peer in self._p:
                self._p[peer].pop(group, None)


class Corroboration:
    """One-peer-one-vote with MAD scale.
    
    Checks if a peer's claims agree with trusted peers.
    Uses median absolute deviation (MAD) for robust scale estimation.
    """
    
    def __init__(self, threshold: float = 3.0):
        """Initialize corroboration engine.
        
        Args:
            threshold: How many MADs from median before penalizing
        """
        self.threshold = threshold
    
    def score(
        self,
        site: Site,
        trusted_sites: list[Site],
        p_values: list[float]
    ) -> float:
        """Score a site against trusted peers.
        
        Args:
            site: Site to evaluate
            trusted_sites: List of trusted sites
            p_values: p values for trusted sites
        
        Returns:
            Multiplicative factor in (0, 1] to apply to p
        """
        if not trusted_sites:
            return 1.0
        
        # Extract precision blocks from sites
        # For Normal: precision is -2 * η₂
        site_precision = -2 * site.delta_eta.Lam
        
        # Weighted median of trusted precisions
        trusted_precisions = np.array([-2 * s.delta_eta.Lam for s in trusted_sites])
        weights = np.array(p_values) / np.sum(p_values)
        
        # Compute weighted median
        median_precision = self._weighted_median(trusted_precisions, weights)
        
        # Compute MAD (median absolute deviation)
        mad = self._weighted_mad(trusted_precisions, weights, median_precision)
        
        # If MAD is too small (e.g., only one trusted site), cannot assess corroboration
        # Return 1.0 (no penalty) - corroboration requires multiple data points
        if mad < 1e-9:
            return 1.0
        
        # Compute discrepancy for each dimension
        discrepancy = np.abs(site_precision - median_precision) / mad
        
        # Use worst direction
        worst_discrepancy = np.max(discrepancy)
        
        # Apply threshold
        if worst_discrepancy > self.threshold:
            # Penalize proportionally
            return 1.0 / (1.0 + (worst_discrepancy - self.threshold))
        
        return 1.0
    
    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted median."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        
        return sorted_values[min(median_idx, len(sorted_values) - 1)]
    
    def _weighted_mad(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        median: float
    ) -> float:
        """Compute weighted median absolute deviation."""
        absolute_deviations = np.abs(values - median)
        return self._weighted_median(absolute_deviations, weights)


def spectral_clip_site(
    site: Eta,
    others_prior: Eta,
    cap: float = 3.0
) -> Eta:
    """Apply Loewner confidence bound to a site.
    
    Bounds the site's claimed precision against what other peers provide:
        Λ_n ⪯ c · Λ_others
    
    Args:
        site: Site to clip
        others_prior: Prior from other peers (excluding this site)
        cap: Confidence bound constant c
    
    Returns:
        Clipped site Eta
    """
    # For Normal family: Λ = -2 * η₂
    site_Lam = site.Lam
    others_Lam = others_prior.Lam
    
    # Compute ratio of precisions
    # site claims -2*site_Lam, others provide -2*others_Lam
    ratio = site_Lam / others_Lam  # Both negative, ratio is positive
    
    # Clip ratio to cap
    clipped_ratio = np.minimum(ratio, cap)
    
    # Rescale η₁ to preserve implied mean
    # Original: μ = -η₁/(2*η₂)
    # Clipped: η₁_new = -μ * 2 * η₂_new = η₁ * (η₂_new / η₂)
    clipped_Lam = others_Lam * clipped_ratio
    clipped_h = site.h * (clipped_Lam / site_Lam)
    
    return Eta(clipped_h, clipped_Lam)
