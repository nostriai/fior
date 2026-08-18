"""Core mathematical functions for FIOR.

Implements:
- Log-partition function A(η) for exponential families
- Composition: η₀ + Σ p·Δη
- Bayesian Model Reduction (BMR) log Bayes factor
- p resolution: p = σ(ΔF + ln(a/b))
"""

import numpy as np
from .types import Eta, FAMILY_NORMAL, FAMILY_GAMMA, FAMILY_BETA, FAMILY_DIRICHLET, FAMILY_CATEGORICAL


def log_partition(eta: Eta, family: str) -> float:
    """Compute log-partition function A(η).
    
    For Normal family:
        A(η) = -h²/(4*Lam) - 0.5 * log(-Lam)
    
    Args:
        eta: Natural parameters
        family: Distribution family
    
    Returns:
        Log-partition function value
    """
    h = eta.h
    Lam = eta.Lam
    
    if family == FAMILY_NORMAL:
        # A(η) = -h²/(4*Lam) - 0.5 * log(-Lam)
        # Note: Lam < 0 for valid Normal parameters
        return np.sum(-h**2 / (4 * Lam) - 0.5 * np.log(-Lam))
    
    elif family == FAMILY_GAMMA:
        # A(η) = -log(-Lam) - (h+1)*log(-h/Lam)
        # For Gamma: h = α-1, Lam = -β
        # A(η) = log Γ(α) - α log(β) = log Γ(h+1) + (h+1)*log(-Lam)
        alpha = h + 1
        beta = -Lam
        return np.sum(np.log(np.abs(alpha)) + alpha * np.log(np.abs(beta)))
    
    elif family == FAMILY_BETA:
        # A(η) = log Γ(a) + log Γ(b) - log Γ(a+b)
        # For Beta: h = a-1, Lam = b-1
        a = h + 1
        b = Lam + 1
        from scipy.special import gammaln
        return np.sum(gammaln(a) + gammaln(b) - gammaln(a + b))
    
    elif family == FAMILY_DIRICHLET:
        # A(η) = log Γ(Σα) - Σ log Γ(α)
        # For Dirichlet: h = α-1
        alpha = h + 1
        from scipy.special import gammaln
        return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    
    elif family == FAMILY_CATEGORICAL:
        # A(η) = log(Σ exp(η))
        # Log-sum-exp for numerical stability
        max_eta = np.max(h)
        return max_eta + np.log(np.sum(np.exp(h - max_eta)))
    
    raise ValueError(f"Unknown family: {family}")


def compose_prior(eta0: Eta, sites: list, p_values: list[float]) -> Eta:
    """Build a local prior from trusted peers.
    
    η_prior = η₀ + Σ p_n · Δη_n
    
    Args:
        eta0: Base prior
        sites: List of Site objects
        p_values: Inclusion probabilities (same order as sites)
    
    Returns:
        Composed prior Eta
    """
    result = eta0.copy()
    
    for site, p in zip(sites, p_values):
        result = result + site.delta_eta * p
    
    return result


def bmr_delta_f(
    eta_q_plus: Eta,
    eta_p_minus: Eta,
    eta_q_minus: Eta,
    eta_p_plus: Eta,
    family: str = FAMILY_NORMAL
) -> float:
    """Compute log Bayes factor from BMR.
    
    ΔF = A(η_q^+) + A(η_p^-) - A(η_q^-) - A(η_p^+)
    
    Args:
        eta_q_plus: Prior without peer + local data
        eta_p_minus: Prior without peer
        eta_q_minus: Prior without peer + local data (cavity + L_A)
        eta_p_plus: Prior with peer at full weight
        family: Distribution family
    
    Returns:
        Log Bayes factor (float)
    """
    A = log_partition
    
    return (
        A(eta_q_plus, family)
        + A(eta_p_minus, family)
        - A(eta_q_minus, family)
        - A(eta_p_plus, family)
    )


def p_from(delta_f: float, beta: float) -> float:
    """Convert ΔF and prior belief to inclusion probability.
    
    p = σ(ΔF + ln(β/(1-β)))
    
    where σ is the logistic function.
    
    Args:
        delta_f: Log Bayes factor from BMR
        beta: Prior inclusion probability from attestations (a/(a+b))
    
    Returns:
        Inclusion probability p in (0, 1)
    """
    # Clamp beta to (0, 1) exclusive
    beta = np.clip(beta, 1e-10, 1 - 1e-10)
    
    # Compute log-odds of beta
    log_odds_beta = np.log(beta / (1 - beta))
    
    # Compute p via logistic
    x = delta_f + log_odds_beta
    return 1 / (1 + np.exp(-x))


def logistic(x: float) -> float:
    """Logistic sigmoid function."""
    return 1 / (1 + np.exp(-x))
