"""Core mathematical functions for FIOR.

Implements the arithmetic of protocol.md section "Mathematical Model" and the
trust layer, per parameter group:

- Log-partition function A(eta) for every supported family
- Composition  eta_prior = eta0 + Σ p_n · Δη_n, with the domain check
- Bayesian Model Reduction log Bayes factor (unit-weight reference)
- p resolution  p = σ(ΔF + ln(a/b))
"""

import math

import numpy as np

from .types import (
    Eta,
    FAMILY_NORMAL,
    FAMILY_MVNORMAL,
    FAMILY_GAMMA,
    FAMILY_BETA,
    FAMILY_DIRICHLET,
    FAMILY_CATEGORICAL,
)


def _log_gamma(x):
    """log Γ(x) via the stdlib; scipy is not a dependency."""
    return math.lgamma(float(x))


def log_partition(eta: Eta) -> float:
    """Log-partition A(η) for one group, in f64.

    BMR takes a difference of four equal-dimensional evaluations, so constants
    that are identical across them (e.g. the `(d/2) ln(2π)` term of the
    multivariate Normal) are dropped here, exactly as in the reference
    simulation test/fior_sim.py.  The gamma and beta forms carry their ln Γ
    terms, which are not constant.
    """
    f = eta.family
    h = eta.h
    lam = eta.Lam

    if f == FAMILY_NORMAL:
        # A = Σ [ -h_i²/(4 η2_i) - ½ ln(-η2_i) ]
        if not np.all(lam < 0):
            raise ValueError("normal group outside its domain: Lam not < 0")
        return float(np.sum(-h**2 / (4 * lam) - 0.5 * np.log(-lam)))

    if f == FAMILY_MVNORMAL:
        # A = ½ h' Λ⁻¹ h - ½ ln det Λ   with Λ = -2 η2
        L = np.linalg.cholesky(-2.0 * lam)
        z = np.linalg.solve(L, h)
        return float(0.5 * (z @ z) - np.log(np.diag(L)).sum())

    if f == FAMILY_GAMMA:
        # η1 = α-1 > -1, η2 = -β < 0 ; A = ln Γ(α) - α ln β
        alpha = h + 1
        beta = -lam
        if not (np.all(alpha > 0) and np.all(beta > 0)):
            raise ValueError("gamma group outside its domain")
        return float(np.sum([_log_gamma(a) for a in alpha] - alpha * np.log(beta)))

    if f == FAMILY_BETA:
        # η1 = a-1, η2 = b-1 ; A = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
        a = h + 1
        b = lam + 1
        if not (np.all(a > 0) and np.all(b > 0)):
            raise ValueError("beta group outside its domain")
        return float(
            np.sum([_log_gamma(ai) + _log_gamma(bi) - _log_gamma(ai + bi) for ai, bi in zip(a, b)])
        )

    if f == FAMILY_DIRICHLET:
        # η = α - 1 ; A = Σ ln Γ(α_i) - ln Γ(Σ α_i)
        alpha = h + 1
        if not np.all(alpha > 0):
            raise ValueError("dirichlet group outside its domain")
        return float(sum(_log_gamma(ai) for ai in alpha) - _log_gamma(alpha.sum()))

    if f == FAMILY_CATEGORICAL:
        # log-sum-exp
        m = float(np.max(h))
        return m + float(np.log(np.sum(np.exp(h - m))))

    raise ValueError(f"unknown family: {f}")


def compose_prior(eta0: Eta, sites: list[Eta], p_values: list[float], strict: bool = True) -> Eta:
    """η_prior = η0 + Σ p_n · Δη_n for one group.

    Sites are differences, so the weighted sum can leave the family domain;
    the protocol specifies no remedy, so a caller composing must be told.
    `strict=True` (default) raises when the composed group is invalid (the
    interface.md requirement); `strict=False` returns the invalid Eta for
    inspection.
    """
    result = eta0.copy()
    for site, p in zip(sites, p_values):
        result = result + site * p
    if strict and not result.in_domain():
        raise ValueError(
            f"composed group {result.family} left its natural parameter domain"
        )
    return result


def bmr_delta_f(eta_q: Eta, eta_p: Eta, contrib: Eta) -> float:
    """ΔF = A(η_q) + A(η_p − c) − A(η_q − c) − A(η_p).

    `c` is the peer's contribution at the weight the comparison is drawn at.
    The protocol's mechanism 1 scores at unit weight: callers pass `eta_p`
    with the peer already present at full weight, so `eta_p − c` is the
    cavity.  Passed a p-scaled `contrib` instead, this asks a different,
    degenerate question (the finding-2 fixed point).
    """
    p_red = eta_p - contrib
    q_red = eta_q - contrib
    if not (p_red.in_domain() and q_red.in_domain()):
        # Cannot form the reduced model; treat as no evidence either way,
        # matching the reference simulation.  Not a silent path: a caller
        # that reaches this with a full-ranking site should inspect.
        return 0.0
    return log_partition(eta_q) + log_partition(p_red) - log_partition(q_red) - log_partition(eta_p)


def logistic(x: float) -> float:
    """Overflow-safe sigmoid; ΔF reaches thousands of nats at d=100."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def p_from(delta_f: float, a: float, b: float) -> float:
    """p = 1 / (1 + (b/a)·exp(−ΔF)) = σ(ΔF + ln(a/b))."""
    return logistic(delta_f + math.log(a) - math.log(b))


def posterior_odds_offset(p: float, kappa_r: float) -> float:
    """The log-prior-odds contribution of one attestation at strength κ_r.

    ln( (1 + κr·p) / (1 + κr·(1−p)) ), monotone in p, bounded by ±ln(1+κr).
    """
    num = 1.0 + kappa_r * p
    den = 1.0 + kappa_r * (1.0 - p)
    return math.log(num / den)