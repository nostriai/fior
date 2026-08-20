"""Trust layer implementation for FIOR (protocol.md, Trust Layer).

Client-side policies, all computable from public sites plus the client's own
data:

- `loewner_clip`         - the per-direction confidence bound (Finding 7):
                           Λ_n ⪯ c · Λ_others in the Loewner order,
                           implied mean preserved.
- `Corroboration`       - one-peer-one-vote weighted median + MAD, judged in
                           each site's own eigenbasis; never weighted by
                           declared precision.
- `causal_span_residual`/`novelty_weight` - the replay defence: how much of a
                           site is unreachable from strictly older sites.
- `resolve_attestation_prior` - (a, b) from live attestations at the reader's
                           own kappa_r.
"""

from typing import Optional

import numpy as np

from .types import Eta, FAMILY_MVNORMAL, FAMILY_NORMAL, ONE_BLOCK_FAMILIES


class TrustTable:
    """Per-peer, per-group inclusion probabilities, persisted as JSON."""

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._p: dict[str, dict[str, float]] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        import json
        import os

        if os.path.exists(path):
            with open(path) as f:
                self._p = json.load(f)

    def save(self, path: Optional[str] = None) -> None:
        import json

        path = path or self._path
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self._p, f, indent=2)

    def get_p(self, peer: str, group: str, default: float = 0.5) -> float:
        return self._p.get(peer, {}).get(group, default)

    def set_p(self, peer: str, group: str, p: float) -> None:
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p must lie strictly in (0,1)")
        self._p.setdefault(peer, {})[group] = float(p)

    def get_all_p(self, peer: str, groups: list[str]) -> dict[str, float]:
        return {g: self.get_p(peer, g) for g in groups}

    def peers(self) -> list[str]:
        return list(self._p.keys())

    def reset(self, peer: Optional[str] = None, group: Optional[str] = None) -> None:
        if peer is None:
            self._p.clear()
        elif group is None:
            self._p.pop(peer, None)
        else:
            self._p.get(peer, {}).pop(group, None)

    def matrix(self, peers: list[str], groups: list[str]) -> dict[str, dict[str, float]]:
        return {p: {g: self.get_p(p, g) for g in groups} for p in peers}


# --------------------------------------------------------------------------
# Loewner confidence bound
# --------------------------------------------------------------------------


def loewner_clip(site: Eta, others: Eta, cap: float = 3.0) -> Eta:
    """Bound site's precision per direction at `cap` × the other peers'.

    `others` is the composed contribution of the evaluator's *other* peers,
    excluding the caller and this peer; a site that counts toward its own
    budget bounds nothing.  The implied mean is preserved: the peer still
    says what it thinks, with bounded confidence.  Returns an unmodified copy
    when the site is already inside the bound.  Only `mvnormal` and `normal`
    groups carry a precision block to bound; others are returned unchanged.
    """
    if site.family == FAMILY_MVNORMAL:
        return _loewner_clip_matrix(site, -2.0 * others.Lam, cap)
    if site.family == FAMILY_NORMAL:
        return _loewner_clip_scalar(site, -2.0 * others.Lam, cap)
    return site.copy()


def _loewner_clip_matrix(site: Eta, ref_prec: np.ndarray, c: float) -> Eta:
    """Cap Lam_n in the metric of ref:  Λ' = L W Lᵀ with eigvals ≤ c."""
    L = np.linalg.cholesky(ref_prec)
    Li = np.linalg.inv(L)
    W = Li @ (-2.0 * site.Lam) @ Li.T
    W = (W + W.T) / 2.0
    D, U = np.linalg.eigh(W)
    if float(D.max()) <= c:
        return site.copy()
    Wc = (U * np.minimum(D, c)) @ U.T
    prec_c = L @ Wc @ L.T
    prec_c = (prec_c + prec_c.T) / 2.0
    m = np.linalg.pinv(-2.0 * site.Lam) @ site.h  # sites are rank-deficient
    return Eta(FAMILY_MVNORMAL, prec_c @ m, -prec_c / 2.0)


def _loewner_clip_scalar(site: Eta, ref_prec: np.ndarray, c: float) -> Eta:
    """Elementwise cap on precision; implied mean preserved."""
    site_prec = -2.0 * site.Lam
    clipped_prec = np.minimum(site_prec, c * np.maximum(ref_prec, 0.0))
    ratio = np.where(site_prec != 0, clipped_prec / site_prec, 1.0)
    return Eta(FAMILY_NORMAL, site.h * ratio, -clipped_prec / 2.0)


# --------------------------------------------------------------------------
# Corroboration
# --------------------------------------------------------------------------


def _wmedian(P: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Weighted median down axis 0, independently per column."""
    order = np.argsort(P, axis=0)
    cw = np.cumsum(np.take_along_axis(wts, order, axis=0), axis=0)
    denom = np.where(cw[-1] > 0, cw[-1], 1.0)
    idx = (cw / denom >= 0.5).argmax(axis=0)
    return np.take_along_axis(
        np.take_along_axis(P, order, axis=0), idx[None, :], axis=0
    )[0]


class Corroboration:
    """Directional agreement between published sites for one parameter group.

    For an `mvnormal` group the "directions" are the site's own eigenvectors
    (each site publishes its observability).  For scalar families the
    directions are the components.  Votes are weighted only by trust already
    earned, and discrepancy is scaled by the pool's observed dispersion
    (MAD) - never by declared precision.
    """

    def __init__(self, sites: list[Eta]):
        self.N = len(sites)
        self.family = sites[0].family if sites else "normal"
        self._is_matrix = self.family == FAMILY_MVNORMAL
        self.means: list[np.ndarray] = []
        self.D: list[np.ndarray] = []
        self.V: list[np.ndarray] = []
        self.Q: list[np.ndarray] = []
        self.P: list[np.ndarray] = []
        for s in sites:
            if self._is_matrix:
                prec = -2.0 * s.Lam
                m = np.linalg.pinv(prec) @ s.h
                self.means.append(m)
                d, v = np.linalg.eigh(prec)
                self.D.append(d)
                self.V.append(v)
            else:
                self.means.append(s.h)
                self.D.append(None)
                self.V.append(None)
        for n in range(self.N):
            if self._is_matrix:
                v = self.V[n]
                q = np.array([np.sum(v * ((-2.0 * s.Lam) @ v), axis=0) for s in sites])
                p = np.array([v.T @ mu for mu in self.means])
            else:
                q = np.array([-2.0 * s.Lam for s in sites])
                p = np.array([s.h / (-2.0 * s.Lam) for s in sites])
            self.Q.append(q)
            self.P.append(p)

    def weights(self, n: int, p_incl: np.ndarray, exclude: int, t0: float) -> np.ndarray:
        """Per-direction credibility in [0,1] for peer n, judged by the
        evaluator's currently trusted peers.  One peer, one vote."""
        w = p_incl.copy()
        w[n] = 0.0
        if 0 <= exclude < self.N:
            w[exclude] = 0.0
        P = self.P[n]
        wts = np.repeat(w[:, None], P.shape[1], axis=1)
        med = _wmedian(P, wts)
        mad = _wmedian(np.abs(P - med), wts)
        scale = np.maximum(mad, 1e-9)
        t = np.abs(P[n] - med) / scale
        return 1.0 / (1.0 + (t / t0) ** 2)

    def apply(self, n: int, site: Eta, s: np.ndarray) -> Eta:
        """Rescale precision per direction, preserving the implied mean."""
        if self._is_matrix:
            v, d = self.V[n], self.D[n]
            prec = (v * (d * s)) @ v.T
            prec = (prec + prec.T) / 2.0
            return Eta(FAMILY_MVNORMAL, prec @ self.means[n], -prec / 2.0)
        lam = site.Lam * s
        h = site.h * s
        return Eta(self.family, h, lam)


def corroborate_sites(
    sites: list[Eta], p_incl: np.ndarray, exclude: int, t0: float
) -> list[Eta]:
    """Apply corroboration to every peer except the evaluator."""
    corr = Corroboration(sites)
    out = []
    for n in range(len(sites)):
        if n == exclude:
            out.append(sites[n].copy())
        else:
            w = corr.weights(n, p_incl, exclude, t0)
            out.append(corr.apply(n, sites[n], w))
    return out


# --------------------------------------------------------------------------
# Novelty / span residual
# --------------------------------------------------------------------------


def _flat_vec(eta: Eta) -> np.ndarray:
    """Flatten a site to a vector whose Euclidean norm is the natural one:
    h stacked on the (upper triangle of the) precision, off-diagonals scaled
    by sqrt(2) for the matrix case so Frobenius norm is preserved."""
    if eta.family == FAMILY_MVNORMAL:
        d = eta.dim
        iu = np.triu_indices(d)
        scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        tri = eta.Lam[iu] * scale
        return np.concatenate([eta.h, tri])
    if eta.family in ONE_BLOCK_FAMILIES:
        return eta.h
    return np.concatenate([eta.h, eta.Lam])


def _residual(sites: list[Eta], order: Optional[np.ndarray]) -> np.ndarray:
    V = np.array([_flat_vec(s) for s in sites])
    out = np.zeros(len(sites))
    for n in range(len(sites)):
        if order is None:
            A = np.delete(V, n, axis=0).T
        else:
            older = np.where(order < order[n])[0]
            if len(older) == 0:
                out[n] = 1.0  # nothing preceded it
                continue
            A = V[older].T
        b = V[n]
        nb = float(np.linalg.norm(b))
        if nb == 0.0:
            out[n] = 0.0
            continue
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        out[n] = float(np.linalg.norm(b - A @ c) / nb)
    return out


def span_residual(sites: list[Eta]) -> np.ndarray:
    """Symmetric test: how much of each site is NOT a linear combination of
    the others.  Flags author and copier alike (see protocol.md)."""
    return _residual(sites, None)


def causal_span_residual(sites: list[Eta], order: np.ndarray) -> np.ndarray:
    """Project each site onto only the strictly older sites' span.

    `order` must come from something the publisher cannot backdate (first-seen
    order).  Only the copier is flagged then.
    """
    return _residual(sites, np.asarray(order))


def novelty_weight(sites: list[Eta], order: np.ndarray, mode: str = "median") -> np.ndarray:
    """Per-site multiplier on p: the causal residual, optionally normalised by
    the median to remove the first-mover decay."""
    r = causal_span_residual(sites, order)
    if mode == "raw":
        return np.clip(r, 0.0, 1.0)
    ref = float(np.median(r))
    if ref <= 1e-9:
        return np.ones(len(sites))
    return np.clip(r / ref, 0.0, 1.0)


# --------------------------------------------------------------------------
# Attestations
# --------------------------------------------------------------------------


def resolve_attestation_prior(
    attestations: list[tuple[str, str, float]],  # (attester, target, p)
    p_attester: dict[str, float],                # evaluator's p for each attester
    target: str,
    kappa_r: float = 4.0,
) -> tuple[float, float]:
    """(a, b) for `target` from live attestations, reader strength κ_r.

    Each attestation p_{B→C} contributes p_{A→B}·κ_r·(p, 1−p), so a single
    contributor supplies at most κ_r pseudo-counts and no clip is required.
    Absent attestations resolve to (1, 1).
    """
    a, b = 1.0, 1.0
    for attester, tgt, p in attestations:
        if tgt != target:
            continue
        w = p_attester.get(attester, 0.5)
        if w <= 1e-9:
            continue
        a += w * kappa_r * p
        b += w * kappa_r * (1.0 - p)
    return a, b