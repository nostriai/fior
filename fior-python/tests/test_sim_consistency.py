"""Cross-check the library's arithmetic against the reference simulation.

test/fior_sim.py is the source of every quantitative claim in protocol.md.
This locks the library to it: log-partition, BMR, p resolution, and the full
multi-round fixed point must produce the same numbers (the simulation's Eta
holds precision Λ PSD; the library's mvnormal Eta holds η2 = -Λ/2).
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "test"))

import fior_sim as sim  # noqa: E402

import fior  # noqa: E402


def _to_lib(e: sim.Eta) -> fior.Eta:
    """Convert a sim Eta (PSD precision Λ) to a library mvnormal Eta (η2)."""
    lam = -0.5 * e.Lam
    lam = (lam + lam.T) / 2.0
    return fior.Eta("mvnormal", e.h.copy(), lam)


def _small_cfg(**kw):
    base = dict(d=3, n_a=3, n_b=3, n_bad=1, rounds=4, honest_attest_frac=0.0,
                attack="random", spectral_cap=0.0, corroborate=0.0, novelty=False)
    base.update(kw)
    return sim.Config(**base)


def test_log_partition_identical():
    cfg = _small_cfg()
    w = sim.build_world(cfg)
    for e in w.sites + w.lik:
        assert abs(sim.log_partition(e) - fior.log_partition(_to_lib(e))) < 1e-10


def test_bmr_delta_f_identical():
    cfg = _small_cfg(rounds=1)
    w = sim.build_world(cfg)
    p = np.full((w.N, w.N), 0.5)
    np.fill_diagonal(p, 0.0)
    for a in range(w.N):
        prior = sim.compose_prior(w.eta0, w.sites, p[a], exclude=a)
        for n in range(w.N):
            if n == a:
                continue
            full = prior + (1.0 - p[a, n]) * w.sites[n]
            full_l = _to_lib(full)
            lprior_l = _to_lib(prior)
            contrib_l = _to_lib(w.sites[n])
            lik_l = _to_lib(w.lik[a])
            want = sim.bmr_delta_f(full + w.lik[a], full, w.sites[n])
            got = fior.bmr_delta_f(full_l + lik_l, full_l, contrib_l)
            assert abs(got - want) < 1e-8, (a, n, got, want)
        break  # one evaluator is enough for exactness


def test_full_round_loop_matches_sim():
    cfg = _small_cfg(rounds=6)
    res = sim.run(cfg)
    w = res["world"]
    p_sim = res["p"]
    honest = np.where(w.label < 2)[0]

    # replicate run() with the library's math
    eta0_l = _to_lib(w.eta0)
    sites_l = [_to_lib(s) for s in w.sites]
    lik_l = [_to_lib(l) for l in w.lik]
    N = w.N
    p = np.full((N, N), cfg.p_init)
    np.fill_diagonal(p, 0.0)
    bad = np.where(w.label == 2)[0]
    for i in bad:
        p[i, :] = 0.0
        p[i, bad] = 1.0
        p[i, i] = 0.0
    for _ in range(cfg.rounds):
        new_p = p.copy()
        for a in honest:
            prior = fior.compose_prior(eta0_l, [sites_l[n] for n in range(N) if n != a],
                                       [p[a, n] for n in range(N) if n != a])
            for n in range(N):
                if n == a:
                    continue
                full = prior + (1.0 - p[a, n]) * sites_l[n]
                dF = fior.bmr_delta_f(full + lik_l[a], full, sites_l[n])
                new_p[a, n] = fior.p_from(dF, 1.0, 1.0)
        p = new_p
    hon = np.ix_(honest, np.arange(N))
    assert np.allclose(p[hon], p_sim[hon], atol=1e-9), np.abs(p[hon]-p_sim[hon]).max()