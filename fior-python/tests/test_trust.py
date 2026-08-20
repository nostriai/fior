"""Trust layer tests: the Loewner bound, corroboration invariants, the
causal span test, and attestation prior resolution."""

import numpy as np
import pytest

from fior import Eta, TrustTable, loewner_clip, Corroboration, novelty_weight
from fior.trust import resolve_attestation_prior, causal_span_residual, span_residual


def test_loewner_matrix_bound_and_mean_preserved():
    site = Eta("mvnormal", np.array([10.0, 0.0]), np.array([[-50.0, 0.0], [0.0, -1.0]]))
    ref = Eta("mvnormal", np.array([0.0, 0.0]), np.array([[-1.0, 0.0], [0.0, -1.0]]))
    clipped = loewner_clip(site, ref, cap=3.0)
    L = np.linalg.cholesky(-2.0 * ref.Lam)
    Li = np.linalg.inv(L)
    W = Li @ (-2.0 * clipped.Lam) @ Li.T
    ev = np.linalg.eigvalsh((W + W.T) / 2.0)
    assert ev.max() <= 3.0 + 1e-6
    m = np.linalg.pinv(-2.0 * site.Lam) @ site.h
    assert np.allclose(clipped.h, (-2.0 * clipped.Lam) @ m, atol=1e-8)


def test_loewner_clip_passing_site_unchanged():
    site = Eta("mvnormal", np.array([1.0]), np.array([[-1.0]]))
    ref = Eta("mvnormal", np.array([0.0]), np.array([[-2.0]]))
    clipped = loewner_clip(site, ref, cap=10.0)
    assert np.allclose(clipped.h, site.h) and np.allclose(clipped.Lam, site.Lam)


def test_loewner_scalar_normal_elementwise():
    site = Eta("normal", np.array([1.0, 10.0]), np.array([-0.1, -100.0]))
    ref = Eta("normal", np.array([0.0, 0.0]), np.array([-0.5, -10.0]))
    clipped = loewner_clip(site, ref, cap=1.0)
    # precisions Λ = -2η2: site claims (0.2, 200), ref (1.0, 20) -> capped at (0.2, 20)
    assert np.allclose(-2.0 * clipped.Lam, [0.2, 20.0])
    # implied mean preserved both dims
    m = -site.h / (2 * site.Lam)
    assert np.allclose(-clipped.h / (2 * clipped.Lam), m, atol=1e-10)


def test_corroboration_ignores_declared_precision():
    # inflating one peer's precision must not change another peer's verdict
    sites = [
        Eta("mvnormal", np.array([1.0, 2.0]), np.array([[-1.0, 0.1], [0.1, -2.0]])),
        Eta("mvnormal", np.array([0.0, 0.0]), np.array([[-0.5, 0.0], [0.0, -0.5]])),
        Eta("mvnormal", np.array([1.1, 2.1]), np.array([[-1.2, 0.1], [0.1, -1.8]])),
    ]
    inflated = [Eta(s.family, s.h.copy(), s.Lam * 1000.0 if i == 1 else s.Lam.copy())
                for i, s in enumerate(sites)]
    c1 = Corroboration(sites)
    c2 = Corroboration(inflated)
    p = np.array([0.5, 0.5, 0.5])
    assert np.allclose(c1.weights(0, p, -1, 15.0), c2.weights(0, p, -1, 15.0), atol=1e-9)


def test_corroboration_weights_are_credibilities():
    sites = [
        Eta("normal", np.array([1.0]), np.array([-1.0])),
        Eta("normal", np.array([0.1]), np.array([-0.5])),
        Eta("normal", np.array([2.0]), np.array([-2.0])),
    ]
    c = Corroboration(sites)
    s = c.weights(0, np.array([0.5, 0.5, 0.5]), -1, 15.0)
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-12


def test_causal_span_flags_copier_not_author():
    sites = [
        Eta("normal", np.array([1.0, 2.0]), np.array([-1.0, -2.0])),
        Eta("normal", np.array([3.0, 4.0]), np.array([-1.5, -1.0])),
        Eta("normal", np.array([1.0, 2.0]), np.array([-1.0, -2.0])),  # copy of site 0
    ]
    order = np.array([0.0, 1.0, 2.0])
    r = causal_span_residual(sites, order)
    assert r[2] < 1e-9  # copier flagged
    assert r[0] > 0.5 and r[1] > 0.5  # originals clean
    sym = span_residual(sites)
    assert sym[0] < 1e-9  # symmetric test also flags the author


def test_novelty_weight_is_fraction():
    from fior.trust import novelty_weight
    sites = [
        Eta("mvnormal", np.array([1.0, 2.0]), np.array([[-1.0, 0.0], [0.0, -2.0]])),
        Eta("mvnormal", np.array([1.0, 2.0]), np.array([[-1.0, 0.0], [0.0, -2.0]])),
    ]
    w = novelty_weight(sites, np.array([0.0, 1.0]))
    assert w.max() <= 1.0 and w.min() >= 0.0
    assert w[1] < 1e-9


def test_attestation_prior_bounded_by_kappa():
    atts = [("attester1", "target", 0.0), ("attester2", "target", 1.0)]
    p_att = {"attester1": 1.0, "attester2": 1.0}
    a, b = resolve_attestation_prior(atts, p_att, "target", kappa_r=4.0)
    assert abs((a - 1.0) + (b - 1.0) - 8.0) < 1e-12  # two contributors, 4 each


def test_attestation_discounted_by_trust_in_attester():
    atts = [("attacker", "target", 0.0)]
    trusted = resolve_attestation_prior(atts, {"attacker": 1.0}, "target", 4.0)
    ignored = resolve_attestation_prior(atts, {"attacker": 0.0}, "target", 4.0)
    assert ignored == (1.0, 1.0)
    assert trusted[1] > 1.0


def test_trust_table_persistence(tmp_path):
    path = str(tmp_path / "trust.json")
    t = TrustTable(path)
    t.set_p("peer", "group", 0.8)
    t.save()
    t2 = TrustTable(path)
    assert t2.get_p("peer", "group") == 0.8
    with pytest.raises(ValueError):
        t.set_p("peer", "g0", 1.0)  # strict (0,1)


def test_trust_table_reset():
    t = TrustTable()
    t.set_p("a", "g", 0.7)
    t.reset("a", "g")
    assert t.get_p("a", "g") == 0.5