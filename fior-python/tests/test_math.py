"""Math tests for FIOR: log-partition correctness for every family,
composition with the domain check, and unit-weight BMR."""

import math

import numpy as np
import pytest

from fior import Eta, Group, value_count
from fior.math import (
    log_partition,
    compose_prior,
    bmr_delta_f,
    p_from,
    logistic,
)


def test_normal_log_partition():
    eta = Eta("normal", np.array([1.0, 2.0]), np.array([-0.5, -1.0]))
    got = log_partition(eta)
    want = -1.0 / (4 * -0.5) - 0.5 * math.log(0.5) + -4.0 / (4 * -1.0) - 0.5 * math.log(1.0)
    assert abs(got - want) < 1e-12


def test_normal_outside_domain_raises():
    with pytest.raises(ValueError):
        log_partition(Eta("normal", np.array([0.0]), np.array([1.0])))


def test_mvnormal_log_partition_matches_reference_formula():
    # A = 0.5 h' Λ⁻¹ h - 0.5 ln det Λ,  Λ = -2 η2
    h = np.array([0.5, -1.0, 2.0])
    lam = np.array([[0.4, 0.1, 0.0], [0.1, 0.3, 0.05], [0.0, 0.05, 0.2]])
    eta = Eta("mvnormal", h, -0.5 * lam)
    prec = -2.0 * eta.Lam
    L = np.linalg.cholesky(prec)
    z = np.linalg.solve(L, h)
    want = 0.5 * (z @ z) - float(np.log(np.diag(L)).sum())
    assert abs(log_partition(eta) - want) < 1e-12


def test_gamma_log_partition_correct():
    for h, lam in [(1.0, -2.0), (3.0, -0.5), (0.0, -1.0), (5.0, -0.1)]:
        alpha, beta = h + 1, -lam
        want = math.lgamma(alpha) - alpha * math.log(beta)
        got = log_partition(Eta("gamma", np.array([h]), np.array([lam])))
        assert abs(got - want) < 1e-12, (h, lam)


def test_beta_log_partition_correct():
    for h, lam in [(0.5, 2.0), (3.0, 1.0), (0.0, 0.0)]:
        a, b = h + 1, lam + 1
        want = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        got = log_partition(Eta("beta", np.array([h]), np.array([lam])))
        assert abs(got - want) < 1e-12


def test_dirichlet_log_partition_correct():
    h = np.array([0.5, 3.0, 1.0])
    alpha = h + 1
    want = sum(math.lgamma(a) for a in alpha) - math.lgamma(alpha.sum())
    got = log_partition(Eta("dirichlet", h))
    assert abs(got - want) < 1e-12


def test_categorical_log_partition_is_logsumexp():
    h = np.array([3.0, -2.0, 0.1, 1.0])
    want = math.log(np.exp(h).sum())
    assert abs(log_partition(Eta("cat", h)) - want) < 1e-12


def test_compose_weights_and_domain_check():
    eta0 = Eta("normal", np.array([0.0]), np.array([-1.0]))
    s1 = Eta("normal", np.array([1.0]), np.array([-0.5]))
    s2 = Eta("normal", np.array([-2.0]), np.array([-0.25]))
    r = compose_prior(eta0, [s1, s2], [0.5, 0.5])
    assert np.allclose(r.h, [0.5 - 1.0])
    assert np.allclose(r.Lam, [-1.0 - 0.25 - 0.125])

    bad = Eta("normal", np.array([10.0]), np.array([2.0]))  # invalid site
    with pytest.raises(ValueError):
        compose_prior(eta0, [bad], [1.0])


def test_compose_domain_check_raises():
    eta0 = Eta("normal", np.array([0.0]), np.array([-1.0]))
    bad = Eta("normal", np.array([10.0]), np.array([2.0]))  # invalid site (η2 > 0)
    with pytest.raises(ValueError):
        compose_prior(eta0, [bad], [1.0])


def test_bmr_unit_weight_positive_for_agreeing_peer():
    eta0 = Eta("normal", np.array([0.0]), np.array([-1.0]))
    peer = Eta("normal", np.array([0.5]), np.array([-0.5]))
    lik = Eta("normal", np.array([0.2]), np.array([-0.3]))
    p_full = eta0 + peer
    dF = bmr_delta_f(p_full + lik, p_full, peer)
    assert dF > 0
    # the finding-2 fixed point: a zero-weight increment carries no evidence
    dF2 = bmr_delta_f(p_full + lik, p_full, 0.0 * peer)
    assert abs(dF2) < 1e-12


def test_p_from_is_sigmoid_of_delta_f_plus_log_odds():
    for dF in (-50.0, -1.0, 0.0, 1.0, 50.0):
        for a, b in ((1.0, 1.0), (3.0, 1.0), (1.0, 4.0)):
            got = p_from(dF, a, b)
            want = 1.0 / (1.0 + (b / a) * math.exp(-dF))
            assert abs(got - want) < 1e-12


def test_p_from_no_attestations_is_sigmoid():
    for dF in (-20.0, 0.0, 20.0):
        assert abs(p_from(dF, 1.0, 1.0) - logistic(dF)) < 1e-12


def test_group_value_count_rules():
    assert value_count("normal", 3) == 6
    assert value_count("mvnormal", 3) == 9
    assert value_count("gamma", 2) == 4
    assert value_count("beta", 2) == 4
    assert value_count("dirichlet", 4) == 4
    assert value_count("cat", 4) == 4
    with pytest.raises(ValueError):
        value_count("nonsense", 3)