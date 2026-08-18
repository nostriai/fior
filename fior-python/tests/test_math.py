"""Tests for FIOR math module."""

import numpy as np
import pytest
from fior.types import Eta
from fior.math import log_partition, compose_prior, bmr_delta_f, p_from


class TestLogPartition:
    """Tests for log-partition function."""
    
    def test_normal_basic(self):
        """Test A(η) for Normal family."""
        eta = Eta(h=np.array([1.0]), Lam=np.array([-0.5]))
        result = log_partition(eta, "normal")
        # A(η) = -h²/(4*Lam) - 0.5*log(-Lam)
        expected = -1.0 / (4 * -0.5) - 0.5 * np.log(0.5)
        assert abs(result - expected) < 1e-10
    
    def test_normal_multidim(self):
        """Test A(η) for multi-dimensional Normal."""
        eta = Eta(h=np.array([1.0, 2.0]), Lam=np.array([-0.5, -1.0]))
        result = log_partition(eta, "normal")
        # Sum over dimensions
        expected = (
            -1.0 / (4 * -0.5) - 0.5 * np.log(0.5)
            + -4.0 / (4 * -1.0) - 0.5 * np.log(1.0)
        )
        assert abs(result - expected) < 1e-10


class TestComposePrior:
    """Tests for prior composition."""
    
    def test_empty_sites(self):
        """Test composition with no sites returns eta0."""
        eta0 = Eta(h=np.array([0.0]), Lam=np.array([-1.0]))
        result = compose_prior(eta0, [], [])
        assert np.allclose(result.h, eta0.h)
        assert np.allclose(result.Lam, eta0.Lam)
    
    def test_single_site_full_weight(self):
        """Test composition with single site at full weight."""
        eta0 = Eta(h=np.array([0.0]), Lam=np.array([-1.0]))
        site = Eta(h=np.array([1.0]), Lam=np.array([-0.5]))
        
        class MockSite:
            def __init__(self, eta):
                self.delta_eta = eta
        
        result = compose_prior(eta0, [MockSite(site)], [1.0])
        assert np.allclose(result.h, [1.0])
        assert np.allclose(result.Lam, [-1.5])
    
    def test_single_site_half_weight(self):
        """Test composition with single site at half weight."""
        eta0 = Eta(h=np.array([0.0]), Lam=np.array([-1.0]))
        site = Eta(h=np.array([2.0]), Lam=np.array([-2.0]))
        
        class MockSite:
            def __init__(self, eta):
                self.delta_eta = eta
        
        result = compose_prior(eta0, [MockSite(site)], [0.5])
        assert np.allclose(result.h, [1.0])
        assert np.allclose(result.Lam, [-2.0])


class TestBmrDeltaF:
    """Tests for BMR log Bayes factor."""
    
    def test_identical_posteriors(self):
        """Test ΔF = 0 when posteriors are identical."""
        eta = Eta(h=np.array([1.0]), Lam=np.array([-0.5]))
        delta_f = bmr_delta_f(eta, eta, eta, eta, "normal")
        assert abs(delta_f) < 1e-10
    
    def test_inclusion_improves(self):
        """Test ΔF > 0 when inclusion improves evidence."""
        # Prior without peer
        eta_p_minus = Eta(h=np.array([0.0]), Lam=np.array([-1.0]))
        # Local data
        L_A = Eta(h=np.array([1.0]), Lam=np.array([-0.5]))
        # Peer contribution
        delta_eta = Eta(h=np.array([0.5]), Lam=np.array([-0.25]))
        
        eta_p_plus = eta_p_minus + delta_eta
        eta_q_minus = eta_p_minus + L_A
        eta_q_plus = eta_p_plus + L_A
        
        delta_f = bmr_delta_f(eta_q_plus, eta_p_minus, eta_q_minus, eta_p_plus, "normal")
        # Should be positive since adding peer helps
        assert delta_f > 0


class TestPFrom:
    """Tests for p resolution."""
    
    def test_zero_delta_f(self):
        """Test p = β when ΔF = 0."""
        p = p_from(0.0, 0.5)
        assert abs(p - 0.5) < 1e-10
    
    def test_positive_delta_f(self):
        """Test p > β when ΔF > 0."""
        p = p_from(2.0, 0.5)
        assert p > 0.5
    
    def test_negative_delta_f(self):
        """Test p < β when ΔF < 0."""
        p = p_from(-2.0, 0.5)
        assert p < 0.5
    
    def test_extreme_positive(self):
        """Test p → 1 for large positive ΔF."""
        p = p_from(100.0, 0.5)
        assert p > 0.99
    
    def test_extreme_negative(self):
        """Test p → 0 for large negative ΔF."""
        p = p_from(-100.0, 0.5)
        assert p < 0.01
