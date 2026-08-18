"""Tests for FIOR trust module."""

import numpy as np
import pytest
from fior.types import Eta, Site
from fior.trust import TrustTable, Corroboration, spectral_clip_site


class TestTrustTable:
    """Tests for TrustTable."""
    
    def test_default_p(self):
        """Test default p value."""
        table = TrustTable()
        assert table.get_p("peer1", "group1") == 0.5
    
    def test_set_get_p(self):
        """Test setting and getting p values."""
        table = TrustTable()
        table.set_p("peer1", "group1", 0.8)
        assert table.get_p("peer1", "group1") == 0.8
    
    def test_multiple_peers(self):
        """Test multiple peers and groups."""
        table = TrustTable()
        table.set_p("peer1", "group1", 0.9)
        table.set_p("peer1", "group2", 0.7)
        table.set_p("peer2", "group1", 0.6)
        
        assert table.get_p("peer1", "group1") == 0.9
        assert table.get_p("peer1", "group2") == 0.7
        assert table.get_p("peer2", "group1") == 0.6
    
    def test_list_peers(self):
        """Test listing peers."""
        table = TrustTable()
        table.set_p("peer1", "group1", 0.9)
        table.set_p("peer2", "group1", 0.6)
        
        peers = table.peers()
        assert "peer1" in peers
        assert "peer2" in peers
    
    def test_reset(self):
        """Test resetting p values."""
        table = TrustTable()
        table.set_p("peer1", "group1", 0.9)
        table.reset("peer1", "group1")
        assert table.get_p("peer1", "group1") == 0.5


class TestCorroboration:
    """Tests for Corroboration."""
    
    def test_no_trusted_sites(self):
        """Test with no trusted sites returns 1.0."""
        corr = Corroboration()
        
        site = Site(
            author="peer1",
            model_id="model1",
            model_version=1,
            delta_eta=Eta(h=np.array([1.0]), Lam=np.array([-1.0])),
        )
        
        score = corr.score(site, [], [])
        assert score == 1.0
    
    def test_agreeing_site(self):
        """Test site agreeing with trusted peers."""
        corr = Corroboration(threshold=3.0)
        
        # Trusted site
        trusted = Site(
            author="peer2",
            model_id="model1",
            model_version=1,
            delta_eta=Eta(h=np.array([1.0]), Lam=np.array([-1.0])),
        )
        
        # Site with similar precision
        site = Site(
            author="peer1",
            model_id="model1",
            model_version=1,
            delta_eta=Eta(h=np.array([1.1]), Lam=np.array([-1.05])),
        )
        
        score = corr.score(site, [trusted], [0.9])
        # Should be close to 1.0 (no penalty)
        assert score > 0.9


class TestSpectralClipSite:
    """Tests for Loewner cap."""
    
    def test_no_clipping_needed(self):
        """Test when site precision is within bounds."""
        site = Eta(h=np.array([1.0]), Lam=np.array([-1.0]))
        others = Eta(h=np.array([0.5]), Lam=np.array([-0.5]))
        
        clipped = spectral_clip_site(site, others, cap=3.0)
        
        # Precision ratio is 2.0, within cap of 3.0
        assert abs(clipped.Lam[0] - site.Lam[0]) < 1e-10
    
    def test_clipping_applied(self):
        """Test when site precision exceeds bounds."""
        site = Eta(h=np.array([10.0]), Lam=np.array([-10.0]))
        others = Eta(h=np.array([0.5]), Lam=np.array([-0.5]))
        
        clipped = spectral_clip_site(site, others, cap=3.0)
        
        # Precision ratio is 20.0, clipped to 3.0
        assert abs(clipped.Lam[0] - (-1.5)) < 1e-10
