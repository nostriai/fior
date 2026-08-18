"""Tests for FIOR nostr module."""

import json
import pytest
from fior.nostr import Event, compute_event_id, sign_event


class TestEvent:
    """Tests for Event class."""
    
    def test_to_dict(self):
        """Test converting event to dict."""
        event = Event(
            kind=30100,
            tags=[["d", "model1"]],
            content="{}",
            created_at=1234567890,
            pubkey="abc123",
        )
        
        d = event.to_dict()
        assert d["kind"] == 30100
        assert d["tags"] == [["d", "model1"]]
        assert d["pubkey"] == "abc123"
    
    def test_serialize(self):
        """Test serializing event for signing."""
        event = Event(
            kind=30100,
            tags=[["d", "model1"]],
            content="{}",
            created_at=1234567890,
            pubkey="abc123",
        )
        
        s = event.serialize()
        data = json.loads(s)
        
        assert data[0] == 0
        assert data[1] == "abc123"
        assert data[2] == 1234567890
        assert data[3] == 30100


class TestComputeEventId:
    """Tests for event ID computation."""
    
    def test_deterministic(self):
        """Test event ID is deterministic."""
        event = Event(
            kind=30100,
            tags=[["d", "model1"]],
            content="{}",
            created_at=1234567890,
            pubkey="abc123",
        )
        
        id1 = compute_event_id(event)
        id2 = compute_event_id(event)
        
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex


class TestSignEvent:
    """Tests for event signing."""
    
    def test_sign_populates_id_and_sig(self):
        """Test signing populates id and sig."""
        # Use a dummy private key (not real)
        private_key = "0000000000000000000000000000000000000000000000000000000000000001"
        
        event = Event(
            kind=30100,
            tags=[["d", "model1"]],
            content="{}",
            created_at=1234567890,
            pubkey="abc123",
        )
        
        signed = sign_event(event, private_key)
        
        assert len(signed.id) == 64
        assert len(signed.sig) == 64
