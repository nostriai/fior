"""Nostr event construction and signing for FIOR.

Uses noble-secp256k1 for BIP-340 signatures.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

# Try to import noble-secp256k1, fall back to basic implementation
try:
    from noble_secp256k1 import sign, getPublicKey
    HAS_NOBLE = True
except ImportError:
    HAS_NOBLE = False


@dataclass
class Event:
    """Signed Nostr event."""
    kind: int
    tags: list[list[str]]
    content: str
    created_at: int
    pubkey: str  # hex
    id: str = ""  # event ID (hex)
    sig: str = ""  # signature (hex)
    
    def to_dict(self) -> dict:
        """Convert to Nostr event dict."""
        return {
            "id": self.id,
            "pubkey": self.pubkey,
            "created_at": self.created_at,
            "kind": self.kind,
            "tags": self.tags,
            "content": self.content,
            "sig": self.sig,
        }
    
    def serialize(self) -> str:
        """Serialize for signing (unsigned event)."""
        return json.dumps([
            0,
            self.pubkey,
            self.created_at,
            self.kind,
            self.tags,
            self.content,
        ], separators=(",", ":"))


def compute_event_id(event: Event) -> str:
    """Compute event ID (SHA-256 of serialized event)."""
    import hashlib
    serialized = event.serialize()
    return hashlib.sha256(serialized.encode()).hexdigest()


def sign_event(event: Event, private_key: str) -> Event:
    """Sign a Nostr event with BIP-340.
    
    Args:
        event: Unsigned event
        private_key: Private key as hex string
    
    Returns:
        Signed event with id and sig populated
    """
    # Compute event ID
    event.id = compute_event_id(event)
    
    if HAS_NOBLE:
        # Use noble-secp256k1
        sig_bytes = sign(bytes.fromhex(event.id), bytes.fromhex(private_key))
        event.sig = sig_bytes.hex()
    else:
        # Fallback: use hashlib for demo (NOT secure)
        import hashlib
        event.sig = hashlib.sha256(event.id.encode()).hexdigest()
    
    return event


def create_model_card_event(
    private_key: str,
    model_id: str,
    title: str,
    version: int,
    onnx_blob: str,
    groups: list,
    eta0_blob: Optional[str] = None,
    blossom_servers: Optional[list[str]] = None,
    summary: Optional[str] = None,
    ttl: Optional[int] = None,
) -> Event:
    """Create a Model Card event (30100).
    
    Args:
        private_key: Signing key
        model_id: Model identifier (d tag)
        title: Human-readable name (t tag)
        version: Model version (v tag)
        onnx_blob: ONNX blob reference (o tag)
        groups: List of (name, family, initializers) tuples
        eta0_blob: Base prior blob reference (x tag)
        blossom_servers: Blossom server URLs (b tag)
        summary: Short description (s tag)
        ttl: Lifetime in seconds (l tag)
    
    Returns:
        Signed Event
    """
    from noble_secp256k1 import getPublicKey
    
    pubkey = getPublicKey(bytes.fromhex(private_key)).hex()
    
    tags = [
        ["d", model_id],
        ["t", title],
        ["v", str(version)],
        ["o", onnx_blob],
    ]
    
    if summary:
        tags.append(["s", summary])
    
    for name, family, initializers in groups:
        tags.append(["g", name, family] + initializers)
    
    if eta0_blob:
        tags.append(["x", eta0_blob])
    
    if blossom_servers:
        tags.append(["b"] + blossom_servers)
    
    if ttl:
        tags.append(["l", str(ttl)])
    
    event = Event(
        kind=30100,
        tags=tags,
        content="{}",
        created_at=int(time.time()),
        pubkey=pubkey,
    )
    
    return sign_event(event, private_key)


def create_site_contribution_event(
    private_key: str,
    model_id: str,
    model_version: int,
    delta_eta_blob: str,
    cavity_members: list,
    samples: Optional[int] = None,
    free_energy: Optional[float] = None,
    duration_sec: Optional[float] = None,
    expiration: Optional[int] = None,
) -> Event:
    """Create a Site Contribution event (30101).
    
    Args:
        private_key: Signing key
        model_id: Model identifier (d tag)
        model_version: Model card version (v tag)
        delta_eta_blob: Blob reference (x tag)
        cavity_members: List of (pubkey, event_id, p_vector) tuples (m tags)
        samples: Training samples (content)
        free_energy: ELBO value (content)
        duration_sec: Training duration (content)
        expiration: NIP-40 expiration (E tag)
    
    Returns:
        Signed Event
    """
    from noble_secp256k1 import getPublicKey
    
    pubkey = getPublicKey(bytes.fromhex(private_key)).hex()
    
    # Build a coordinate reference
    # Format: "30100:<creator_hex>:<model_id>"
    # For simplicity, use model_id as the reference
    a_ref = f"30100:{pubkey}:{model_id}"
    
    tags = [
        ["d", model_id],
        ["a", a_ref, "", "model"],
        ["v", str(model_version)],
        ["x", delta_eta_blob],
    ]
    
    # Add provenance (m tags) and member pubkeys (p tags)
    for member_pubkey, member_event_id, p_vector in cavity_members:
        tags.append(["m", member_pubkey, member_event_id, p_vector])
        tags.append(["p", member_pubkey])
    
    if expiration:
        tags.append(["E", str(expiration)])
    
    # Content JSON
    content = {}
    if samples is not None:
        content["samples"] = samples
    if free_energy is not None:
        content["free_energy"] = free_energy
    if duration_sec is not None:
        content["duration_sec"] = duration_sec
    
    event = Event(
        kind=30101,
        tags=tags,
        content=json.dumps(content) if content else "{}",
        created_at=int(time.time()),
        pubkey=pubkey,
    )
    
    return sign_event(event, private_key)


def create_trust_attestation_event(
    private_key: str,
    target_pubkey: str,
    scope: str,
    inclusion_probability: float,
    model_ref: Optional[str] = None,
    expiration: Optional[int] = None,
) -> Event:
    """Create a Trust Attestation event (30102).
    
    Args:
        private_key: Signing key
        target_pubkey: Target peer's pubkey (hex)
        scope: "peer", "peer:model", or "peer:model:group"
        inclusion_probability: p value in (0,1)
        model_ref: Model card coordinate (a tag)
        expiration: NIP-40 expiration (E tag)
    
    Returns:
        Signed Event
    """
    from noble_secp256k1 import getPublicKey
    
    pubkey = getPublicKey(bytes.fromhex(private_key)).hex()
    
    # Build d tag from scope
    if scope == "peer":
        d_value = target_pubkey
    else:
        d_value = f"{target_pubkey}:{scope}"
    
    tags = [
        ["d", d_value],
        ["p", target_pubkey],
        ["i", str(inclusion_probability)],
    ]
    
    if model_ref:
        tags.append(["a", model_ref, "", "model"])
    
    if expiration:
        tags.append(["E", str(expiration)])
    
    event = Event(
        kind=30102,
        tags=tags,
        content="{}",
        created_at=int(time.time()),
        pubkey=pubkey,
    )
    
    return sign_event(event, private_key)


async def publish_event(event: Event, relay_url: str) -> bool:
    """Publish event to a Nostr relay via WebSocket.
    
    Args:
        event: Signed event to publish
        relay_url: WebSocket URL of the relay
    
    Returns:
        True if published successfully
    """
    import websockets
    
    async with websockets.connect(relay_url) as ws:
        message = json.dumps(["EVENT", event.to_dict()])
        await ws.send(message)
        
        response = await ws.recv()
        result = json.loads(response)
        
        return result[0] == "OK"
