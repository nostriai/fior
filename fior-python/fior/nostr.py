"""Nostr event construction and signing for FIOR.

Signing is BIP-340 Schnorr via the vendored pure-Python `bip340` module; there
is no native or external crypto dependency.  Event tags follow protocol.md,
including the `a`/`o`/`x` shapes that specify the model card creator and blob
metadata (hash, encoding, size, server hints).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from . import bip340


@dataclass
class Event:
    """Signed Nostr event."""

    kind: int
    tags: list[list[str]]
    content: str
    created_at: int
    pubkey: str = ""
    id: str = ""
    sig: str = ""

    def to_dict(self) -> dict:
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
        """NIP-01 serialization for hashing (no whitespace)."""
        return json.dumps(
            [0, self.pubkey, self.created_at, self.kind, self.tags, self.content],
            separators=(",", ":"),
        )

    def verify(self) -> bool:
        """Check the event id and the BIP-340 signature over it."""
        import hashlib

        if self.id != hashlib.sha256(self.serialize().encode()).hexdigest():
            return False
        if len(self.sig) != 128:
            return False
        return bip340.verify(bytes.fromhex(self.pubkey), bytes.fromhex(self.id), bytes.fromhex(self.sig))


def compute_event_id(event: Event) -> str:
    import hashlib

    return hashlib.sha256(event.serialize().encode()).hexdigest()


def sign_event(event: Event, private_key_hex: str, aux: Optional[bytes] = None) -> Event:
    """Fill in pubkey (from the key), id, and a BIP-340 signature."""
    sk = bytes.fromhex(private_key_hex)
    event.pubkey = bip340.get_pubkey(sk).hex()
    event.id = compute_event_id(event)
    sig = bip340.sign(sk, bytes.fromhex(event.id), aux=aux)
    event.sig = sig.hex()
    return event


def blob_ref(sha256: str, enc: str, size: int, servers: list[str]) -> list[str]:
    """The tail of an `x`/`o` tag: [<sha256>, <enc>, <bytes>, <server>...]."""
    return [sha256, enc, str(size)] + list(servers)


def verification(event: dict) -> bool:
    """Events arriving from a relay are dicts; verify id and signature."""
    e = Event(
        kind=event["kind"],
        tags=event["tags"],
        content=event["content"],
        created_at=event["created_at"],
        pubkey=event["pubkey"],
        id=event["id"],
        sig=event["sig"],
    )
    return e.verify()


# --------------------------------------------------------------------------
# Kind 30100 - Model Card
# --------------------------------------------------------------------------


def create_model_card_event(
    private_key_hex: str,
    *,
    model_id: str,
    title: str,
    version: int,
    onnx_ref: list,
    groups: list[tuple[str, str, list[str]]],
    eta0_ref: Optional[list] = None,
    blossom_servers: Optional[list[str]] = None,
    summary: Optional[str] = None,
    ttl: Optional[int] = None,
) -> Event:
    """Create and sign a model card.

    `onnx_ref`/`eta0_ref` are `blob_ref(...)` outputs: [hash, enc, size,
    servers...].  `groups` is [(name, family, [initializer...])].
    """
    tags = [
        ["d", model_id],
        ["t", title],
        ["v", str(version)],
    ]
    if summary:
        tags.append(["s", summary])
    tags.append(["o"] + onnx_ref)

    families_seen: set[str] = set()
    for name, family, inits in groups:
        tags.append(["g", name, family, *[str(i) for i in inits]])
        tags.append(["G", name])
        families_seen.add(family)
    for family in sorted(families_seen):
        tags.append(["F", family])

    if eta0_ref is not None:
        tags.append(["x"] + eta0_ref)
    if blossom_servers:
        tags.append(["b"] + blossom_servers)
    if ttl is not None:
        tags.append(["l", str(ttl)])

    event = Event(kind=30100, tags=tags, content="{}", created_at=int(time.time()))
    return sign_event(event, private_key_hex).to_dict()


# --------------------------------------------------------------------------
# Kind 30101 - Site Contribution
# --------------------------------------------------------------------------


def create_site_contribution_event(
    private_key_hex: str,
    *,
    model_id: str,
    model_version: int,
    model_coordinate: str,
    delta_eta_ref: list,
    cavity_members: Optional[list[tuple[str, str, list[float]]]] = None,
    expiration: Optional[int] = None,
    samples: Optional[int] = None,
    free_energy: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> dict:
    """Create and sign a Site Contribution (30101).

    `model_coordinate` is the model card's own coordinate
    (`30100:<creator>:<model>`), never the site author's pubkey.  Each cavity
    member gives (pubkey, exact_site_event_id, p_vector).
    """
    tags = [
        ["d", model_id],
        ["a", model_coordinate, "", "model"],
        ["v", str(model_version)],
        ["x"] + delta_eta_ref,
    ]
    for member_pubkey, member_event_id, p_vector in cavity_members or []:
        tags.append(["m", member_pubkey, member_event_id, ",".join(f"{p:.17g}" for p in p_vector)])
        tags.append(["p", member_pubkey])
        tags.append(["e", member_event_id])
    if expiration is not None:
        tags.append(["E", str(expiration)])

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
        content=json.dumps(content, separators=(",", ":")) if content else "{}",
        created_at=int(time.time()),
    )
    return sign_event(event, private_key_hex).to_dict()


# --------------------------------------------------------------------------
# Kind 30102 - Trust Attestation
# --------------------------------------------------------------------------


def create_trust_attestation_event(
    private_key_hex: str,
    *,
    target_pubkey: str,
    inclusion_probability: float,
    model_id: Optional[str] = None,
    group: Optional[str] = None,
    model_coordinate: Optional[str] = None,
    expiration: Optional[int] = None,
) -> dict:
    """A single scalar p for a peer, optionally scoped to model/group.

    The `d` tag is `<target>[:<model>[:<group>]]`.  The attestation carries
    the belief only: how strongly to hold it is the reader's `κ_r`.
    """
    if not 0.0 < inclusion_probability < 1.0:
        raise ValueError("inclusion probability must lie strictly in (0,1)")
    d = target_pubkey
    if model_id is not None:
        d += f":{model_id}"
        if group is not None:
            d += f":{group}"
    tags = [
        ["d", d],
        ["p", target_pubkey],
        ["i", f"{inclusion_probability:.17g}"],
    ]
    if model_id is not None and model_coordinate is not None:
        tags.append(["a", model_coordinate, "", "model"])
    if expiration is not None:
        tags.append(["E", str(expiration)])

    event = Event(
        kind=30102,
        tags=tags,
        content='{"basis":"bmr"}',
        created_at=int(time.time()),
    )
    return sign_event(event, private_key_hex).to_dict()


# --------------------------------------------------------------------------
# Kind 5 - Withdrawal
# --------------------------------------------------------------------------


def create_deletion_event(
    private_key_hex: str,
    target_ids: list[str],
    reason: str = "FIOR withdrawal",
) -> dict:
    tags = [["e", eid] for eid in target_ids]
    event = Event(
        kind=5,
        tags=tags,
        content=reason,
        created_at=int(time.time()),
    )
    return sign_event(event, private_key_hex).to_dict()


# --------------------------------------------------------------------------
# Relay I/O
# --------------------------------------------------------------------------


@dataclass
class PubResult:
    ok: bool
    event_id: str = ""
    reason: str = ""
    relay_msgs: list = field(default_factory=list)


async def publish_event(event_dict: dict, relay_url: str, timeout: float = 15.0) -> PubResult:
    """Publish one event; return relay acceptance and reason.

    Handles OK/NOTICE and tolerates a relay asking for AUTH by reporting it
    through reason rather than hanging.
    """
    import asyncio

    import websockets

    async with websockets.connect(relay_url, open_timeout=timeout) as ws:
        await ws.send(json.dumps(["EVENT", event_dict]))
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return PubResult(False, event_dict.get("id", ""), "timeout waiting for OK")
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except asyncio.TimeoutError:
                return PubResult(False, event_dict.get("id", ""), "timeout waiting for OK")
            if msg[0] == "OK":
                ok = len(msg) > 2 and msg[2] is True
                return PubResult(ok, msg[1], msg[3] if len(msg) > 3 else "", msg)
            if msg[0] == "NOTICE":
                continue
            if msg[0] == "AUTH":
                continue
    return PubResult(False, event_dict.get("id", ""), "relay closed without OK")


async def subscribe(relay_url: str, filters: list[dict], timeout: float = 10.0):
    """Open a subscription, yield (subscription_id, event_dict) until EOSE."""
    import asyncio

    import websockets

    sub_id = f"fior-{int(time.time()*1000)}"
    async with websockets.connect(relay_url, open_timeout=timeout) as ws:
        await ws.send(json.dumps(["REQ", sub_id] + filters))
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            except asyncio.TimeoutError:
                break
            if msg[0] == "EOSE":
                break
            if msg[0] == "EVENT" and len(msg) > 2:
                yield msg[1], msg[2]
            elif msg[0] == "NOTICE":
                continue
            elif msg[0] == "AUTH":
                continue