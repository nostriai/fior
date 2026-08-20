"""Nostr event tests: BIP-340 conformance, event id canonicalization, tag
shapes per protocol.md (a/o/x/m/p/e/i), and event verification."""

import hashlib
import json

import pytest

from fior import bip340, nostr

VECTOR_MSG = "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89"
VECTOR_SEK = "B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF"
VECTOR_PUB = "DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659"
VECTOR_SIG = "6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE33418906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A"


def test_bip340_official_vectors_self_test():
    assert bip340.self_test()


def test_bip340_sign_matches_official_vector():
    sig = bip340.sign(
        bytes.fromhex(VECTOR_SEK), bytes.fromhex(VECTOR_MSG),
        aux=bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000001"),
    )
    assert sig.hex().upper() == VECTOR_SIG


def test_event_id_small_vector():
    # NIP-01 known vector: kind 1 "hello world" event
    ev = nostr.Event(
        kind=1,
        tags=[],
        content="hello world",
        created_at=1698350645,
        pubkey="76c71aae3a491f1d9eec47cba24e8d2a4f0c38e4b5b8b8c9b0b0b0b0b0b0b0b0b",
        id="",
        sig="",
    )
    ev.id = nostr.compute_event_id(ev)
    assert len(ev.id) == 64
    # recompute independently
    ser = json.dumps([0, ev.pubkey, ev.created_at, ev.kind, ev.tags, ev.content], separators=(",", ":"))
    assert ev.id == hashlib.sha256(ser.encode()).hexdigest()


def test_sign_and_verify_roundtrip():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    ev = nostr.Event(kind=30100, tags=[["d", "m"]], content="{}", created_at=12345)
    nostr.sign_event(ev, sk)
    assert len(ev.sig) == 128
    assert ev.verify()
    assert nostr.verification(ev.to_dict())

    # tamper
    ev2 = ev.to_dict()
    ev2["content"] = "{} "
    assert not nostr.verification(ev2)


def test_model_card_tag_shapes():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    ev = nostr.create_model_card_event(
        sk, model_id="pump-failure-v1", title="T", version=3,
        onnx_ref=["ab" * 32, "f64le", "48192", "https://blossom.example"],
        groups=[("fc-layers", "normal", ["fc1.weight", "fc1.bias"]),
                ("coefs", "mvnormal", ["w"])],
        eta0_ref=["cd" * 32, "f64le", "196608", "https://blossom.example"],
        blossom_servers=["https://blossom.example"],
        ttl=2592000,
    )
    tags = {t[0]: t for t in ev["tags"]}
    assert tags["o"] == ["o", "ab" * 32, "f64le", "48192", "https://blossom.example"]
    assert tags["x"] == ["x", "cd" * 32, "f64le", "196608", "https://blossom.example"]
    assert tags["d"] == ["d", "pump-failure-v1"]
    assert ["F", "normal"] in ev["tags"]
    assert ["G", "fc-layers"] in ev["tags"]
    assert ["g", "coefs", "mvnormal", "w"] in ev["tags"]
    assert nostr.verification(ev)


def test_site_tag_shapes_a_m_p_e():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    creator = bip340.get_pubkey(bytes.fromhex(sk)).hex()
    ev = nostr.create_site_contribution_event(
        sk, model_id="pump-failure-v1", model_version=3,
        model_coordinate=f"30100:{creator}:pump-failure-v1",
        delta_eta_ref=["ee" * 32, "i8", "24576", "https://blossom.example"],
        cavity_members=[("aa" * 32, "bb" * 32, [0.94, 0.31]), ("cc" * 32, "dd" * 32, [0.88])],
        expiration=1786752000,
        samples=8234, free_energy=-1247.3,
    )
    tags = {t[0] for t in ev["tags"]}
    assert "m" in tags and "p" in tags and "e" in tags and "E" in tags
    a = [t for t in ev["tags"] if t[0] == "a"][0]
    assert a == ["a", f"30100:{creator}:pump-failure-v1", "", "model"]
    m = [t for t in ev["tags"] if t[0] == "m"][0]
    assert m[3].count(",") == 1  # two group weights
    assert nostr.verification(ev)


def test_attestation_d_tag_scopes():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    pk = bip340.get_pubkey(bytes.fromhex(sk)).hex()
    peer = "aa" * 32
    ev = nostr.create_trust_attestation_event(
        sk, target_pubkey=peer, inclusion_probability=0.82,
        model_id="pump-failure-v1", group="fc-layers",
        model_coordinate=f"30100:{pk}:pump-failure-v1",
    )
    d = [t for t in ev["tags"] if t[0] == "d"][0][1]
    assert d == f"{peer}:pump-failure-v1:fc-layers"
    i = [t for t in ev["tags"] if t[0] == "i"][0][1]
    assert 0.0 < float(i) < 1.0
    assert nostr.verification(ev)


def test_attestation_rejects_out_of_bounds_p():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    with pytest.raises(ValueError):
        nostr.create_trust_attestation_event(sk, target_pubkey="aa" * 32, inclusion_probability=1.0)


def test_deletion_event_shape():
    sk = "0000000000000000000000000000000000000000000000000000000000000002"
    ev = nostr.create_deletion_event(sk, ["bb" * 32])
    assert ev["kind"] == 5
    assert ["e", "bb" * 32] in ev["tags"]
    assert nostr.verification(ev)


def test_blob_ref_helper():
    r = nostr.blob_ref("ab" * 32, "f64le", 100, ["https://a", "https://b"])
    assert r == ["ab" * 32, "f64le", "100", "https://a", "https://b"]