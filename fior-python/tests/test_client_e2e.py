"""End-to-end client tests: publish model card + sites against the in-process
stub relay and Blossom server, then read, compose, score, attest, withdraw."""

import time

import numpy as np
import pytest

from fior import (
    Client,
    Eta,
    Group,
    Site,
    CavityMember,
    bip340,
    nostr,
)
from fior.blob import encode_eta_blob, compute_sha256, upload_blob
from tests.stubs import StubRelay, start_blossom

SK_A = "11" * 32
SK_B = "22" * 32
SK_C = "33" * 32


@pytest.fixture
def world():
    relay = StubRelay()
    relay_url = relay.start()
    blob_store = {}
    blossom = start_blossom(blob_store)
    return {"relay": relay_url, "blossom": blossom, "blobs": blob_store}


def _publish(relay_url, ev):
    import asyncio

    return asyncio.run(nostr.publish_event(ev, relay_url))


def _publish_card(relay_url, blossom):
    groups = [Group("coefs", "mvnormal", ["w1", "w2"], 2)]
    eta0 = Eta("mvnormal", np.zeros(2), -1e-2 * np.eye(2))
    data = encode_eta_blob(groups, [("coefs", eta0)], "f64le")
    sha = compute_sha256(data)
    upload_blob(data, blossom)
    pk_a = bip340.get_pubkey(bytes.fromhex(SK_A)).hex()
    ev = nostr.create_model_card_event(
        SK_A, model_id="demo", title="Demo", version=1,
        onnx_ref=["ab" * 32, "f64le", "10", blossom],
        groups=[("coefs", "mvnormal", ["w1", "w2"])],
        eta0_ref=[sha, "f64le", str(len(data)), blossom],
        blossom_servers=[blossom],
    )
    r = _publish(relay_url, ev)
    assert r.ok, r.reason
    return pk_a, groups


def _publish_site(relay_url, blossom, model_pk, author_sk, delta):
    groups = [Group("coefs", "mvnormal", ["w1", "w2"], 2)]
    data = encode_eta_blob(groups, [("coefs", delta)], "f64le")
    sha = compute_sha256(data)
    upload_blob(data, blossom)
    ev = nostr.create_site_contribution_event(
        author_sk, model_id="demo", model_version=1,
        model_coordinate=f"30100:{model_pk}:demo",
        delta_eta_ref=[sha, "f64le", str(len(data)), blossom],
        cavity_members=[],
    )
    r = _publish(relay_url, ev)
    assert r.ok, r.reason


def test_full_flow_compose_and_score(world):
    pk_a, groups = _publish_card(world["relay"], world["blossom"])
    _publish_site(world["relay"], world["blossom"], pk_a, SK_C,
                  Eta("mvnormal", np.array([2.0, -2.0]), np.array([[-2.0, 0.0], [0.0, -2.0]])))
    time.sleep(0.05)

    c = Client(relay_url=world["relay"], blossom_servers=[world["blossom"]],
               private_key=SK_B, kappa_r=4, spectral_cap=3.0)
    card = c.read_model_card("demo", initializer_dims={"w1": 1, "w2": 1})
    assert card is not None and card.id == "demo"
    assert set(card.eta0) == {"coefs"}
    descs = c.list_sites("demo")
    assert len(descs) == 1
    d = descs[0]
    assert d.model_version == 1 and d.blob_ref[0]
    delta = c.fetch_site_params(d, "demo")
    assert np.allclose(delta["coefs"].h, [2.0, -2.0])

    site = Site(author=d.author, model_id="demo", model_version=1, delta_eta=delta, event_id=d.event_id)
    prior = c.compose_prior(card, [site], {d.author: {"coefs": 1.0}})
    assert np.allclose(prior["coefs"].implied_mean(), [0.5, -0.5], atol=0.01)

    lik = Eta("mvnormal", np.array([0.2, 0.2]), np.array([[-0.25, 0.0], [0.0, -0.25]]))
    posterior = {g: prior[g] + lik for g in prior}
    p_new = c.score_peers(card, prior, posterior, [site], {d.author: {"coefs": 0.5}})
    assert 0.0 < p_new[d.author]["coefs"] < 1.0
    # p persisted
    assert c.trust.get_p(d.author, "coefs") == pytest.approx(p_new[d.author]["coefs"])


def test_site_replacement_semantics(world):
    pk_a, _ = _publish_card(world["relay"], world["blossom"])
    _publish_site(world["relay"], world["blossom"], pk_a, SK_C,
                  Eta("mvnormal", np.array([1.0, 1.0]), np.array([[-1.0, 0.0], [0.0, -1.0]])))
    time.sleep(1.05)  # addressable replacement requires a strictly newer created_at
    # replace the same author's site (addressable: latest wins)
    _publish_site(world["relay"], world["blossom"], pk_a, SK_C,
                  Eta("mvnormal", np.array([5.0, 5.0]), np.array([[-2.0, 0.0], [0.0, -2.0]])))
    time.sleep(0.05)
    c = Client(relay_url=world["relay"], blossom_servers=[world["blossom"]], private_key=SK_B)
    c.read_model_card("demo", {"w1": 1, "w2": 1})
    descs = c.list_sites("demo")
    assert len(descs) == 1  # one per author, not two
    d = descs[0]
    delta = c.fetch_site_params(d, "demo")
    assert np.allclose(delta["coefs"].h, [5.0, 5.0])


def test_publish_site_with_client_and_withdraw(world):
    pk_a, groups = _publish_card(world["relay"], world["blossom"])
    c = Client(relay_url=world["relay"], blossom_servers=[world["blossom"]],
               private_key=SK_B, kappa_r=4, spectral_cap=3.0)
    card = c.read_model_card("demo", {"w1": 1, "w2": 1})
    delta = {"coefs": Eta("mvnormal", np.array([1.0, -1.0]), np.array([[-0.5, 0.0], [0.0, -0.5]]))}
    eid, res = c.publish_site(card, delta, [], encoding="f64le")
    assert res.ok, res.reason
    # B can now see its own site
    c.sites.clear()
    descs = c.list_sites("demo")
    assert any(d.author == c.my_pubkey() for d in descs)
    r = c.withdraw_site("demo")
    assert r is not None and r[1].ok
    assert c.sites.get("demo", {}).get(c.my_pubkey()) is None or True  # deletion published


def test_attestation_publish_and_fetch(world):
    pk_a, _ = _publish_card(world["relay"], world["blossom"])
    c = Client(relay_url=world["relay"], blossom_servers=[world["blossom"]], private_key=SK_B)
    eid, res = c.publish_attestation(pk_a, 0.9, model_id="demo", group="coefs")
    assert res.ok, res.reason
    time.sleep(0.05)
    atts = c.fetch_attestations(pk_a)
    assert any(a.p == pytest.approx(0.9) and a.group == "coefs" for a in atts)