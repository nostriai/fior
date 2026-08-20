"""CLI tests: commands talk to real (stub) relays and Blossom - no
placeholders.  Config is redirected into a temp dir for isolation."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

import fior.cli.main as cli
from fior import Eta, Group, bip340, nostr
from fior.blob import encode_eta_blob, compute_sha256, upload_blob
from tests.stubs import StubRelay, start_blossom

runner = CliRunner()
SK = "11" * 32


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    d = tmp_path / "config" / "fior"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "CONFIG_DIR", d)
    monkeypatch.setattr(cli, "CONFIG_FILE", d / "config.toml")
    monkeypatch.setattr(cli, "TRUST_FILE", d / "trust.json")
    return d


@pytest.fixture
def environment():
    relay = StubRelay()
    relay_url = relay.start()
    store = {}
    blossom = start_blossom(store)
    return relay_url, blossom


def _configure(monkeypatch, relay_url, blossom):
    import asyncio

    cli.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "CONFIG_FILE", cli.CONFIG_DIR / "config.toml")
    monkeypatch.setattr(cli, "TRUST_FILE", cli.CONFIG_DIR / "trust.json")
    for cmd in (["config", "init"],
                ["config", "set", "relay.url", relay_url],
                ["config", "set", "blossom.server", blossom],
                ["config", "set", "identity.private_key", SK],
                ["config", "set", "trust.kappa_r", "4"]):
        r = runner.invoke(cli.app, cmd)
        assert r.exit_code == 0, (cmd, r.output)


def _seed(relay_url, blossom):
    import asyncio

    groups = [Group("coefs", "mvnormal", ["w1", "w2"], 2)]
    eta0 = Eta("mvnormal", np.zeros(2), -1e-2 * np.eye(2))
    data = encode_eta_blob(groups, [("coefs", eta0)], "f64le")
    sha = compute_sha256(data)
    upload_blob(data, blossom)
    pk = bip340.get_pubkey(bytes.fromhex(SK)).hex()
    ev = nostr.create_model_card_event(
        SK, model_id="demo", title="Demo", version=1,
        onnx_ref=["ab" * 32, "f64le", "10", blossom],
        groups=[("coefs", "mvnormal", ["w1", "w2"])],
        eta0_ref=[sha, "f64le", str(len(data)), blossom],
        blossom_servers=[blossom],
    )
    assert asyncio.run(nostr.publish_event(ev, relay_url)).ok


def test_cli_config_roundtrip(cli_home):
    r = runner.invoke(cli.app, ["config", "init"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli.app, ["config", "show", "--json"])
    cfg = json.loads(r.output)
    assert cfg["relay"]["url"] == "wss://relay.example.com"


def test_cli_fetch_model_returns_real_data(environment, cli_home, monkeypatch):
    relay_url, blossom = environment
    _seed(relay_url, blossom)
    _configure(monkeypatch, relay_url, blossom)
    r = runner.invoke(cli.app, ["fetch", "model", "demo", "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["id"] == "demo"
    assert out["groups"][0]["family"] == "mvnormal"
    assert "Sample Model" not in r.output


def test_cli_fetch_sites_empty_list(environment, cli_home, monkeypatch):
    relay_url, blossom = environment
    _configure(monkeypatch, relay_url, blossom)
    r = runner.invoke(cli.app, ["fetch", "sites", "demo", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output) == []


def test_cli_trust_set_list(environment, cli_home, monkeypatch):
    relay_url, blossom = environment
    _configure(monkeypatch, relay_url, blossom)
    r = runner.invoke(cli.app, ["trust", "set", "aa" * 32, "coefs", "0.7"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli.app, ["trust", "list", "--json"])
    out = json.loads(r.output)
    assert any(e["group"] == "coefs" and e["p"] == 0.7 for e in out)