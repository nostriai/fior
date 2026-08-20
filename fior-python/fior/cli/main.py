"""FIOR command-line interface.

Backed by fior.client - no placeholders.  Configuration lives at
~/.config/fior/config.toml.  `--json` emits machine-readable JSON;
`--no-color` disables rich styling.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from fior import (
    Client,
    ClientError,
    DomainError,
    Eta,
    Site,
    CavityMember,
    FAMILY_MVNORMAL,
    ONE_BLOCK_FAMILIES,
)

CONFIG_DIR = Path.home() / ".config" / "fior"
CONFIG_FILE = CONFIG_DIR / "config.toml"
TRUST_FILE = CONFIG_DIR / "trust.json"

_NO_COLOR = False


def _console():
    from rich.console import Console

    return Console(color_system=None if _NO_COLOR else "auto")


app = typer.Typer(name="fior", help="FIOR - Federated Inference Over Relays", no_args_is_help=True)


@app.callback()
def _main(no_color: bool = typer.Option(False, "--no-color", help="Disable colors.")):
    global _NO_COLOR
    _NO_COLOR = no_color


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def _read_toml() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        import tomli

        if not CONFIG_FILE.exists():
            return {}
        with open(CONFIG_FILE, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    # minimal fallback for the flat `[section]` / `key = "value"` files we write
    cfg: dict = {}
    section = None
    for raw in CONFIG_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            cfg.setdefault(section, {})
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"')
            if section:
                cfg[section][k.strip()] = v
            else:
                cfg[k.strip()] = v
    return cfg


def _write_toml(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# FIOR configuration", ""]
    for section, values in cfg.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                if k == "private_key" and v:
                    lines.append(f"# private_key = \"<set in file>\"")
                    lines.append(f"private_key = \"{v}\"")
                else:
                    lines.append(f"{k} = \"{v}\"")
            lines.append("")
        else:
            lines.append(f"{section} = \"{values}\"")
    CONFIG_FILE.write_text("\n".join(lines) + "\n")


@app.command()
def config(
    action: str = typer.Argument(help="init | show | set"),
    key: Optional[str] = typer.Argument(None, help="dotted key, e.g. relay.url"),
    value: Optional[str] = typer.Argument(None, help="value for set"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Manage the FIOR configuration file."""
    if action == "init":
        _write_toml({
            "relay": {"url": "wss://relay.example.com"},
            "blossom": {"server": "https://blossom.example.com"},
            "identity": {"private_key": ""},
            "trust": {"kappa_r": "4"},
        })
        _console().print(f"[green]Config initialized at {CONFIG_FILE}[/green]")
    elif action == "show":
        cfg = _read_toml()
        if json_output:
            _console().print(json.dumps(cfg, indent=2))
        else:
            shown = dict(cfg)
            if "identity" in shown and shown["identity"].get("private_key"):
                shown["identity"]["private_key"] = "***hidden***"
            _console().print(json.dumps(shown, indent=2))
    elif action == "set":
        if not key or value is None:
            _console().print("[red]set requires KEY and VALUE[/red]")
            raise typer.Exit(2)
        cfg = _read_toml()
        if "." in key:
            section, k = key.split(".", 1)
            cfg.setdefault(section, {})[k] = value
        else:
            cfg[key] = value
        _write_toml(cfg)
        _console().print(f"[green]Set {key} = {value}[/green]")
    else:
        _console().print(f"[red]unknown action: {action}[/red]")
        raise typer.Exit(2)


# --------------------------------------------------------------------------
# client construction
# --------------------------------------------------------------------------


def _client(relay: Optional[str] = None) -> Client:
    cfg = _read_toml()
    relay_url = relay or cfg.get("relay", {}).get("url", "wss://relay.example.com")
    blossom = [cfg.get("blossom", {}).get("server")] if cfg.get("blossom", {}).get("server") else []
    private_key = cfg.get("identity", {}).get("private_key") or None
    try:
        kappa_r = float(cfg.get("trust", {}).get("kappa_r", "4"))
    except ValueError:
        kappa_r = 4.0
    return Client(
        relay_url=relay_url,
        blossom_servers=blossom,
        private_key=private_key,
        kappa_r=kappa_r,
        trust_path=str(TRUST_FILE),
    )


def _eta_from_spec(group_name: str, family: str, spec: dict) -> Eta:
    """Build an Eta from a JSON spec: {"h": [...], "lam": [...]|[[...]]}."""
    h = spec["h"]
    if family in ONE_BLOCK_FAMILIES:
        return Eta(family, h)
    if "lam" not in spec:
        raise typer.BadParameter(f"group {group_name}: spec requires 'lam'")
    lam = spec["lam"]
    if family == FAMILY_MVNORMAL:
        import numpy as np
        a = np.array(lam, dtype=float)
        if a.ndim == 1:
            a = np.diag(a)
        if a.shape[0] != len(h) or a.shape[1] != len(h):
            raise typer.BadParameter(f"mvnormal lam must be {len(h)}x{len(h)}")
    return Eta(family, h, lam)


def _eta_json(eta: Eta) -> dict:
    import numpy as np
    out = {"h": np.asarray(eta.h).round(6).tolist()}
    if eta.Lam is not None:
        lam = np.asarray(eta.Lam)
        if eta.family == FAMILY_MVNORMAL:
            out["lam"] = lam.round(6).tolist()
        else:
            out["lam"] = lam.round(6).tolist()
    return out


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


@app.command()
def fetch(
    target: str = typer.Argument(help="model | sites | attestations"),
    identifier: str = typer.Argument(help="model id, or pubkey for attestations"),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Fetch data from the relay."""
    c = _client(relay)
    console = _console()
    try:
        if target == "model":
            card = c.read_model_card(identifier, fetch_eta0=False)
            if card is None:
                console.print(f"[yellow]no model card for {identifier}[/yellow]")
                raise typer.Exit(1)
            out = {
                "id": card.id, "title": card.title, "version": card.version,
                "creator": card.creator, "event_id": card.event_id,
                "groups": [{"name": g.name, "family": g.family, "dim": g.dim,
                            "initializers": g.initializers} for g in card.groups],
                "eta0_count": len(card.eta0),
                "blossom": card.blossom_servers,
            }
            console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
        elif target == "sites":
            c.read_model_card(identifier)
            descs = c.list_sites(identifier)
            out = [{"author": d.author, "event_id": d.event_id,
                    "version": d.model_version, "blob": d.blob_ref,
                    "created_at": d.created_at, "expires_at": d.expires_at,
                    "samples": d.samples} for d in descs]
            console.print(json.dumps(out, indent=2) if json_output else _fmt({"sites": out}))
        elif target == "attestations":
            atts = c.fetch_attestations(identifier)
            out = [{"attester": a.attester, "target": a.target, "p": a.p,
                    "model": a.model_id, "group": a.group} for a in atts]
            console.print(json.dumps(out, indent=2) if json_output else _fmt({"attestations": out}))
        else:
            console.print(f"[red]unknown fetch target: {target}[/red]")
            raise typer.Exit(2)
    except (ClientError, DomainError, LookupError, RuntimeError) as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


def _fmt(obj) -> str:
    """Compact human-readable rendering."""
    if isinstance(obj, dict) and "id" in obj:
        return "\n".join(f"{k}: {v}" for k, v in obj.items())
    return json.dumps(obj, indent=2)


# --------------------------------------------------------------------------
# compose / score
# --------------------------------------------------------------------------


@app.command()
def compose(
    model_id: str = typer.Argument(),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Compose the local prior from currently trusted peers."""
    c = _client(relay)
    console = _console()
    try:
        card = c.read_model_card(model_id)
        if card is None:
            console.print(f"[red]model {model_id} not found[/red]")
            raise typer.Exit(1)
        descs = c.list_sites(model_id)
        members = []
        for d in descs:
            delta = c.fetch_site_params(d, model_id)
            if delta:
                members.append(Site(author=d.author, model_id=model_id,
                                    model_version=d.model_version, delta_eta=delta,
                                    event_id=d.event_id))
        p_table = {d.author: {g.name: c.trust.get_p(d.author, g.name) for g in card.groups}
                   for d in descs}
        prior = c.compose_prior(card, members, p_table)
        out = {"model": model_id, "members": len(members),
               "groups": {g: _eta_json(prior[g]) for g in prior}}
        console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
    except DomainError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except (ClientError, RuntimeError, LookupError) as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def score(
    model_id: str = typer.Argument(),
    likelihood: str = typer.Option(..., "--likelihood", help="JSON with the caller's local likelihood per group"),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
    json_output: bool = typer.Option(False, "--json"),
    rounds: int = typer.Option(1, "--rounds"),
):
    """Score peers via BMR against the caller's local data."""
    c = _client(relay)
    console = _console()
    try:
        card = c.read_model_card(model_id)
        if card is None:
            raise typer.Exit(1)
        spec = json.load(open(likelihood))
        lik = {g.name: _eta_from_spec(g.name, g.family, spec[g.name]) for g in card.groups if g.name in spec}
        descs = c.list_sites(model_id)
        members = []
        for d in descs:
            delta = c.fetch_site_params(d, model_id)
            if delta:
                members.append(Site(author=d.author, model_id=model_id,
                                    model_version=d.model_version, delta_eta=delta,
                                    event_id=d.event_id))
        p_table = {}
        for _ in range(rounds):
            for d in descs:
                p_table[d.author] = {g.name: c.trust.get_p(d.author, g.name) for g in card.groups}
            prior = c.compose_prior(card, members, p_table)
            posterior = {g.name: prior[g.name] + lik[g.name] for g in card.groups}
            p_table = c.score_peers(card, prior, posterior, members, p_table)
        out = {"model": model_id,
               "p": {a: {g: round(v, 6) for g, v in gr.items()} for a, gr in p_table.items()}}
        console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
    except (ClientError, DomainError, RuntimeError, LookupError, FileNotFoundError) as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------

publish_app = typer.Typer(help="Publish model cards, sites, or attestations.")
app.add_typer(publish_app, name="publish")


@publish_app.command("model")
def publish_model(
    spec: str = typer.Argument(help="JSON file: {id,title,version,groups:[{name,family,initials,dim}],eta0:{...},onnx:{sha,size,servers}}"),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
):
    """Publish a model card; requires the eta0 blob handled by the caller."""
    c = _client(relay)
    console = _console()
    data = json.load(open(spec))
    from fior import nostr
    groups = [(g["name"], g["family"], [str(i) for i in g.get("initials", [])]) for g in data["groups"]]
    ev = nostr.create_model_card_event(
        c.private_key,
        model_id=data["id"], title=data["title"], version=data["version"],
        onnx_ref=[data["onnx"]["sha"], "f64le", str(data["onnx"].get("size", 0))] + data["onnx"].get("servers", []),
        groups=groups,
        eta0_ref=None,
        blossom_servers=data.get("blossom_servers"),
    )
    res = c._publish(ev)
    console.print(json.dumps({"event_id": ev["id"], "ok": res.ok, "reason": res.reason}))
    return 0


@publish_app.command("site")
def publish_site_cmd(
    model_id: str = typer.Option(..., "--model", "-m"),
    delta_json: str = typer.Option(..., "--delta"),
    members_json: Optional[str] = typer.Option(None, "--members"),
    encoding: str = typer.Option("f64le"),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Upload Δη and publish the site contribution."""
    c = _client(relay)
    console = _console()
    try:
        card = c.read_model_card(model_id)
        if card is None:
            raise typer.Exit(1)
        spec = json.load(open(delta_json))
        delta = {g.name: _eta_from_spec(g.name, g.family, spec[g.name]) for g in card.groups if g.name in spec}
        members = []
        if members_json:
            mdata = json.load(open(members_json))
            for m in mdata:
                members.append(CavityMember(pubkey=m["pubkey"], event_id=m["event_id"], p_vector=m.get("p_vector", [])))
        eid, res = c.publish_site(card, delta, members, encoding=encoding)
        out = {"event_id": eid, "ok": res.ok, "reason": res.reason}
        console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
    except (ClientError, DomainError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


@publish_app.command("attestation")
def publish_attestation_cmd(
    peer: str = typer.Option(..., "--peer", "-p"),
    p: float = typer.Option(..., "--p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    group: Optional[str] = typer.Option(None, "--group", "-g"),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
    json_output: bool = typer.Option(False, "--json"),
):
    c = _client(relay)
    console = _console()
    try:
        eid, res = c.publish_attestation(peer, p, model_id=model, group=group)
        out = {"event_id": eid, "ok": res.ok, "reason": res.reason}
        console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
    except (ClientError, ValueError) as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def withdraw(
    model_id: str = typer.Argument(),
    relay: Optional[str] = typer.Option(None, "--relay", "-r"),
):
    c = _client(relay)
    console = _console()
    r = c.withdraw_site(model_id)
    if r is None:
        console.print("[yellow]no site to withdraw[/yellow]")
        return
    eid, res = r
    console.print(f"withdrawn {eid}: {res.reason or res.ok}")
    return res


# --------------------------------------------------------------------------
# blob
# --------------------------------------------------------------------------


@app.command("blob")
def blob(
    action: str = typer.Argument(help="upload | fetch"),
    arg: str = typer.Argument(help="file path, or sha256 for fetch"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    output: Optional[str] = typer.Option(None, "-o"),
    json_output: bool = typer.Option(False, "--json"),
):
    from fior.blob import upload_blob, fetch_blob, compute_sha256

    console = _console()
    server_url = server
    if not server_url:
        cfg = _read_toml()
        server_url = cfg.get("blossom", {}).get("server")
    if not server_url:
        console.print("[red]no Blossom server configured (use --server or config)[/red]")
        raise typer.Exit(2)
    try:
        if action == "upload":
            data = Path(arg).read_bytes()
            sha = upload_blob(data, server_url)
            console.print(json.dumps({"sha256": sha, "server": server_url}) if json_output else f"uploaded {sha}")
        elif action == "fetch":
            data = fetch_blob(arg, [server_url])
            if output:
                Path(output).write_bytes(data)
                console.print(f"saved {len(data)} bytes to {output}")
            else:
                sys.stdout.buffer.write(data)
        else:
            console.print(f"[red]unknown blob action: {action}[/red]")
            raise typer.Exit(2)
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# trust
# --------------------------------------------------------------------------


@app.command("trust")
def trust(
    action: str = typer.Argument(help="list | set | reset"),
    peer: Optional[str] = typer.Argument(None, help="peer pubkey"),
    group: Optional[str] = typer.Argument(None, help="group name"),
    p: Optional[float] = typer.Argument(None, help="inclusion probability"),
    json_output: bool = typer.Option(False, "--json"),
):
    c = _client()
    console = _console()
    if action == "list":
        out = []
        for pp in c.trust.peers():
            for gg, vv in c.trust._p.get(pp, {}).items():
                out.append({"peer": pp, "group": gg, "p": round(vv, 6)})
        console.print(json.dumps(out, indent=2) if json_output else _fmt(out))
    elif action == "set":
        if not peer or not group or p is None:
            console.print("[red]set requires PEER GROUP P[/red]")
            raise typer.Exit(2)
        c.trust.set_p(peer, group, p)
        c.trust.save(str(TRUST_FILE))
        console.print(f"set {peer} {group} = {p}")
    elif action == "reset":
        if peer:
            c.trust.reset(peer, group)
        else:
            c.trust.reset()
        c.trust.save(str(TRUST_FILE))
        console.print("trust table reset")
    else:
        console.print(f"[red]unknown trust action: {action}[/red]")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()