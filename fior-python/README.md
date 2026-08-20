# FIOR Python library

Python implementation of FIOR (Federated Inference Over Relays) - collaborative
Bayesian inference over Nostr relays without sharing raw data.

- Pure-Python BIP-340 signing (no native or third-party crypto dependency;
  validated against the official BIP-340 test vectors)
- Full natural-parameter blob codec: `f64le` / `f32le` / `i16le` / `i8` with
  stochastic rounding, `normal` (diagonal) and `mvnormal` (full covariance)
  groups, fail-loud validation
- The complete client API: model cards, site contributions, BMR scoring at
  unit weight, the Loewner confidence bound, corroboration, the novelty
  (replay) defence, and trust attestations
- A working CLI (`fior`) for config, fetch, compose, score, publish, blob,
  trust, and withdrawal

The protocol specification lives in the repository root (`protocol.md`,
`interface.md`); the reference arithmetic is `test/fior_sim.py`.

## Install

```sh
pip install .
```

(or from PyPI once published).  Python 3.10+; tested on 3.14.

## CLI quick start

```sh
fior config init                # ~/.config/fior/config.toml
fior config set relay.url wss://relay.example.com
fior config set blossom.server https://blossom.example.com
fior config set identity.private_key <hex>
fior fetch model <model-id>
fior compose <model-id>
```

See `CLI.md` for the full command reference.

## Library quick start

```python
from fior import Client

c = Client(relay_url="wss://relay.example.com",
           blossom_servers=["https://blossom.example.com"],
           private_key="<hex>")

card = c.read_model_card("pump-failure-v1", initializer_dims={"fc1.weight": 256})
sites = c.list_sites("pump-failure-v1")
for d in sites:
    c.fetch_site_params(d, "pump-failure-v1")

p_table = {d.author: {"fc-layers": c.trust.get_p(d.author, "fc-layers")} for d in sites}
prior = c.compose_prior(card, members, p_table)
```

## Tests

```sh
pip install -e .[dev]
python -m pytest tests/
```