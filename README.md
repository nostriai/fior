# FIOR - Federated Inference Over Relays

A Nostr protocol for collaborative machine learning without sharing raw data.

## Overview

FIOR enables multiple parties to train a model on their private data and exchange only statistical summaries. Each party combines contributions locally, weighting peers by how much they improve inference. There is no aggregator - every participant computes independently.

## How It Works

1. **Model Card** (30100): Creator publishes ONNX model with distribution mappings
2. **Site Contribution** (30101): Nodes publish what they learned (Δη)
3. **Trust Attestation** (30102): Nodes rate peers with a scalar p value
4. **Local Composition**: Each client builds its own prior: `η₀ + Σ p·Δη`

## Quick Start

### Python

```python
from fior import Client, ModelCard

# Connect to relay
client = Client("wss://relay.example.com")

# Fetch a model
model = client.fetch_model_card("pump-failure-v1")

# Fetch sites from trusted peers
sites = client.fetch_sites("pump-failure-v1")

# Compose prior from trusted peers
p_values = [0.9, 0.7, 0.8]  # trust weights
prior = client.compose_prior(model.eta0, sites, p_values)

# Train locally...
# Publish your site
client.publish_site(private_key, model.id, model.version, delta_eta, cavity)
```

### JavaScript

```javascript
import { Client } from 'fior';

const client = new Client('wss://relay.example.com');

// Fetch a model
const model = await client.fetchModelCard('pump-failure-v1');

// Fetch sites
const sites = await client.fetchSites('pump-failure-v1');

// Compose prior
const prior = client.composePrior(model.eta0, sites, [0.9, 0.7, 0.8]);

// Train locally...
// Publish your site
await client.publishSite(privateKey, model.id, model.version, deltaEta, cavity);
```

## Event Kinds

| Kind | Name | Purpose |
|------|------|---------|
| 30100 | Model Card | ONNX blob, distributions, base prior |
| 30101 | Site Contribution | Node's likelihood approximation Δη |
| 30102 | Trust Attestation | Scalar p rating a peer |

All kinds are addressable: one event per `(kind, pubkey, d)`, latest supersedes.

## Trust Layer

- **BMR**: Bayesian Model Reduction scores peers via log Bayes factor
- **Loewner Cap**: Bounds claimed precision against other peers
- **Corroboration**: One-peer-one-vote checking agreement with trusted peers
- **Novelty**: First-seen tracking defends against replay attacks

p is local and per-client. No consensus score exists.

## Repository Structure

```
fior/
├── protocol.md          # Protocol specification v3
├── architecture.md      # Architecture and implementation plan
├── interface.md         # Client API reference
├── ui-integration.md    # UI integration guide
├── distributions.md     # Exponential family reference
├── fior-python/         # Python reference library (in progress)
├── fior-js/             # JavaScript reference library (planned)
└── test/
    └── fior_sim.py      # Algorithm reference implementation
```

## Implementation Status

| Component | Status |
|-----------|--------|
| Protocol spec | v3 complete |
| Python library | In progress |
| JavaScript library | Not started |
| Algorithm reference | fior_sim.py |

## References

- [Federated Learning as Variational Inference](https://arxiv.org/abs/2302.04228)
- [Partitioned Variational Inference](https://arxiv.org/abs/2202.12275)
- [Bayesian Model Reduction](https://arxiv.org/abs/1805.07092)
