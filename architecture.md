# FIOR Architecture

## Brief

**FIOR (Federated Inference Over Relays)** is a Nostr protocol for collaborative machine learning without sharing raw data. Multiple parties train the same model on their own private data and exchange only the statistical parameters that describe what each party learned. An aggregator combines these into a single model that is better than any party could train alone.

### The Moving Parts

| Part | Language | Repo | Status |
|------|----------|------|--------|
| Protocol spec | Markdown | `fior/protocol.md` | Done |
| UI integration guide | Markdown | `fior/ui-integration.md` | Done |
| Test data generator | Python | `forum/testmodel/` | Done |
| Protocol handler | Go | `forum/fiorgo/` | Done |
| Docker e2e stack | YAML/Dockerfile | `forum/` | Done |
| Aggregator service | Go | TBD | Not started |
| Marketplace UI | React | `forum/src/` | Not started |
| Trust/reputation layer | Go/Python | TBD | Not started |

### How It Works

1. A model creator exports their model definition in ONNX format, uploads it to a Blossom server, and publishes a **Model Card** (Nostr kind 30100) referencing the ONNX file.

2. An aggregator publishes an initial **Prior** (30101) -- a vector of zeros meaning "no information yet."

3. Nodes join the federation (30103), download the prior, and train the model on their own local data. Their raw data never leaves their machine. Training produces a **Posterior** -- updated distribution parameters describing what the model learned from the local data.

4. Nodes publish their posterior as a **Posterior Submit** (30102). The parameters are encoded as float64 natural parameters (exponential family form), hex-encoded in Nostr event tags.

5. The aggregator collects posteriors, validates them, and **aggregates** by summing the natural parameter differences. This is mathematically sound because the exponential family is closed under addition.

6. The aggregator publishes an updated **Prior** (30101) with the combined result. All nodes download it and repeat.

7. Nodes can back-test other nodes' posteriors against their own holdout data and publish **Trust Evaluations** (30104) -- per-parameter-group scores showing whether the contribution improved or degraded predictions. The aggregator publishes **Trust Snapshots** (30105) summarizing these evaluations for the marketplace UI.

### To Be Done

- **Persistent aggregator**: the current e2e test runs once; need a daemon that continuously watches for posteriors and publishes priors on a schedule.
- **Marketplace UI**: wire the existing React/NDK forum app to the FIOR event kinds per the UI integration guide.
- **Trust layer (stage 2)**: evaluator reputation tracking, consensus-weighted aggregation to route around poisoned parameters.
- **Production model support**: the test model uses a simple Bayesian linear regression with closed-form posteriors. Production models need Pyro/Numpyro integration for arbitrary ONNX models with VI training.
- **Blossom integration**: the protocol spec references ONNX files via Blossom blobs; this needs wiring into the aggregator and marketplace.

---

## 1. Protocol Layer -- `fior/protocol.md`

Defines six Nostr event kinds for the FIOR protocol. All events are standard Nostr JSON, signed with BIP-340 Schnorr signatures, transmitted over WebSocket to Nostr relays.

### Event Kinds

#### 30100 -- Model Card (Replaceable, NIP-33)

Defines a model's structure. Instead of inventing a custom schema, FIOR references an ONNX file (the industry standard ML interchange format). The model card maps ONNX initializer tensors to exponential family distributions and groups related parameters.

```
Tags: d, title, summary, version, agg, onnx, onnxhash, dist, group
```

The `d` tag carries the model identifier (slug). The `onnx` tag is a Nostr event ID pointing to the Blossom upload of the `.onnx` file. `dist` and `group` tags assign exponential family distributions to ONNX initializer tensors.

#### 30101 -- Prior Broadcast

Published by the aggregator. Each publication supersedes the previous for the same model. Carries the global natural parameter vector per group, hex-encoded as LE float64.

```
Tags: d, round, p, η
```

The initial prior is all zeros (`η = [0] * len(group)`) representing infinite variance -- "no information yet."

#### 30102 -- Posterior Submit

Published by a node after local training. References which prior the training started from (`p` tag). Carries the local posterior natural parameters per group.

```
Tags: d, p, η
```

The aggregator computes `η_diff = η_local - η_prior_ref` and adds it to the global sum.

#### 30103 -- Node Registration

Announces joining or leaving a model's federation.

```
Tags: d, action
```

#### 30104 -- Trust Evaluation

Published by a node that back-tested another node's posterior on their own holdout data. Carries per-group scores with variance.

```
Tags: d, post, group, metric, samples
Group tag: ["group", "<name>", "<score>", "<variance>"]
```

Score > 0 means the contribution improved predictions; score < 0 means it degraded them. Variance quantifies how consistent the improvement was across the holdout set.

#### 30105 -- Trust Snapshot

Published by the aggregator. Summarizes all 30104 evaluations for a specific posterior into per-group consensus scores. This gives the marketplace UI a fast lookup without replaying the full evaluation chain.

```
Group tag: ["group", "<name>", "<mean_score>", "<std_score>", "<evaluator_count>"]
```

### Natural Parameter Encoding

Parameters are exchanged in the exponential family's natural parameter form. For a group with distribution `normal` and `d` scalar parameters:

```
η₁_i = μ_i / σ²_i       (mean natural parameter)
η₂_i = -1 / (2 * σ²_i)   (precision natural parameter)
```

Total: `2d` float64 values, LE binary, hex-encoded in the `η` tag.

Aggregation is addition: `η_global += η_diff`. This works because the exponential family is closed under addition of natural parameters. No complex merging algorithm needed.

### Aggregation (Stage 1)

Simple unweighted sum. For each posterior:

```
η_diff = η_posterior - η_prior_ref
η_global += η_diff
```

Stage 2 will add consensus-weighted aggregation using trust evaluation data to suppress poisoned contributions.

---

## 2. Test Model -- `forum/testmodel/` (Python)

A deterministic, verifiable test harness for the FIOR protocol. Provides everything needed to validate the full protocol flow end-to-end without requiring real ML infrastructure.

### Data Generator (`generator.py`)

Generates synthetic time series data for a 4-feature input model:

| Feature | Pattern | Represents |
|---------|---------|------------|
| x1 | Sine wave | Periodic seasonal effect |
| x2 | Linear drift | Gradual degradation |
| x3 | Square wave | Discrete state changes |
| x4 | Smoothed random walk | Cumulative stochastic process |

The ground truth is known: `y = 0.5*x1 - 0.3*x2 + 1.2*x3 - 0.8*x4 + 0.1 + noise`.

Each simulated "node" receives a variant dataset. A node seed determines the exact noise pattern. Same seed always produces the same dataset (deterministic PRNG). Feature noise simulates sensor imprecision. The generator also supports label flip probability for simulating misclassified data.

### Bayesian Linear Regression (`model.py`)

Mean-field variational inference with per-parameter independent Normal posteriors. Prior: each parameter `~ Normal(0, 10²)`. Known observation noise `σ² = 0.01`. Coordinate ascent VI updates each parameter's posterior mean and variance independently.

Key methods:
- `fit(X, y)` -- run VI on local data
- `natural_parameters()` -- encode posterior as `[η₁..., η₂...]` float64 vector
- `load_natural_parameters(eta)` -- decode natural parameters back to mean/variance
- `prior_natural_parameters()` -- get the prior's natural parameters

### Protocol Operations (`protocol.py`)

- `encode_hex(eta)` / `decode_hex(data)` -- float64 LE binary ↔ hex string
- `prior_diff(posterior_eta, prior_eta)` -- compute η difference
- `aggregate(global_eta, diffs)` -- sum diffs into global
- `simulate_federation(n_nodes, ...)` -- run full simulated round

### CLI (`__main__.py`)

Called by the Go binary via subprocess. Usage:

```
python -m testmodel --seed 42 --samples 2000 --node 1 --iters 30
```

Outputs JSON with `natural_eta` (hex), `post_mean`, `true_parameters`, etc.

### Tests (`test_protocol.py`)

12 tests, all passing. Covers:
- Deterministic generation (same seed = same data)
- Single-node parameter recovery
- Natural parameter round-trip (encode/decode preserves float64)
- Hex encoding round-trip
- Zero diff when no training
- Nonzero diff with training data
- Aggregation correctness (sum of diffs)
- Federation convergence (MAE < 0.15 vs ground truth)
- More nodes beats the zero-prior baseline
- More data produces tighter posteriors (lower variance)

---

## 3. Protocol Handler -- `forum/fiorgo/` (Go)

A Go module using the orly Nostr library (`git.smesh.lol/orly` v0.65.60) for event construction, signing, and relay communication.

### Package: `pkg/fior`

**`kinds.go`** -- Event kind constants (30100-30105).

**`natparam.go`** -- Natural parameter encode/decode and aggregation:

- `DecodeNaturalParams(hex, dim) → (eta1, eta2, error)` -- hex to float64 vectors
- `EncodeNaturalParams(eta1, eta2) → hex` -- float64 vectors to hex
- `Aggregate(globalEta1, globalEta2, diffs)` -- sum η differences in-place
- `Diff(posteriorEta1, posteriorEta2, priorEta1, priorEta2) → (d1, d2)` -- compute η difference

### Binary: `cmd/e2etest`

Single binary that runs the complete FIOR protocol flow against a live Nostr relay. Steps:

1. Generate a BIP-340 keypair (secp256k1)
2. Connect to relay via WebSocket
3. Publish Model Card (30100) for "test-model-v1"
4. Publish initial Prior (30101) with zero natural parameters
5. For each of N nodes: call Python testmodel via subprocess, capture posterior
6. Publish each posterior as a 30102 event
7. Aggregate all posteriors locally
8. Publish aggregated Prior (30101, round 1)
9. Compare against ground truth, print MAE

Run: `FIOR_TESTMODEL_DIR=.. FIOR_RELAY=wss://nos.lol go run ./cmd/e2etest/`

Verified: MAE 0.031 vs ground truth (3 nodes × 2000 samples).

---

## 4. Docker Stack -- `forum/`

Self-contained local test environment.

### `docker-compose.e2e.yml`

Two services:

| Service | Image | Purpose |
|---------|-------|---------|
| `relay` | `scsibug/nostr-rs-relay:latest` | Local Nostr relay with SQLite, no auth |
| `e2etest` | Built from `Dockerfile.e2e` | Runs the full FIOR protocol test |

Usage: `docker compose -f docker-compose.e2e.yml up --build --abort-on-container-exit`

The relay uses `relay-config.toml` configured for testing: no authentication, no rate limiting, all event kinds allowed, port 8080.

### `Dockerfile.e2e`

Multi-stage build:
1. `golang:1.25-alpine` -- downloads orly dependency, builds Go binary
2. `python:3.12-slim` -- installs numpy, copies testmodel + Go binary

The e2etest binary connects to `ws://relay:8080` (the compose service name).

---

## 5. Marketplace UI Integration -- `fior/ui-integration.md`

Design document mapping FIOR protocol events to React marketplace UI state. Covers:

- **Model discovery**: query 30100 events, display model name, description, parameter count
- **Prior display**: query latest 30101, show round number and derived summary stats
- **Posterior marketplace**: list 30102 events with trust data from 30105 snapshots
- **Trust signal indicator**: `z = |score|/std` → strong/moderate/weak positive or negative
- **Evaluation workflow**: user clicks "Evaluate", client splits local data, runs back-test, publishes 30104
- **Relay subscription pattern**: subscribe to all FIOR kinds on mount, maintain in-memory event store

---

## 6. To Be Done

### Persistent Aggregator

The current `cmd/e2etest` runs once and exits. The production aggregator needs to:
- Run as a daemon
- Subscribe continuously to 30102 and 30103 events for its models
- Maintain per-model state (current prior, registered nodes, pending posteriors)
- Aggregate on a schedule (every N posteriors or T seconds)
- Publish 30105 snapshots when enough evaluations accumulate
- Handle node deregistration (remove parameters when a node leaves)

### Marketplace UI (React)

The forum SPA (`forum/src/`) currently handles NIP-07 login and Blossom file upload for arbitrary files. Needs to be wired to FIOR event kinds:
- Replace file listing with model card browsing (query 30100)
- Replace file detail with posterior listing (query 30102 + 30105)
- Add trust evaluation workflow (back-test → publish 30104)
- Show trust signal indicators per parameter group

### Trust Layer (Stage 2)

Currently, aggregation is unweighted -- every posterior contributes equally. Stage 2 adds:
- Evaluator reputation: track which evaluators produce scores that correlate with actual model improvement
- Consensus-weighted aggregation: `η_global += w_g * η_diff` where `w_g = sigmoid(mean(score/variance))`
- Adversarial resistance testing: introduce nodes that submit poisoned parameters, verify the trust layer suppresses them

### Production Model Support

The test model uses a simple Bayesian linear regression with closed-form VI updates (analytically tractable). Production models need:
- Pyro/Numpyro backend for arbitrary ONNX models
- Stochastic variational inference (SVI) for models without closed-form posteriors
- GPU-accelerated training (local to each node)
- ONNX export/import for model definition exchange

### Blossom Integration

The protocol spec references ONNX files via Blossom blob event IDs. The aggregator and marketplace need to:
- Fetch ONNX files from Blossom servers to validate parameter shapes
- Verify `onnxhash` (SHA256) against downloaded files

### Buzz Relay Integration

The client operates a Buzz relay at `buzz.nostri.xyz`. The protocol should eventually be tested against this relay. Buzz requires authentication -- the aggregator and nodes will need Buzz invites or API keys.

### NIP Submission

The protocol spec is clean enough to be submitted as a NIP to `nostr-protocol/nips`. The event kind numbers (30100-30105) have been verified as unused. The client should submit the PR when ready.
