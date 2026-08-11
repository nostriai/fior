# FIOR Architecture

## Brief

**FIOR (Federated Inference Over Relays)** is a Nostr protocol for collaborative machine learning without sharing raw data. Multiple parties train the same model on their own private data and exchange only the statistical parameters describing what each learned. Every party then combines those contributions locally, weighting each peer by how much that peer measurably improves its own inference.

There is no aggregator. Nothing in the protocol requires a participant to wait on, or trust the state of, any other participant.

### Repository Layout

| Part | Language | Location | Status |
|------|----------|----------|--------|
| Protocol spec | Markdown | `fior/protocol.md` | v3, done |
| Client interface | Markdown | `fior/interface.md` | v3, done |
| UI integration guide | Markdown | `fior/ui-integration.md` | v3, done |
| Distribution reference | Markdown | `fior/distributions.md` | Done |
| Test data generator | Python | `forum/testmodel/` | Done, implements v2 |
| Protocol handler | Go | `forum/fiorgo/` | Done, implements v2 |
| Docker e2e stack | YAML/Dockerfile | `forum/` | Done, implements v2 |
| Marketplace UI | React | `forum/src/` | Not started |
| Trust engine (BMR) | Go/Python | TBD | Not started |
| Blossom integration | Go | TBD | Not started |

The Go and Python components live on the `feat/test-model` branch of the `forum` repository, not on its `main`.

**The implementation currently targets v2.** It predates the removal of the aggregator, the switch from posteriors to site contributions, and the move of parameters into Blossom blobs. Section 6 enumerates the migration.

### How It Works

1. A model creator exports the model definition to ONNX, uploads it to a Blossom server, and publishes a **Model Card** (kind 30100) carrying the blob hash, the map from ONNX initializer tensors to exponential family distributions, and the base prior `η₀`.

2. A node fetches the model card, the ONNX graph, and `η₀`.

3. The node fetches the site contributions of whichever peers it judges worth the bandwidth, and **composes its own prior**:

   ```
   η_A^prior = η₀ + ρ · Σ_{n ≠ A} β_{A→n} · Δη_n
   ```

   On the first round there are no peers and the prior is just `η₀`.

4. The node trains on its own local data. Raw data never leaves the machine. Training produces a posterior; subtracting the prior it trained against yields the node's **site contribution** `Δη` — its likelihood approximation, and the quantity that composes.

5. The node uploads `Δη` as a Blossom blob, then publishes a **Site Contribution** (30101) referencing that blob and pinning the cavity it trained against. Publishing the site *is* joining; there is no registration step.

6. Each client independently scores every peer via **Bayesian model reduction**, a closed-form evaluation of how the local log evidence changes if that peer is removed. This sets `β` per peer per parameter group, and the client recomposes its prior.

7. Repeat from step 3. Steps 3 and 6 happen independently on every client.

In parallel and entirely optionally, any node may publish a **Composite** (30102) recording a specific weighted combination, a **Benchmark Descriptor** (30105) defining a comparable evaluation, **Benchmark Results** (30103) measuring sites or composites, and **Trust Attestations** (30104) sharing its Beta parameters for a peer.

### Why It Works

Aggregation is addition. The exponential family is closed under addition of natural parameters, and Nostr's addressable event semantics store exactly one event per `(kind, pubkey, d)`. Keying sites by `(author, model)` therefore gives "the latest site replaces the previous one" for free — precisely the expectation-propagation invariant that makes summing latest-per-author double-count nothing.

That is the whole reason no coordinator is needed: the relay's replacement rule *is* the consistency mechanism.

---

## 1. Protocol Layer — `fior/protocol.md`

Six addressable Nostr event kinds. All events are standard Nostr JSON signed with BIP-340 Schnorr signatures over WebSocket.

| Kind  | Name                 | `d`                                | Purpose                                  |
|-------|----------------------|------------------------------------|------------------------------------------|
| 30100 | Model Card           | `<model-id>`                       | ONNX blob, distribution map, `η₀`        |
| 30101 | Site Contribution    | `<model-id>`                       | A node's `Δη`                            |
| 30102 | Composite            | `<composite-id>`                   | A reproducible weighted combination      |
| 30103 | Benchmark Result     | `<bench-id>:<target-event-id>`     | A measurement                            |
| 30104 | Trust Attestation    | `<target-hex>[:<model>[:<group>]]` | Beta parameters for β                    |
| 30105 | Benchmark Descriptor | `<bench-id>`                       | What a benchmark measures                |

### Natural Parameters

Parameters are exchanged in exponential family natural parameter form. For a group with `d` scalar parameters under `normal`:

```
η₁ᵢ = μᵢ / σ²ᵢ        (mean natural parameter)
η₂ᵢ = −1 / (2σ²ᵢ)      (precision natural parameter)
```

Sites carry *differences* `Δη`, not posteriors.

**All η payloads are Blossom blobs.** Events carry a SHA-256 and never the parameters themselves. Beyond event size limits, the reason is that an inline payload cannot be declined: a subscriber would receive every member's full parameter vector regardless of intent, defeating the selective fetch the trust layer depends on. Encodings are raw binary — `f64le`, `f32le`, `i16le`, `i8` — with per-index scales in a blob header for the integer forms, and mandatory stochastic rounding so that quantization error stays unbiased and cancels across members rather than accumulating.

### Trust

Peer inclusion is a Bernoulli indicator with a Beta prior:

```
β_{n,g} = 1 / (1 + (b/a)·exp(−ΔF_{n,g}))
```

`ΔF` is the log Bayes factor from Bayesian model reduction, in closed form over the log-partition function `A(η)`:

```
ΔF = A(η_q) + A(η_p − βΔη) − A(η_q − βΔη) − A(η_p)
```

`A(η)` is tabulated per family in `distributions.md`, so this needs no new mathematics and costs two extra evaluations per peer per group — no holdout split, no retraining, no second pass over data.

`ΔF` is recomputed every round and never stored: sites are replaceable, so evidence about a superseded site describes something that no longer exists. Only `(a, b)` accumulates, and only from attestations — information a client cannot recompute for itself. With no attestations, `(a, b) = (1, 1)` and `β = σ(ΔF)`.

There is no global trust score. Each client's β table is derived from its own data, and two honest clients with different data will legitimately assign different β to the same peer.

---

## 2. Test Model — `forum/testmodel/` (Python)

A deterministic, verifiable harness for the protocol flow, requiring no real ML infrastructure. **Implements v2.**

### Data Generator (`generator.py`)

Synthetic time series for a 4-feature model:

| Feature | Pattern | Represents |
|---------|---------|------------|
| x1 | Sine wave | Periodic seasonal effect |
| x2 | Linear drift | Gradual degradation |
| x3 | Square wave | Discrete state changes |
| x4 | Smoothed random walk | Cumulative stochastic process |

Ground truth is known: `y = 0.5·x1 − 0.3·x2 + 1.2·x3 − 0.8·x4 + 0.1 + noise`.

Each simulated node receives a variant dataset determined by a node seed; the same seed always produces the same data. Feature noise simulates sensor imprecision, and a label flip probability simulates misclassification.

### Bayesian Linear Regression (`model.py`)

Mean-field variational inference with per-parameter independent Normal posteriors. Prior `Normal(0, 10²)`, known observation noise `σ² = 0.01`. Coordinate ascent VI updates each parameter's posterior mean and variance independently.

- `fit(X, y)` — run VI on local data
- `natural_parameters()` — encode posterior as `[η₁…, η₂…]`
- `load_natural_parameters(eta)` — decode back to mean/variance
- `prior_natural_parameters()` — the prior's natural parameters

Note the prior is `Normal(0, 10²)`, whose `η₂ = −0.005`. Under v2's zero-vector base prior this regularisation was private and invisible to other nodes; under v3 it belongs in the model card's `η₀`.

### Protocol Operations (`protocol.py`)

- `encode_hex(eta)` / `decode_hex(data)` — float64 LE ↔ hex
- `prior_diff(posterior_eta, prior_eta)` — natural parameter difference
- `aggregate(global_eta, diffs)` — sum diffs
- `simulate_federation(n_nodes, …)` — run a full simulated round

### CLI (`__main__.py`)

Invoked by the Go binary as a subprocess:

```
python -m testmodel --seed 42 --samples 2000 --node 1 --iters 30
```

Emits JSON with `natural_eta` (hex), `post_mean`, `true_parameters`.

### Tests (`test_protocol.py`)

12 tests, all passing: deterministic generation, single-node recovery, natural parameter and hex round-trips, zero diff without training, nonzero diff with data, aggregation correctness, federation convergence (MAE < 0.15 against ground truth), more nodes beating the zero-prior baseline, and more data producing tighter posteriors.

---

## 3. Protocol Handler — `forum/fiorgo/` (Go)

A Go module using the orly Nostr library (`git.smesh.lol/orly` v0.65.60) for event construction, signing, and relay communication. **Implements v2.**

### Package `pkg/fior`

**`kinds.go`** — event kind constants 30100–30105. The numbers survive into v3; four of the six meanings do not.

**`natparam.go`** — natural parameter encode/decode and aggregation:

- `DecodeNaturalParams(hex, dim) → (eta1, eta2, error)`
- `EncodeNaturalParams(eta1, eta2) → hex`
- `Aggregate(globalEta1, globalEta2, diffs)` — sum in place
- `Diff(posteriorEta1, posteriorEta2, priorEta1, priorEta2) → (d1, d2)`

The arithmetic here carries over to v3 unchanged. `Diff` already computes what v3 calls a site contribution; the hex codec is what changes.

### Binary `cmd/e2etest`

Runs the full v2 flow against a live relay:

1. Generate a BIP-340 keypair (secp256k1)
2. Connect via WebSocket
3. Publish a Model Card (30100) for `test-model-v1`
4. Publish an initial Prior (30101) with zero natural parameters
5. For each of N nodes, invoke the Python testmodel as a subprocess and capture the posterior
6. Publish each posterior as a 30102
7. Aggregate all posteriors locally
8. Publish the aggregated Prior (30101, round 1)
9. Compare against ground truth and print MAE

```
FIOR_TESTMODEL_DIR=.. FIOR_RELAY=wss://nos.lol go run ./cmd/e2etest/
```

Verified: MAE 0.031 against ground truth with 3 nodes × 2000 samples.

---

## 4. Docker Stack — `forum/`

Self-contained local test environment. **Implements v2.**

### `docker-compose.e2e.yml`

| Service   | Image                            | Purpose                                |
|-----------|----------------------------------|----------------------------------------|
| `relay`   | `scsibug/nostr-rs-relay:latest`  | Local Nostr relay, SQLite, no auth      |
| `e2etest` | Built from `Dockerfile.e2e`      | Runs the full protocol test            |

```
docker compose -f docker-compose.e2e.yml up --build --abort-on-container-exit
```

`relay-config.toml` configures the relay for testing: no authentication, no rate limiting, all event kinds allowed, port 8080.

### `Dockerfile.e2e`

Multi-stage: `golang:1.25-alpine` builds the Go binary, `python:3.12-slim` installs numpy and copies the testmodel plus the binary. The e2etest binary connects to `ws://relay:8080`.

**v3 requires a third service.** With parameters in blobs, the stack needs a Blossom server, and `cmd/e2etest` needs an upload step. This is the one place where blob-only genuinely costs something: the test model's η is 80 bytes, and it now takes a round trip. The compensation is that the e2e path exercises the same code as production instead of a toy path that only tests exist to use.

---

## 5. Marketplace UI — `forum/src/` (React)

The forum SPA currently handles NIP-07 login and Blossom upload for arbitrary files. `fior/ui-integration.md` maps v3 events to UI state.

The shape differs substantially from v2's design. The marketplace is a **membership roster**, not a feed of submissions, because sites are addressable and there is one per author. There is **no consensus trust score** to display; the UI shows the viewer's own β, computed locally, clearly labelled as local, alongside benchmark results, which are the only cross-client comparable numbers. And because parameters live in blobs, the UI must distinguish a loaded descriptor from fetched parameters and let the user decide when to spend the bandwidth.

The existing Blossom upload code is directly reusable — it is now on the critical path for every participant rather than an auxiliary feature.

---

## 6. v3 Migration

Concrete work to bring the implementation in line with the spec.

### `forum/fiorgo/`

- `kinds.go` — the constants stay; 30101 through 30105 all change meaning. 30103 moves from Node Registration to Benchmark Result, and 30105 from Trust Snapshot to Benchmark Descriptor.
- `natparam.go` — replace the hex codec with raw binary `f64le`/`f32le`/`i16le`/`i8`, add per-index scales and stochastic rounding for the integer forms, and add the blob header. `Aggregate` and `Diff` carry over.
- New: `A(η)` per family, and the BMR `ΔF` evaluation. This is the core of the trust engine and has no v2 equivalent.
- New: Blossom upload and fetch with SHA-256 verification.
- New: β resolution — the Bernoulli–Beta combination, the hierarchical attestation prior, and transitive propagation.
- `cmd/e2etest` — restructure around per-client composition rather than a central aggregation step. Each simulated node composes its own prior, publishes a site, and scores its peers.

### `forum/testmodel/`

- Publish `Δη` rather than the full posterior. `prior_diff` already computes it; what changes is which quantity is published.
- Move the `Normal(0, 10²)` prior into the model card's `η₀` so peers can see and reproduce the regularisation.
- New tests worth having, given v3's specific claims: that per-client composition with all β = 1 reproduces the v2 aggregate; that quantization error with stochastic rounding falls as 1/√N across members; that a poisoned site receives ΔF < 0 in the round it appears; and that ρ backtracking fires when a composed group leaves its domain.

### `forum/` stack

- Add a Blossom server service to `docker-compose.e2e.yml`.
- Add blob upload to the e2e flow, before event publication.

### Validation the spec now requires of clients

- Decoded vector length against ONNX initializer shapes times the family's value count
- Blob SHA-256 on fetch
- Model card `version` on every site before composing it
- Blob header `group_count` against the model card

---

## 7. Open Work

**Production model support.** The test model is Bayesian linear regression with closed-form VI updates. Real models need a Pyro/NumPyro backend, stochastic variational inference where posteriors are not analytically tractable, local GPU training, and ONNX import for arbitrary graphs.

**Trust engine.** BMR scoring, the Bernoulli–Beta combination, and hierarchical plus transitive attestation resolution. Adversarial testing should verify the specific claim the spec makes: that a poisoned site is suppressed in the round it is published, without any coordinated action, because each evaluator scores it against its own free energy.

**Marketplace UI.** Wire the React app to the v3 event kinds per the integration guide.

**Buzz relay integration.** The client operates a Buzz relay at `buzz.nostri.xyz`. Buzz requires authentication, so participants will need invites or API keys, and the client will need NIP-42.

**NIP submission.** The kind numbers 30100–30105 were verified unused against `nostr-protocol/nips` for v2, but the semantics have changed entirely and the check should be repeated before submission. The `fior` NIP-11 extension object should be included in the proposal, since it is how a relay declares support.

**Open question the spec records rather than solves.** Transitive trust discounts a peer's attestations by `E[β]` — trust in that peer's *parameters* — used as a proxy for trust in their *judgement about others*. These are not the same quantity. Separating them would require a second Beta per edge.
