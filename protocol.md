# FIOR Protocol Specification v3

## Overview

Federated Inference Over Relays. Nodes train Bayesian models on local private data and exchange the natural parameters of their posterior distributions over Nostr. Model structure is defined using ONNX. Distribution metadata layers on top of ONNX initializer tensors.

There is no aggregator. Every participant computes its own view of the global posterior locally, from whichever peers it chooses to weight and however it chooses to weight them. Some participants may choose to publish their combinations and benchmark them; that is a service, not a role the protocol depends on.

## Roles

The protocol recognises exactly two roles.

**FIOR relay.** A NIP-01 relay that accepts the FIOR kind block, advertises support in its NIP-11 document, and enforces size and rate policy. It MAY reject structurally malformed events — wrong tag arity, invalid base64, declared size mismatch. It MUST NOT compute, weight, combine, or otherwise alter parameters. A relay is storage and transport.

**FIOR client.** Everything else: fetching model cards and ONNX graphs, running local inference, maintaining a private trust table, building its own prior, publishing contributions, and optionally publishing combinations, benchmarks, and trust attestations.

Aggregating, benchmarking, and indexing are things a client may choose to do. No participant holds state that another participant must trust to make progress.

## Why ONNX

ONNX is the standard interchange format for ML models. It defines a directed acyclic computation graph, initializer tensors, versioned operator specifications, and tooling to export from PyTorch, TensorFlow, scikit-learn, JAX, and others.

FIOR does not reinvent model description. An `.onnx` file defines the inputs, outputs, parameter names, shapes, and computation graph. FIOR layers on top:

1. **Distribution assignments** — which exponential family distribution models each initializer tensor
2. **Natural parameter wire format** — the federated exchange of those distributions' natural parameters
3. **Composition rules** — how a client builds its own prior from peers' contributions
4. **Trust** — how a client decides how much precision to assign each peer

---

## Mathematical Model

### Sites, not posteriors

Each node publishes a **site contribution** — its likelihood approximation — rather than its posterior:

```
Δη_n = η_n^post − η_n^cavity
```

where `η_n^cavity` is the prior the node trained against. This is the quantity that composes. Publishing full posteriors instead would count the base prior once per member: `Σ_n η_n = N·η₀ + Σ_n Δη_n`, over-confident by `(N−1)·η₀`. The two coincide only when `η₀ = 0`, which forces an improper base prior and pushes regularisation into each node's private, unreproducible policy.

### Composition

Every client builds its own prior locally:

```
η_A^prior,g = η₀,g + ρ · Σ_{n ∈ M, n ≠ A}  β_{A→n,g} · Δη_{n,g}
```

- `η₀` — base prior, carried in the model card, so every client starts from the same point without a coordinator
- `M` — the members client A chooses to consider
- `β_{A→n,g} ∈ (0,1)` — the precision A assigns to peer n's contribution to group g
- `ρ ∈ (0,1]` — global damping (see below)

Sites are addressable events keyed by `(author, model)`, so a relay stores exactly one per author and the latest supersedes the previous. That is precisely the expectation-propagation invariant "the latest site replaces the previous one," which means summing latest-per-author double-counts nothing and requires no coordination.

Setting `M` to all known peers and every `β` to 1 recovers unweighted summation.

### What β means

For a Normal site, `η₁ = μ/σ²` and `η₂ = −1/(2σ²)`. Scaling both by β leaves the implied mean invariant:

```
−η₁'/(2η₂') = μ           for any β > 0
σ²' = σ²/β
```

So β adjusts **how much you believe a peer, not what you think they said**. β → 0 approaches ignoring them; β = 1 accepts at face value. This is why β multiplies natural parameters rather than posterior means.

### Validity and damping

The composed result must lie in the natural parameter domain of its family:

| Family      | Constraint            |
|-------------|-----------------------|
| Normal      | `η₂ < 0`              |
| Gamma       | `η₁ > −1`, `η₂ < 0`   |
| Dirichlet   | `η_i > −1`            |
| Categorical | unconstrained         |

If the composition violates its domain, the client backtracks `ρ ← ρ/2` until valid. Clients SHOULD keep `ρ ≤ 1` even when valid, as standard EP damping against round-to-round oscillation. The `ρ` actually applied MUST be recorded in any published Composite, or recomputation will not reproduce.

---

## Event Kinds

All six are addressable (NIP-01, kinds 30000–39999): a relay stores one event per `(kind, pubkey, d)` and the latest supersedes.

| Kind  | Name                 | `d`                              | Published by                              |
|-------|----------------------|----------------------------------|-------------------------------------------|
| 30100 | Model Card           | `<model-id>`                     | Creator — ONNX blob, distribution map, η₀ |
| 30101 | Site Contribution    | `<model-id>`                     | Any node — its Δη                         |
| 30102 | Composite            | `<composite-id>`                 | Any node — member set, β, resulting η     |
| 30103 | Benchmark Result     | `<bench-id>:<target-event-id>`   | Any node — metrics for a site or composite|
| 30104 | Trust Attestation    | `<target-hex>[:<model>[:<group>]]` | Any node — Beta parameters for β        |
| 30105 | Benchmark Descriptor | `<bench-id>`                     | Any node — what a benchmark measures      |

There is no registration kind. Publishing a site **is** joining; a NIP-09 deletion or an expired site **is** leaving. There is no registry, so nothing can disagree about who is a member.

There is no round counter. Ordering is `created_at`, and provenance is the cavity each site pins by reference.

Kinds are specified below in the order 30100, 30101, 30102, 30103, 30105, then 30104. The Trust Attestation is documented after the [Trust Layer](#trust-layer), because its `beta` tag is meaningless without it.

---

## 30100 — Model Card

Defines a model. The `d` tag carries the model identifier.

### Tags

| Tag       | Required | Description                                              |
|-----------|----------|----------------------------------------------------------|
| `d`       | yes      | Model identifier (slug, e.g. `pump-failure-v1`)          |
| `title`   | yes      | Human-readable model name                                |
| `version` | yes      | Model card version; MUST increment on any change to the group set, group order, or distribution assignments |
| `summary` | no       | Short description                                        |
| `onnx`    | yes      | ONNX blob reference                                      |
| `dist`    | yes*     | Distribution assignment for an ONNX initializer tensor   |
| `group`   | yes*     | Named group of initializers sharing a distribution       |
| `eta`     | no       | Base prior `η₀` per group (inline mode)                  |
| `x`       | no       | Base prior `η₀` blob reference (blob mode)               |
| `ttl`     | no       | Seconds after which a site without its own `expiration` is considered stale |

`*` At least one `dist` or `group` tag.

If no `eta`/`x` is present, `η₀` is the zero vector. Publishing an explicit proper base prior is RECOMMENDED — a zero vector means `σ² = ∞`, and every node then has to regularise privately in a way no other node can see or reproduce.

### `onnx` Tag Format

```
["onnx", "<sha256-hex>", "<bytes>", "<server-root>", ...]
```

The blob is addressed by SHA-256 per Blossom BUD-01: fetch from `<server-root>/<sha256-hex>`. Server roots are hints; any Blossom server holding the hash serves the same bytes. Clients MUST verify the hash on fetch.

### `dist` Tag Format

```
["dist", "<init_name>", "<family>"]
```

| Field       | Type   | Description                                                     |
|-------------|--------|-----------------------------------------------------------------|
| `init_name` | string | ONNX initializer tensor name (e.g. `fc1.weight`)                |
| `family`    | string | `normal`, `gamma`, `dirichlet`, `cat`, `beta`                   |

### `group` Tag Format

Groups multiple initializer tensors under a shared distribution, reducing wire overhead when parameters share a family and can be concatenated into a single vector.

```
["group", "<name>", "<family>", "<init1>", "<init2>", ...]
```

A special group name `*` means "all initializers not otherwise assigned, with this family."

**Group order is significant.** In blob mode, groups are concatenated in the order their `group`/`dist` tags appear in the model card. Reordering or adding groups therefore requires a `version` increment, and sites carry the version they were built against.

### Content

Optional JSON with extended metadata.

### Example

```json
{
  "kind": 30100,
  "tags": [
    ["d", "pump-failure-v1"],
    ["title", "Pump Failure Predictor"],
    ["summary", "Predicts centrifugal pump failure within 7 days from sensor readings"],
    ["version", "3"],
    ["onnx", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "48192", "https://blossom.example"],
    ["group", "fc-layers", "normal", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    ["group", "convs", "normal", "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"],
    ["eta", "fc-layers", "f64le.b64", "AAAAAAAA8L8AAAAAAADwvw=="],
    ["eta", "convs", "f64le.b64", "AAAAAAAA8L8AAAAAAADwvw=="],
    ["ttl", "2592000"]
  ],
  "content": "{\"framework\":\"pyro\",\"onnx_opset\":18,\"min_samples\":500,\"license\":\"MIT\"}"
}
```

---

## 30101 — Site Contribution

A node's likelihood approximation `Δη`. One per `(author, model)`; the latest supersedes.

### Tags

| Tag          | Required | Description                                                        |
|--------------|----------|--------------------------------------------------------------------|
| `d`          | yes      | Model identifier                                                   |
| `a`          | yes      | Model card coordinate, marker `model`                              |
| `v`          | yes      | Model card version this site was built against                     |
| `a`          | no       | Composite used as cavity, marker `cavity`; absent means `η₀` alone  |
| `eta`        | yes*     | `Δη` per group (inline mode)                                       |
| `x`          | yes*     | `Δη` blob reference (blob mode)                                    |
| `expiration` | no       | NIP-40 expiration; RECOMMENDED                                     |

`*` Exactly one of `eta` (one or more) or `x`.

```
["a", "30100:<creator-hex>:<model-id>", "<relay-hint>", "model"]
["a", "30102:<author-hex>:<composite-id>", "<relay-hint>", "cavity"]
```

The `cavity` reference is what makes `Δη = η^post − η^cavity` auditable: anyone can fetch the pinned composite and check the arithmetic.

Clients MUST reject a site whose `v` does not match the model card version they are using, and MUST reject a site whose decoded vector length disagrees with the group dimensions implied by the ONNX initializer shapes.

### Content

JSON with local training metadata. All fields optional and advisory — none of them are inputs to composition or to trust.

```json
{
  "samples": 8234,
  "free_energy": -1247.3,
  "duration_sec": 42.1,
  "framework_version": "pyro-1.9.0"
}
```

### Example

```json
{
  "kind": 30101,
  "tags": [
    ["d", "pump-failure-v1"],
    ["a", "30100:c0ffee...:pump-failure-v1", "", "model"],
    ["v", "3"],
    ["a", "30102:deadbeef...:pump-failure-v1-tw-0811", "", "cavity"],
    ["eta", "fc-layers", "i8.b64", "f4mHhIB7dnFsZ2I=", "3.05e-4", "1.12e-2"],
    ["eta", "convs", "i8.b64", "AQIDBAUGBwgJCgsM", "8.81e-5", "4.30e-3"],
    ["expiration", "1786752000"]
  ],
  "content": "{\"samples\":8234,\"free_energy\":-1247.3}"
}
```

### Withdrawal

A node leaves by publishing a NIP-09 kind 5 deletion for its site, or by letting the site expire. No other participant needs to act; each client's next composition simply omits it.

---

## 30102 — Composite

A specific β-weighted combination, recorded so it can be reproduced and benchmarked. A Composite is a **receipt for an arithmetic operation**, not a statement of belief.

Composites serve two purposes: they pin the cavity a node trained against, and they give benchmarkers something concrete to measure. A Composite's β vector is frequently *counterfactual* — uniform-weight baselines and leave-one-out ablations are the most useful things to benchmark, and in an ablation `β = 0` for a member is the second arm of an experiment, not an accusation.

### Composites are not composable

A Composite carries no data-dependent likelihood term. It MUST NOT be summed as a site. Its publisher contributes a measurement, not evidence.

### Tags

| Tag       | Required | Description                                                     |
|-----------|----------|-----------------------------------------------------------------|
| `d`       | yes      | Composite identifier (author-chosen slug)                       |
| `a`       | yes      | Model card coordinate, marker `model`                           |
| `v`       | yes      | Model card version                                              |
| `m`       | yes*     | Member entry                                                    |
| `p`       | yes*     | Member pubkey, one per member (indexed, enables reverse lookup) |
| `rho`     | yes      | Damping factor actually applied                                 |
| `purpose` | no       | `trust-weighted`, `uniform`, `ablation`, or free text           |
| `eta`     | yes*     | Resulting `η` per group (inline mode)                           |
| `x`       | yes*     | Resulting `η` blob reference (blob mode)                        |

### `m` Tag Format

```
["m", "<pubkey-hex>", "<site-event-id>", "<β_g1,β_g2,...>"]
```

| Field       | Type   | Description                                                              |
|-------------|--------|--------------------------------------------------------------------------|
| `pubkey`    | string | Member's public key, hex                                                 |
| `site id`   | string | Event id of the **exact site version** included                          |
| `β vector`  | string | Comma-separated β per group, in model card group order; a single value means the same β for every group |

The site is pinned by event id, not by `a` coordinate, because sites are replaceable — an `a` coordinate moves, and a benchmark score against a moving target means nothing.

### Content

Optional JSON.

### Example

An ablation probing what happens without member C:

```json
{
  "kind": 30102,
  "tags": [
    ["d", "pump-failure-v1-ablate-C-0811"],
    ["a", "30100:c0ffee...:pump-failure-v1", "", "model"],
    ["v", "3"],
    ["m", "aaa1...", "e1e1e1...", "1.0"],
    ["m", "bbb2...", "e2e2e2...", "1.0"],
    ["m", "ccc3...", "e3e3e3...", "0.0"],
    ["p", "aaa1..."],
    ["p", "bbb2..."],
    ["p", "ccc3..."],
    ["rho", "1.0"],
    ["purpose", "ablation"],
    ["x", "5f3a...", "f64le.b64", "196608", "https://blossom.example"]
  ],
  "content": "{\"note\":\"leave-one-out probe for member C\"}"
}
```

Anyone can fetch the three pinned sites, recompute `η₀ + ρ Σ βΔη`, and verify the published result byte-for-byte.

---

## 30103 — Benchmark Result

A measurement of a site or a composite against a named benchmark. This is the sole measurement kind — it scores individual contributions and whole combinations through the same mechanism, so one leaderboard covers both.

### Tags

| Tag       | Required | Description                                                   |
|-----------|----------|---------------------------------------------------------------|
| `d`       | yes      | `<bench-id>:<target-event-id>`                                |
| `a`       | yes      | Benchmark descriptor coordinate, marker `bench`               |
| `a`       | yes      | Target coordinate (30101 or 30102), marker `target`           |
| `e`       | yes      | Exact target event id                                         |
| `p`       | yes      | Target author pubkey                                          |
| `metric`  | yes*     | One measured metric                                           |
| `samples` | no       | Evaluation sample count                                       |

### `metric` Tag Format

```
["metric", "<name>", "<value>", "<stderr>"]
```

The metric name MUST be one declared by the referenced descriptor. Results are only comparable within a descriptor.

### Content

JSON describing methodology.

### Example

```json
{
  "kind": 30103,
  "tags": [
    ["d", "pump-q3-2025:e3e3e3..."],
    ["a", "30105:f00d...:pump-q3-2025", "", "bench"],
    ["a", "30102:deadbeef...:pump-failure-v1-ablate-C-0811", "", "target"],
    ["e", "e3e3e3..."],
    ["p", "deadbeef..."],
    ["metric", "log_lik", "-1198.7", "14.2"],
    ["samples", "1647"]
  ],
  "content": "{\"runner\":\"fiorgo-0.3\",\"wall_sec\":31.4}"
}
```

---

## 30105 — Benchmark Descriptor

Defines what a benchmark measures, so that results are comparable.

This kind exists because scores measured on different private holdouts are not on the same scale. A log-likelihood delta on one plant's Q3 sensor data and one on another plant's data cannot be meaningfully averaged. **Clients MUST only aggregate Benchmark Results that share a descriptor.**

### Tags

| Tag      | Required | Description                                         |
|----------|----------|-----------------------------------------------------|
| `d`      | yes      | Benchmark identifier                                |
| `a`      | yes      | Model card coordinate, marker `model`               |
| `title`  | yes      | Human-readable name                                 |
| `set`    | yes      | Evaluation set definition                           |
| `metric` | yes*     | A metric this benchmark reports                     |

### `set` Tag Format

Public, reproducible by anyone:

```
["set", "public", "<sha256-hex>", "<bytes>", "<server-root>", ...]
```

Private holdout, reproducible only by parties holding equivalent data:

```
["set", "private", "<protocol-name>"]
```

A `private` descriptor still makes results comparable *within one publisher's* series, which is the honest limit of what a private holdout can support. Clients SHOULD display results from `private` descriptors as attributable to their publisher rather than as consensus.

### `metric` Tag Format

```
["metric", "<name>", "<direction>"]
```

`direction` is `higher` or `lower`, indicating which way is better.

### Example

```json
{
  "kind": 30105,
  "tags": [
    ["d", "pump-q3-2025"],
    ["a", "30100:c0ffee...:pump-failure-v1", "", "model"],
    ["title", "Pump sensors, Q3 2025 holdout"],
    ["set", "public", "1b4f0e...", "8412330", "https://blossom.example"],
    ["metric", "log_lik", "higher"],
    ["metric", "brier", "lower"]
  ],
  "content": "{\"window\":\"2025-07-01/2025-09-30\",\"rows\":16470}"
}
```

---

## Natural Parameter Encoding

### Encoding token

```
["eta", "<group>", "<enc>", "<payload>", "<scale₁>", "<scale₂>", ...]
```

`<enc>` is `<dtype><endian>.<transport>`:

| Token       | Bytes/value | Notes                                        |
|-------------|-------------|----------------------------------------------|
| `f64le.b64` | 8           | Default                                      |
| `f32le.b64` | 4           | Opt-in per group                             |
| `i16.b64`   | 2           | Requires scales                              |
| `i8.b64`    | 1           | Requires scales                              |
| `f64le.hex` | 8           | Legacy/debugging; 2× the size of base64      |

Base64 is standard-alphabet with padding. It is 4/3 expansion against hex's 2×, so `f64le.b64` is 33% smaller than v2's hex for no loss.

### Natural parameter layout per family

For a group with `d` scalar parameters:

| Family      | Values | Layout                                                        |
|-------------|--------|---------------------------------------------------------------|
| `normal`    | `2d`   | first `d`: `η₁ᵢ = μᵢ/σᵢ²`; last `d`: `η₂ᵢ = −1/(2σᵢ²)`         |
| `gamma`     | `2d`   | first `d`: `η₁ᵢ = αᵢ − 1`; last `d`: `η₂ᵢ = −βᵢ`               |
| `beta`      | `2d`   | first `d`: `η₁ᵢ = aᵢ − 1`; last `d`: `η₂ᵢ = bᵢ − 1`            |
| `dirichlet` | `k`    | `ηᵢ = αᵢ − 1`                                                  |
| `cat`       | `k`    | `ηᵢ = log P(category i)`                                       |

For a site, these are *differences* `Δη`, not posteriors.

### Quantization

Integer encodings carry one scale per natural-parameter index of the family — two for `normal`, `gamma`, and `beta`; one for `dirichlet` and `cat`. The count is implied by the family in the model card, so the tag is self-describing. Dequantization is `value = scale × q`.

Per-index scaling is required, not cosmetic: `η₂ = −1/(2σ²)` spans orders of magnitude between a sharply determined parameter and a barely identified one, and a single global scale cannot cover both natural-parameter indices at once.

**Publishers MUST use stochastic rounding when quantizing.** Federated summation is unusually quantization-tolerant — with N members, independent zero-mean errors add in quadrature (~√N) while the signal grows ~N, so relative error in the composed prior falls as 1/√N. That property depends entirely on unbiasedness. Deterministic round-to-nearest is biased, and bias accumulates linearly alongside the signal rather than cancelling.

Two consequences worth stating plainly:

- Quantization noise near `η₂ ≈ 0` can push the composed result out of its domain. The `ρ` backtracking above handles it; low precision simply costs more damping.
- **The protocol sets no precision floor.** `ΔF` (below) is computed on the quantized `Δη` — the exact bytes a consumer would sum — so an over-quantized site stops helping, its `ΔF` goes negative, and its β falls. Each consumer prices the publisher's bandwidth/accuracy trade-off independently. Precision is a publisher choice, not a conformance requirement.

Wire precision and compute precision are separate concerns: `ΔF` is a difference of differences of `A(η)` and is prone to catastrophic cancellation, so clients MUST dequantize to f64 and evaluate `A()` in f64 regardless of how a site arrived.

### Inline and blob modes

`f64le` Normal posteriors cost 16 bytes per parameter, ≈21.3 characters base64. Against a conservative 32 KiB event budget that is roughly **1,500 parameters inline** at f64, or **12,000 at i8**. Real models exceed both, so blob mode is the normal path, not an exception.

**Inline mode** — one `eta` tag per group.

**Blob mode** — one `x` tag:

```
["x", "<sha256-hex>", "<enc>", "<bytes>", "<server-root>", ...]
```

The blob holds every group concatenated in model card order. Group boundaries derive from the ONNX initializer shapes and the family's value count, so nothing extra is transmitted. Integer encodings put their scales in a header: `<u16 group count>` followed by, per group, `<u8 scale count><f64le scales...>`.

Presence of `x` selects blob mode. The two modes MUST NOT be mixed within one event. Clients MUST verify the blob SHA-256 on fetch.

---

## Trust Layer

β is computed locally and privately by every client. Publishing an attestation is a separate, voluntary act. A node that publishes no attestations and reads none is a fully functional participant — the social layer is a cold-start and bandwidth convenience, never a dependency.

### The model

Peer inclusion is a Bernoulli indicator with a Beta prior:

```
z_{n,g} ~ Bernoulli(π_{n,g})     does peer n's group-g site enter my prior?
π_{n,g} ~ Beta(a, b)             prior, from attestations
ΔF_{n,g}                          log Bayes factor, from BMR, recomputed each round
```

Posterior odds are prior odds times the Bayes factor, so:

```
β_{n,g} = E[z_{n,g} | my data] = 1 / (1 + (b/a)·exp(−ΔF_{n,g}))
```

The two inputs have cleanly separated jobs:

- **`ΔF` is recomputed every round and never stored.** Sites are replaceable, so `Δη_n` at round `t+1` is a different object than at round `t`; evidence about a superseded site is evidence about something that no longer exists. BMR's `ΔF` is exact given the inputs, with no sampling noise to pool away, so nothing justifies an accumulator here. This also removes the earn-then-defect attack: a defecting peer's β collapses the same round, with no banked history to spend.
- **`(a, b)` accumulates**, because it summarises other participants' reports, which a client cannot recompute for itself. It is a prior only. A few nats of direct evidence swamps a handful of pseudo-counts from strangers, which is the correct ordering.

With no attestations at all, `(a, b) = (1, 1)` and `β = σ(ΔF)` — pure self-evaluation.

Because the logistic never reaches its asymptotes, `β ∈ (0,1)` strictly. Every member always contributes slightly and is therefore always being re-tested, so no peer is ever permanently excluded; a peer that starts contributing usefully again recovers on its own. Clients MAY hard-threshold `β < ε` to zero to avoid fetching a large blob, accepting that the member goes untested until the threshold is revisited.

### Mechanism 1 — Bayesian model reduction

Let `η_p` be the client's working prior (the composition above), `η_q` its local posterior fitted under `η_p` on private data. Removing peer n offsets prior and posterior by the same amount, giving a closed form:

```
ΔF_{n,g} = A(η_q,g) + A(η_p,g − β_{n,g}Δη_{n,g}) − A(η_q,g − β_{n,g}Δη_{n,g}) − A(η_p,g)
```

`ΔF > 0` means including n at its current weight raises the log evidence.

`A(η)` is the log-partition function, already tabulated for every supported family in [distributions.md](distributions.md), so this introduces no new mathematics. Because groups are independent blocks, `A` decomposes as a sum over them and `ΔF_{n,g}` follows by restricting to that group's slice — which is what gives trust its per-group resolution.

The cost is **two extra evaluations of `A(η)` per peer per group**, on parameters the client already holds. No holdout split, no retraining, no second pass over data. One local fit per round serves both inference and the evaluation of every peer.

Across rounds this is a fixed-point iteration: `β⁽⁰⁾ = a/(a+b)`, then each round's `ΔF` yields the next `β`, which shapes the next round's prior.

### Mechanism 2 — Hierarchy

Attestations address β at three levels — `peer`, `peer:model`, `peer:model:group`. Because the kind is addressable and the `d` values differ, all three coexist rather than replacing each other.

Coarse levels are not a first-hit fallback; they are a **hierarchical prior**, with the parent supplying discounted pseudo-counts to the child. In Beta natural parameters `η^β = (a−1, b−1)`:

```
η^β_eff(B,m,g) = η^β_own(B,m,g) + γ · η^β_eff(B,m)
η^β_eff(B,m)   = η^β_own(B,m)   + γ · η^β_eff(B)
η^β_eff(B)     = η^β_own(B)     + η^β_0
```

`γ ∈ [0,1]` is the pooling factor: `γ = 0` is fully qualified per-group with no cold-start help, `γ = 1` is full pooling. Partial pooling lets a peer with a track record on one model start a new one with a real but discounted prior, while per-group evidence overrides it as it arrives.

This applies only to `(a, b)`. Direct evidence is exact per group and shares no strength across levels.

### Mechanism 3 — Transitivity

Attestations from peers a client already trusts enter as prior pseudo-counts, discounted by that trust and clipped per contributor:

```
η^β_trans(A→C) = Σ_B  E[β_{A→B}] · clip( η^β_pub(B→C), κ_max )
```

which adds into the hierarchy above as another prior term. Depth 1 is the default and is cycle-free by construction; multi-hop propagation with product discounting is local policy, not protocol.

A known simplification, recorded rather than solved: `E[β_{A→B}]` measures trust in B's *parameters*, and is used here as a proxy for trust in B's *judgment about others*. These are not the same quantity. Separating them would require a second Beta per edge.

### Mechanism 4 — Attestations as evidence

Binary likes and dislikes are `(a,b) += (1,0)` and `(0,1)` — the integer special case of the same event, requiring no separate kind. Publishing a continuous β means publishing `(a, b)` where `a + b` encodes confidence.

A benchmarked leave-one-out Composite pair against a shared descriptor is the *measured* counterpart of `ΔF`, and updates `(a, b)` through the same rule. Three mechanisms, one accumulator.

### The loop

```
attestations ──(one input to)──▶ composite ──(benchmarked)──▶ result
      ▲                                                         │
      └────────────── evidence updates (a, b) ◀─────────────────┘
```

Trust beliefs are one input to constructing a composite; benchmarking composites — especially ablations — is evidence that updates the Beta prior. The analytic `ΔF` and the measured benchmark estimate the same quantity, so a client can use the cheap closed form locally while a benchmarker publishes the expensive measured version, and both update the same `(a, b)`.

### Sybil resistance

Direct evidence `ΔF` grows in influence without bound as rounds accumulate, while the transitive channel is discounted by `E[β] ≤ 1`, clipped at `κ_max` per contributor, and open only to peers already trusted. Direct evidence therefore dominates asymptotically.

More fundamentally, minting N identities buys nothing: each is scored on its own contribution to *the evaluator's* free energy, so a Sybil earns β only by actually helping. The residual attack is earn-then-defect, which per-round recomputation of `ΔF` bounds to a single round.

### Trust is not consensus

There is no global trust score and no snapshot event. Each client's β table is its own, derived from its own data. Two honest clients with different data will legitimately assign different β to the same peer. Benchmark Results give the marketplace a comparable, verifiable signal to display; they do not define anyone's β.

---

## 30104 — Trust Attestation

An optional, voluntary publication of a client's Beta parameters for a peer. Carries a *belief about a peer*: no member set, no η, no version pinning.

### Tags

| Tag          | Required | Description                                                    |
|--------------|----------|-----------------------------------------------------------------|
| `d`          | yes      | `<target-hex>`, `<target-hex>:<model-id>`, or `<target-hex>:<model-id>:<group>` |
| `p`          | yes      | Target pubkey (indexed)                                        |
| `a`          | no       | Model card coordinate, marker `model`, when model-scoped       |
| `beta`       | yes      | Beta parameters                                                |
| `expiration` | no       | NIP-40 expiration; RECOMMENDED                                 |

### `beta` Tag Format

```
["beta", "<a>", "<b>"]
```

Both strictly positive floats. `E[β] = a/(a+b)`; `Var[β] = ab/((a+b)²(a+b+1))`. The Beta natural parameters are `(a−1, b−1)`, which is the form transitive propagation sums.

A binary like is `["beta", "2", "1"]`; a dislike is `["beta", "1", "2"]`.

### Content

Optional JSON.

```json
{"basis": "bmr", "rounds": 12}
```

### Example

```json
{
  "kind": 30104,
  "tags": [
    ["d", "bbb2...:pump-failure-v1:fc-layers"],
    ["p", "bbb2..."],
    ["a", "30100:c0ffee...:pump-failure-v1", "", "model"],
    ["beta", "14.2", "3.1"],
    ["expiration", "1789430400"]
  ],
  "content": "{\"basis\":\"bmr\",\"rounds\":12}"
}
```

`E[β] = 0.82`, sd 0.09.

---

## Relay Conformance

A FIOR relay advertises support in its NIP-11 document. This is the entire mechanism by which a relay declares it supports the marketplace — no new role or handshake is introduced.

```json
{
  "supported_nips": [1, 9, 11, 40, 42, 45],
  "limitation": { "max_message_length": 262144, "max_event_tags": 2000 },
  "fior": {
    "version": "3",
    "kinds": [30100, 30101, 30102, 30103, 30104, 30105],
    "models": ["pump-failure-v1"],
    "max_eta_bytes": 65536,
    "validates": ["structure"],
    "blossom": ["https://blossom.example"]
  }
}
```

`models` omitted means any model is accepted; present means an allowlist.

### Validation split

- **Clients MUST** verify decoded vector lengths against group dimensions derived from ONNX initializer shapes times the family's value count; verify blob SHA-256 on fetch; verify the model card `version` of every site they compose.
- **Relays MAY** perform cheap structural checks only: tag arity, base64 validity, declared size agreement, `d` well-formedness. A relay cannot be expected to parse ONNX, and per the roles above it must never touch the numbers.

---

## Query Patterns

| Need                          | Filter                                                       |
|-------------------------------|--------------------------------------------------------------|
| Browse models                 | `{kinds:[30100]}`                                            |
| All sites for a model         | `{kinds:[30101], "#d":["pump-failure-v1"]}`                  |
| One peer's current site       | `{kinds:[30101], authors:[B], "#d":["pump-failure-v1"]}`     |
| Composites including B        | `{kinds:[30102], "#p":[B]}`                                  |
| Attestations about B          | `{kinds:[30104], "#p":[B]}`                                  |
| Leaderboard for a benchmark   | `{kinds:[30103], "#a":["30105:<pk>:<bench-id>"]}`            |
| Benchmarks of a composite     | `{kinds:[30103], "#a":["30102:<pk>:<composite-id>"]}`        |

Only single-letter tags are indexed under NIP-01, which is why event and coordinate references use `e`, `a`, and `p` rather than descriptive names.

Benchmark Results carry **two** `a` tags — descriptor and target — rather than encoding the benchmark id in `d`, because `#d` is exact-match with no prefix support, and `d = <bench>:<target>` would leave the leaderboard unqueryable.

NIP-01 filters OR within a key and cannot AND across keys, so "benchmarks of composite X *against* descriptor D" is a fetch-by-one, filter-client-side operation.

---

## Flow Summary

```
1. Creator exports the model to ONNX, uploads it to a Blossom server
2. Creator publishes a Model Card (30100) with the blob hash, group map, and η₀
3. A node fetches the model card and ONNX graph
4. The node composes its prior: η₀ + ρ Σ β Δη over whichever peers it chooses
   (first round: no peers, so the prior is η₀)
5. The node trains locally on private data
6. The node publishes its site (30101), pinning the cavity it trained against
7. Each client independently recomputes ΔF per peer per group via BMR,
   updates β, and recomposes its prior
8. Repeat from step 5
```

Steps 4 and 7 happen independently on every client. Nothing in this loop requires any participant to wait on any other.

Optional, in parallel:

```
A. Any node publishes a Composite (30102) — its own combination, a uniform
   baseline, or a leave-one-out ablation
B. Any node publishes a Benchmark Descriptor (30105) defining a comparable
   evaluation
C. Any node publishes Benchmark Results (30103) measuring sites or composites
D. Any node publishes Trust Attestations (30104)
```

---

## Security

- All events are Nostr-signed by the publishing keypair.
- Clients verify site vector lengths against ONNX initializer shapes from the model card, and reject on mismatch.
- Clients verify blob integrity by SHA-256 before use.
- Clients verify a site's model card `version` before composing it.
- No participant is obliged to accept any other participant's parameters. β is local, and `β → 0` costs nothing to apply.
- Poisoned parameters are suppressed by the same mechanism that weights honest ones: a site that lowers the evaluator's log evidence gets `ΔF < 0` and a β below the prior mean, in the round it is published.
- Trust is subjective and per-client. There is no consensus score to capture.
- Composites and Benchmark Results are fully reproducible from pinned event ids, so a false claim is falsifiable by anyone.

---

## Changes from v2

### Structural

| v2                                     | v3                                                      |
|----------------------------------------|----------------------------------------------------------|
| Aggregator computes and broadcasts the global prior | Every client composes its own prior locally |
| 30101 Prior Broadcast (aggregator)     | 30102 Composite (anyone, non-composable, reproducible)   |
| 30102 Posterior Submit (full posterior)| 30101 Site Contribution (`Δη`)                           |
| 30103 Node Registration                | removed — publishing a site is joining                   |
| 30104 Trust Evaluation (back-test)     | 30103 Benchmark Result (against a named descriptor)      |
| 30105 Trust Snapshot (aggregator)      | removed — no consensus score exists                      |
| —                                      | 30104 Trust Attestation (Beta parameters)                |
| —                                      | 30105 Benchmark Descriptor (makes scores comparable)     |
| `round` counter                        | removed — no global rounds                               |
| `agg` tag                              | removed                                                  |

### Nostr conformance fixes

| v2                              | v3                       | Reason                                                                 |
|---------------------------------|--------------------------|------------------------------------------------------------------------|
| 30102 treated as an accumulating list | one site per (author, model), latest wins | 30000–39999 is addressable; a conformant relay drops the others |
| 30105 declared non-replaceable  | n/a                      | Same — the kind range makes that impossible                            |
| `["η", ...]`                    | `["eta", ...]`           | Non-ASCII tag key is a cross-implementation hazard and was never indexable |
| `["p", "<event-id>"]`           | `["a", ...]` / `["e", ...]` | `p` is a pubkey reference in NIP-01 and relays index it as one       |
| `["post", "<event-id>"]`        | `["e", ...]` + `["a", ...]` | Multi-letter tags are not indexed, so evaluations were unqueryable  |
| `["agg", "npub1..."]`           | removed                  | Tags carry 32-byte hex; npub is NIP-19 display encoding                |
| `["onnx", "<nostr event id>"]`  | `["onnx", "<sha256>", ...]` | Blossom addresses blobs by SHA-256, not event id                    |
| "NIP-33"                        | NIP-01                   | NIP-33 was merged into NIP-01                                          |
| hex-only `η` encoding           | base64 default, f32/i16/i8 available | 33% smaller at no loss; up to 8× with quantization         |
| unbounded event size            | inline/blob modes with an explicit threshold | Typical relay caps are 64–256 KiB          |

### Trust layer

v2 scored contributions by back-testing: split local data, retrain against the peer's posterior, evaluate both on a holdout, publish per-group deltas. v3 replaces the default path with Bayesian model reduction — a closed form in `A(η)`, computable from parameters already held, with no holdout split and no retraining. Back-testing survives as Benchmark Results, now anchored to a descriptor so results are comparable.

v2 averaged trust scores across evaluators using different private holdouts, which are not on a common scale. v3 forbids aggregating results across descriptors.

v2 planned consensus-weighted aggregation at the aggregator. v3 has no aggregator and no consensus: weighting is local, per-client, and per-group by construction.

---

## References

- [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach](https://arxiv.org/abs/2302.04228) — Guo, Greengard, Wang, Gelman, Kim, Xing
- [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning](https://arxiv.org/abs/2202.12275) — Ashman, Bui, Nguyen, Markou, Weller, Swaroop, Turner
- [Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/abs/1811.11206) — Bui, Nguyen, Swaroop, Turner
- [Bayesian model reduction](https://arxiv.org/abs/1805.07092) — Friston, Parr, Zeidman
- [distributions.md](distributions.md) — supported exponential families and their log-partition functions
- NIP-01 (events, addressable kinds, filters), NIP-09 (deletion), NIP-11 (relay information), NIP-40 (expiration), NIP-42 (authentication), NIP-45 (COUNT)
- Blossom BUD-01 (blob retrieval by SHA-256), BUD-02 (upload)
