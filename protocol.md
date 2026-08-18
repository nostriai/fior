# FIOR Protocol Specification v3

## Overview

Federated Inference Over Relays. Nodes train Bayesian models on local private data and exchange the natural parameters of their posterior distributions over Nostr. Model structure is defined using ONNX. Distribution metadata layers on top of ONNX initializer tensors.

There is no aggregator. Every participant computes its own view of the global posterior locally, from whichever peers it chooses to weight and however it chooses to weight them.

## Glossary

| Term | Plain Language | Mathematical Notation |
|------|----------------|----------------------|
| Site | A node's published contribution - what it learned from its local data | `Δη = η_post - η_prior` |
| Cavity | The prior used for training, excluding the site being evaluated | `η_p^{-n} = η₀ + Σ_{m≠n} p_m·Δη_m` |
| Composition | Building a local prior from trusted peers | `η_prior = η₀ + Σ p_n·Δη_n` |
| p | How much weight to give a peer's site (0 = ignore, 1 = full weight) | `p = σ(ΔF + ln(a/b))` |
| β | Prior belief from attestations alone, before seeing your own data | `β = a/(a+b)` |
| ΔF | Log Bayes factor: does including this peer improve my evidence? | `ΔF = A(η_q^+) + A(η_p^-) - A(η_q^-) - A(η_p^+)` |
| BMR | Bayesian Model Reduction - closed-form scoring without retraining | See above |
| η₀ | Base prior - the starting point before any peer contributions | From model card, or zero vector |
| A(η) | Log-partition function - measures how well parameters fit data | Family-specific, see distributions.md |
| Loewner cap | Bound a peer's claimed precision against what others provide | `Λ_n ⪯ c · Λ_others` |
| Corroboration | Check if a peer's claims agree with trusted peers | One-peer-one-vote with MAD scale |
| Novelty | Weight by how "new" a site is - defends against replay | First-seen tracking |

## How FIOR Works

FIOR enables multiple parties to collaboratively train a model without sharing raw data. Each party trains on their own private data, publishes only the statistical summary of what they learned (a "site contribution"), and locally combines contributions from peers they trust.

**Why it works:** The exponential family of distributions is closed under addition of natural parameters. If you have a prior and add a peer's likelihood contribution, you get a valid posterior. No complex merging algorithm is needed - just addition.

**The trust problem:** How do you know a peer's contribution is good? You don't retrain or hold out data. Instead, you ask: "Does adding this peer at full weight improve my log evidence?" This is Bayesian Model Reduction - a closed-form computation using the log-partition function A(η).

**Defenses against attacks:**
- **Loewner cap:** Bounds how confident a peer can claim to be, relative to what other peers provide
- **Corroboration:** Checks if a peer's claims agree with your trusted peers (one-peer-one-vote)
- **Novelty:** Weights by how recently a site was first seen - defends against replay attacks

## Roles

The protocol recognises exactly two roles.

**FIOR relay.** A NIP-01 relay that accepts the FIOR kind block, advertises support in its NIP-11 document, and enforces size and rate policy. It MAY reject structurally malformed events. It MUST NOT compute, weight, combine, or otherwise alter parameters. A relay is storage and transport.

**FIOR client.** Everything else: fetching model cards and ONNX graphs, running local inference, maintaining a private trust table, building its own prior, publishing contributions, and optionally publishing attestations.

## Event Kinds

| Kind  | Name               | Description |
|-------|--------------------|-------------|
| 30100 | Model Card         | Defines a model, points to ONNX blob, distribution mapping |
| 30101 | Site Contribution  | A node's likelihood approximation Δη |
| 30102 | Trust Attestation  | One scalar p rating a peer's inclusion probability |

All three are addressable (NIP-01, kinds 30000-39999): a relay stores one event per `(kind, pubkey, d)` and the latest supersedes.

---

## 30100 - Model Card

Defines a model. The `d` tag carries the model identifier. Addressable: one per `(pubkey, d)`.

### Tags

| Tag | Required | Description |
|-----|----------|-------------|
| `d` | yes | Model identifier (slug, e.g. `pump-failure-v1`) |
| `t` | yes | Human-readable model name |
| `s` | no | Short description |
| `v` | yes | Model card version; MUST increment on any change to group set, group order, or distribution assignments |
| `o` | yes | ONNX blob reference: `["o", "<sha256-hex>", "<bytes>", "<server-root>", ...]` |
| `D` | yes* | Distribution assignment: `["D", "<init_name>", "<family>"]` |
| `g` | yes* | Named group: `["g", "<name>", "<family>", "<init1>", "<init2>", ...]` |
| `x` | no | Base prior η₀ blob reference |
| `b` | no | Model-wide Blossom server hints |
| `l` | no | Seconds after which a site without its own `E` is considered stale |

`*` At least one `D` or `g` tag.

### `D` Tag Format

Maps an ONNX initializer tensor to a distribution family.

```
["D", "<init_name>", "<family>"]
```

| Field | Type | Description |
|-------|------|-------------|
| `init_name` | string | ONNX initializer tensor name (e.g. `fc1.weight`) |
| `family` | string | `normal`, `gamma`, `beta`, `dirichlet`, `cat` |

### `g` Tag Format

Groups multiple initializers under a shared distribution.

```
["g", "<name>", "<family>", "<init1>", "<init2>", ...]
```

A special group name `*` means "all initializers not otherwise assigned."

**Group order is significant.** Every η blob concatenates its groups in the order their `g`/`D` tags appear in the model card.

### `o` Tag Format - ONNX Blob

```
["o", "<sha256-hex>", "<bytes>", "<server-root>", ...]
```

Fetch from `<server-root>/<sha256-hex>` per Blossom BUD-01. Clients MUST verify the hash on fetch.

### Content

Optional JSON with extended metadata.

### Example

```json
{
  "kind": 30100,
  "tags": [
    ["d", "pump-failure-v1"],
    ["t", "Pump Failure Predictor"],
    ["s", "Predicts centrifugal pump failure within 7 days from sensor readings"],
    ["v", "3"],
    ["o", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "48192", "https://blossom.example"],
    ["g", "fc-layers", "normal", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    ["g", "convs", "normal", "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"],
    ["x", "3a7bd3e2360a3d29eea436fcfb7e44c735d117c42d1c1835420b6b9942dd4f1b", "f64le", "196608", "https://blossom.example"],
    ["b", "https://blossom.example", "https://blossom.mirror.example"],
    ["l", "2592000"]
  ],
  "content": "{\"framework\":\"pyro\",\"onnx_opset\":18,\"min_samples\":500,\"license\":\"MIT\"}"
}
```

---

## 30101 - Site Contribution

A node's likelihood approximation `Δη`. One per `(author, model)`; the latest supersedes. Addressable.

### Tags

| Tag | Required | Description |
|-----|----------|-------------|
| `d` | yes | Model identifier |
| `a` | yes | Model card coordinate: `["a", "30100:<creator-hex>:<model-id>", "", "model"]` |
| `v` | yes | Model card version this site was built against |
| `m` | yes* | One member of the cavity: `["m", "<pubkey-hex>", "<site-event-id>", "<p_g1,p_g2,...>"]` |
| `p` | yes* | Member pubkey (indexed), one per `m` |
| `x` | yes | Δη blob reference |
| `E` | no | NIP-40 expiration; RECOMMENDED |

`*` Required only when the cavity was not η₀ alone. A first-round site carries no `m` or `p` tags.

### `m` Tag Format - Provenance

```
["m", "<pubkey-hex>", "<site-event-id>", "<p_g1,p_g2,...>"]
```

| Field | Type | Description |
|-------|------|-------------|
| `pubkey` | string | Member's public key, hex |
| `site_id` | string | Event id of the exact site version used |
| `p_vector` | string | Comma-separated p per group, in model card group order |

This list IS the prior: `η^prior = η₀ + Σ_m p_m · Δη_m`. Publishing it makes `Δη = η_post - η^prior` auditable.

### `x` Tag Format - Blob Reference

```
["x", "<sha256-hex>", "<enc>", "<bytes>", "<server-root>", ...]
```

| Token | Bytes/value | Notes |
|-------|-------------|-------|
| `f64le` | 8 | Default |
| `f32le` | 4 | Opt-in |
| `i16le` | 2 | Requires scales |
| `i8` | 1 | Requires scales |

Clients MUST verify SHA-256 before use.

### Content

JSON with local training metadata. All fields optional.

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
    ["m", "aaa1...", "e1e1e1...", "0.94,0.31"],
    ["m", "bbb2...", "e2e2e2...", "0.88,0.77"],
    ["p", "aaa1..."],
    ["p", "bbb2..."],
    ["x", "7d3f21a0c48b5e6f9012345678abcdef0123456789abcdef0123456789abcdef", "f64le", "24576", "https://blossom.example"],
    ["E", "1786752000"]
  ],
  "content": "{\"samples\":8234,\"free_energy\":-1247.3}"
}
```

### Withdrawal

A node leaves by publishing a NIP-09 kind 5 deletion for its site, or by letting the site expire.

---

## 30102 - Trust Attestation

An optional, voluntary publication of a client's posterior inclusion probability `p` for a peer. One scalar, no confidence, no member set, no η.

### Tags

| Tag | Required | Description |
|-----|----------|-------------|
| `d` | yes | `<target-hex>`, `<target-hex>:<model-id>`, or `<target-hex>:<model-id>:<group>` |
| `p` | yes | Target pubkey (indexed) |
| `a` | no | Model card coordinate, when model-scoped |
| `i` | yes | The attester's posterior inclusion probability, one float in (0,1) |
| `E` | no | NIP-40 expiration; RECOMMENDED |

### `i` Tag Format

```
["i", "<p>"]
```

A single float in (0,1): the attester's own posterior inclusion probability for the target. A like is `["i", "1"]`, a dislike `["i", "0"]`; readers clamp endpoints into (0,1) before use.

### Content

Optional JSON.

```json
{"basis": "bmr", "rounds": 12}
```

### Example

```json
{
  "kind": 30102,
  "tags": [
    ["d", "bbb2...:pump-failure-v1:fc-layers"],
    ["p", "bbb2..."],
    ["a", "30100:c0ffee...:pump-failure-v1", "", "model"],
    ["i", "0.82"],
    ["E", "1789430400"]
  ],
  "content": "{\"basis\":\"bmr\",\"rounds\":12}"
}
```

---

## Natural Parameter Encoding

**All η payloads are Blossom blobs.** Events carry only SHA-256.

### Blob layout

```
header   u16le  group_count
         per group, in model card order:
           u8     scale_count      (0 for float encodings)
           f64le  scale × scale_count
body     groups concatenated in model card order
```

### Natural parameter layout per family

For a group with `d` scalar parameters:

| Family | Values | Layout |
|--------|--------|--------|
| `normal` | `2d` | first `d`: `η₁ᵢ = μᵢ/σᵢ²`; last `d`: `η₂ᵢ = -1/(2σᵢ²)` |
| `gamma` | `2d` | first `d`: `η₁ᵢ = αᵢ - 1`; last `d`: `η₂ᵢ = -βᵢ` |
| `beta` | `2d` | first `d`: `η₁ᵢ = aᵢ - 1`; last `d`: `η₂ᵢ = bᵢ - 1` |
| `dirichlet` | `k` | `ηᵢ = αᵢ - 1` |
| `cat` | `k` | `ηᵢ = log P(category i)` |

For a site, these are *differences* `Δη`, not posteriors.

---

## Trust Layer

p is computed locally and privately by every client. Publishing an attestation is voluntary.

### The Model

Peer inclusion is a Bernoulli indicator with a Beta prior:

```
z_{n,g} ~ Bernoulli(π_{n,g})     does peer n's group-g site enter my prior?
π_{n,g} ~ Beta(a, b)             prior over that, from attestations
ΔF_{n,g}                          log Bayes factor, from BMR
```

Posterior inclusion probability:

```
p_{n,g} = 1 / (1 + (b/a)·exp(-ΔF_{n,g}))
```

**β and p are different quantities:**
- `β = a/(a+b)` is the *prior* inclusion probability from attestations
- `p = P(z=1 | my data)` is the *posterior* that weights composition

### Mechanism 1 - Bayesian Model Reduction

p answers: "Does including peer n at full weight raise my log evidence?"

Both models are judged against the same reference holding every *other* peer at its current p:

```
η_p^{-n} = η₀ + Σ_{m≠A,m≠n} p_m · Δη_m     (cavity without n)
η_p^{+n} = η_p^{-n} + Δη_n                    (with n at full weight)
η_q^{±n} = η_p^{±n} + L_A                      (plus local data)

ΔF_n = A(η_q^{+n}) + A(η_p^{-n}) - A(η_q^{-n}) - A(η_p^{+n})
```

**Score at unit weight, not at the peer's current p.** As p_n → 0 the contribution vanishes, so ΔF → 0, so p returns to β - a degenerate fixed point.

**The reference is mean-field.** This is the same step expectation propagation takes.

### Bounding Claimed Confidence

A site may claim arbitrary precision. An attacker can read which directions the federation constrains least from public data, then plant a confident falsehood there.

**Loewner cap:** For Normal groups, bound each peer against what other peers provide:

```
Λ_n ⪯ c · Λ_others
```

where `Λ_others` excludes the peer itself. Applied by clipping eigenvalues and rescaling η₁ to preserve the implied mean. `c ≈ 3` is effective against single attackers.

### Corroboration Between Sites

A client's own ΔF has zero power in directions its data doesn't span. But blind spots differ. A site's precision block IS its publisher's observability - the map of who can see what is public.

For each direction, weigh a peer's assertion against trusted peers' assertions:

- One peer, one vote, weighted only by trust already earned (never declared precision)
- Scale discrepancy by the pool's observed dispersion (MAD)

The tolerance is the one free constant and MUST NOT be fixed by fiat.

### Defending Against Replay

Free-riders republish another peer's site verbatim. The claim is true, so ΔF scores it positively.

**Novelty weight:** Track when each site was first seen. Weight p by how "new" the site is relative to its author's previous publication.

**MUST NOT be used alone** - a tailored attack is novel by construction and scores highest.

---

## Query Patterns

| Need | Filter |
|------|--------|
| Browse models | `{kinds:[30100]}` |
| All sites for a model | `{kinds:[30101], "#d":["pump-failure-v1"]}` |
| One peer's current site | `{kinds:[30101], authors:[B], "#d":["pump-failure-v1"]}` |
| Sites built on B's | `{kinds:[30101], "#p":[B]}` |
| Attestations about B | `{kinds:[30102], "#p":[B]}` |

---

## Flow Summary

```
1. Creator exports model to ONNX, uploads to Blossom
2. Creator publishes Model Card (30100) with blob hash, group map, η₀
3. Node fetches model card, ONNX graph, η₀ blob
4. Node fetches Δη blobs of peers it trusts, composes prior: η₀ + Σ p·Δη
5. Node trains locally on private data
6. Node uploads Δη blob, publishes site (30101) referencing blob and pinning cavity
7. Each client recomputes ΔF per peer per group, updates p, recomposes prior
8. Repeat from step 4
```

Optional, in parallel:

```
A. Any node publishes Trust Attestations (30102)
```

## Security

- All events are Nostr-signed by the publishing keypair.
- Clients verify site vector lengths against ONNX initializer shapes from model card.
- Clients verify blob SHA-256 on fetch.
- Clients verify a site's model card version before composing.
- No participant is obliged to accept any other participant's parameters.
- Trust is subjective and per-client. There is no consensus score.
- A site's cavity is fully reproducible from the event ids its `m` tags pin.
