
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

**FIOR relay.** A NIP-01 relay that accepts the FIOR kind block, advertises support in its NIP-11 document, and enforces size and rate policy. It MAY reject structurally malformed events — wrong tag arity, malformed hex, `d` values that do not parse. It MUST NOT compute, weight, combine, or otherwise alter parameters. A relay is storage and transport, and it never holds parameters at all: η payloads live in Blossom blobs and events carry only their hashes.

**FIOR client.** Everything else: fetching model cards and ONNX graphs, running local inference, maintaining a private trust table, building its own prior, publishing contributions, and optionally publishing combinations, benchmarks, and trust attestations.

Aggregating and indexing are things a client may choose to do. No participant holds state that another participant must trust to make progress.

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
Δη_n = η_n^post − η_n^prior
```

where `η_n^prior` is the sufficient statistics of the prior distribution the node trained against. This is the quantity that composes.

### Composition

Every client builds its own prior locally:

```
η_A^prior,g = η₀,g + Σ_{n ∈ M, n ≠ A}  p_{A→n,g} · Δη_{n,g}
```

- `η₀` — base prior, carried in the model card, so every client starts from the same point without a coordinator
- `M` — the members client A chooses to consider
- `p_{A→n,g} ∈ (0,1)` — the precision A assigns to peer n's contribution to group g

Sites are addressable events keyed by `(author, model)`, so a relay stores exactly one per author and the latest supersedes the previous. That is precisely the expectation-propagation invariant "the latest site replaces the previous one," which means summing latest-per-author double-counts nothing and requires no coordination.

Setting `M` to all known peers and every `p` to 1 recovers unweighted summation.

### What p means

For a Normal site, `η₁ = μ/σ²` and `η₂ = −1/(2σ²)`. Scaling both by p leaves the implied mean invariant:

```
−η₁'/(2η₂') = μ           for any p > 0
σ²' = σ²/p
```

So p adjusts **how much you believe a peer, not what you think they said**. p → 0 approaches ignoring them; p = 1 accepts at face value. This is why p multiplies natural parameters rather than posterior means.

### Validity

The composed result must lie in the natural parameter domain of its family:

| Family      | Constraint            |
|-------------|-----------------------|
| Normal      | `η₂ < 0`              |
| Gamma       | `η₁ > −1`, `η₂ < 0`   |
| Dirichlet   | `η_i > −1`            |
| Categorical | unconstrained         |

Sites carry *differences*, so a `Δη` may lower precision and a weighted sum of valid sites need not be valid. A client MUST check the constraint before using a composed group and MUST NOT proceed with one that violates it. **This document specifies no remedy**, because none has been tested; see *What has been tested, and what has not*.

---

## Event Kinds

All three are addressable (NIP-01, kinds 30000–39999): a relay stores one event per `(kind, pubkey, d)` and the latest supersedes.

| Kind  | Name                 | `d`                              | Published by                              |
|-------|----------------------|----------------------------------|-------------------------------------------|
| 30100 | Model Card           | `<model-id>`                     | Creator — ONNX blob, distribution map, η₀ |
| 30101 | Site Contribution    | `<model-id>`                     | Any node — its Δη                         |
| 30102 | Trust Attestation    | `<target-hex>[:<model>[:<group>]]` | Any node — one scalar p for a peer      |

There is no registration kind. Publishing a site **is** joining; a NIP-09 deletion or an expired site **is** leaving. There is no registry, so nothing can disagree about who is a member.

There is no round counter. Ordering is `created_at`, and provenance is the member list each site carries.

Three kinds. Kinds are specified below in the order 30100, 30101, then 30102; the Trust Attestation is documented after the [Trust Layer](#trust-layer), because its `i` tag is meaningless without it. Kind numbers 30102, 30103 and 30105 were used by earlier revisions and are **retired, not reassigned** — see *Changes from v2*.

---

## 30100 — Model Card

Defines a model. The `d` tag carries the model identifier.

### Tags

| Tag       | Required | Description                                              |
|-----------|----------|----------------------------------------------------------|
| `d`       | yes      | Model identifier (slug, e.g. `pump-failure-v1`)          |
| `t`   | yes      | Human-readable model name                                |
| `v` | yes      | Model card version; MUST increment on any change to the group set, group order, or distribution assignments |
| `s` | no       | Short description                                        |
| `o`    | yes      | ONNX blob reference                                      |
| `D`    | yes*     | Distribution assignment for an ONNX initializer tensor   |
| `g`   | yes*     | Named group of initializers sharing a distribution       |
| `x`       | no       | Base prior `η₀` blob reference                           |
| `b` | no       | Model-wide Blossom server hints                          |
| `l`     | no       | Seconds after which a site without its own `E` is considered stale |

`*` At least one `D` or `g` tag.

If no `x` is present, `η₀` is the zero vector. Publishing an explicit proper base prior is RECOMMENDED — a zero vector means `σ² = ∞`, and every node then has to regularise privately in a way no other node can see or reproduce.

### `b` Tag Format

```
["b", "<server-root>", ...]
```

Model-wide fallback servers, tried when a blob's own per-event hints fail. Publishing at least one is RECOMMENDED: per-event hints go stale as servers come and go, and this list is replaceable by the model creator.

### `o` Tag Format

```
["o", "<sha256-hex>", "<bytes>", "<server-root>", ...]
```

The blob is addressed by SHA-256 per Blossom BUD-01: fetch from `<server-root>/<sha256-hex>`. Server roots are hints; any Blossom server holding the hash serves the same bytes. Clients MUST verify the hash on fetch.

### `D` Tag Format

```
["D", "<init_name>", "<family>"]
```

| Field       | Type   | Description                                                     |
|-------------|--------|-----------------------------------------------------------------|
| `init_name` | string | ONNX initializer tensor name (e.g. `fc1.weight`)                |
| `family`    | string | `normal`, `gamma`, `dirichlet`, `cat`, `beta`                   |

### `g` Tag Format

Groups multiple initializer tensors under a shared distribution, reducing wire overhead when parameters share a family and can be concatenated into a single vector.

```
["g", "<name>", "<family>", "<init1>", "<init2>", ...]
```

A special group name `*` means "all initializers not otherwise assigned, with this family."

**Group order is significant.** Every η blob concatenates its groups in the order their `g`/`D` tags appear in the model card. Reordering or adding groups therefore requires a `v` increment, and sites carry the version they were built against.

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

## 30101 — Site Contribution

A node's likelihood approximation `Δη`. One per `(author, model)`; the latest supersedes.

### Tags

| Tag          | Required | Description                                                        |
|--------------|----------|--------------------------------------------------------------------|
| `d`          | yes      | Model identifier                                                   |
| `a`          | yes      | Model card coordinate, marker `model`                              |
| `v`          | yes      | Model card version this site was built against                     |
| `m`          | yes*     | One member of the cavity this site was fitted against               |
| `p`          | yes*     | Member pubkey, one per `m` (indexed, enables reverse lookup)        |
| `x`          | yes      | `Δη` blob reference                                                |
| `E` | no       | NIP-40 expiration; RECOMMENDED                                     |

```
["a", "30100:<creator-hex>:<model-id>", "<relay-hint>", "model"]
```

`*` — required only when the cavity was not `η₀` alone. A first-round site, composed against
the base prior with no peers, carries no `m` or `p` tags, and that absence is the claim that
it used none.

### `m` Tag Format — provenance

```
["m", "<pubkey-hex>", "<site-event-id>", "<p_g1,p_g2,...>"]
```

| Field       | Type   | Description                                                              |
|-------------|--------|--------------------------------------------------------------------------|
| `pubkey`    | string | Member's public key, hex                                                 |
| `site id`   | string | Event id of the **exact site version** used                              |
| `p vector`  | string | Comma-separated `p` per group, in model card group order; a single value means the same `p` for every group |

Members are pinned by **event id**, not by `a` coordinate. Sites are replaceable, so a
coordinate names whatever is current rather than what was actually summed, and provenance
against a moving target is not provenance.

This list **is** the prior: `η^prior = η₀ + Σ_m p_m · Δη_m` over exactly these members at
exactly these weights. That is what makes `Δη = η^post − η^prior` auditable — anyone can
refetch the pinned sites, recompute the sum, and check the subtraction. Publishing it costs
roughly 110 bytes per member on an event every peer already fetches.

Because each member's pubkey is also an indexed `p` tag, `{kinds:[30101], "#p":[B]}` returns
every site that was built on B's — a public derivation graph, obtained for free from tags a
client had to publish anyway.

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
    ["m", "aaa1...", "e1e1e1...", "0.94,0.31"],
    ["m", "bbb2...", "e2e2e2...", "0.88,0.77"],
    ["p", "aaa1..."],
    ["p", "bbb2..."],
    ["x", "7d3f21a0c48b5e6f9012345678abcdef0123456789abcdef0123456789abcdef", "i8", "24576", "https://blossom.example"],
    ["E", "1786752000"]
  ],
  "content": "{\"samples\":8234,\"free_energy\":-1247.3}"
}
```

### Withdrawal

A node leaves by publishing a NIP-09 kind 5 deletion for its site, or by letting the site expire. No other participant needs to act; each client's next composition simply omits it.

---

## Natural Parameter Encoding

**All η payloads are Blossom blobs.** There is no inline form. Events carry a SHA-256 and never the parameters themselves.

This is not only about event size. An inline payload cannot be declined: subscribing to a model would push every member's full parameter vector at you whether or not you intend to use it. With blobs, the event is a cheap descriptor and a client fetches only what its `(a, b)` prior says is worth the bandwidth — which is the selective behaviour the trust layer is built around. It also keeps relays doing what they are good at, small events and broad fanout, rather than re-serving megabytes of float data on every subscription.

The event signature covers the SHA-256, which commits to the bytes exactly as strongly as signing them inline.

### Blob reference

```
["x", "<sha256-hex>", "<enc>", "<bytes>", "<server-root>", ...]
```

Fetch from `<server-root>/<sha256-hex>` per Blossom BUD-01. Server roots are hints, tried in order, then the model card's `b` list. Clients MUST verify the hash before use.

Because blobs are raw binary rather than JSON strings, there is no transport encoding to specify — `<enc>` is just the numeric type:

| Token   | Bytes/value | Notes                      |
|---------|-------------|----------------------------|
| `f64le` | 8           | Default                    |
| `f32le` | 4           | Opt-in                     |
| `i16le` | 2           | Requires scales            |
| `i8`    | 1           | Requires scales            |

Raw binary is half the size of v2's hex encoding and 25% smaller than base64 would be.

### Blob layout

```
header   u16le  group_count
         per group, in model card order:
           u8     scale_count      (0 for float encodings)
           f64le  scale × scale_count
body     groups concatenated in model card order
```

Each group's value count derives from its ONNX initializer shapes times its family's value multiplier, so no lengths are transmitted. `group_count` MUST match the model card at the version the event declares, which makes a stale or reordered model card fail loudly rather than silently misparse.

### Natural parameter layout per family

For a group with `d` scalar parameters:

| Family      | Values | Layout                                                        |
|-------------|--------|---------------------------------------------------------------|
| `normal`    | `2d`   | first `d`: `η₁ᵢ = μᵢ/σᵢ²`; last `d`: `η₂ᵢ = −1/(2σᵢ²)`         |
| `gamma`     | `2d`   | first `d`: `η₁ᵢ = αᵢ − 1`; last `d`: `η₂ᵢ = −pᵢ`               |
| `beta`      | `2d`   | first `d`: `η₁ᵢ = aᵢ − 1`; last `d`: `η₂ᵢ = bᵢ − 1`            |
| `dirichlet` | `k`    | `ηᵢ = αᵢ − 1`                                                  |
| `cat`       | `k`    | `ηᵢ = log P(category i)`                                       |

For a site, these are *differences* `Δη`, not posteriors.

### Quantization

Integer encodings carry one scale per natural-parameter index of the family — two for `normal`, `gamma`, and `beta`; one for `dirichlet` and `cat`. Dequantization is `value = scale × q`.

Per-index scaling is required, not cosmetic: `η₂ = −1/(2σ²)` spans orders of magnitude between a sharply determined parameter and a barely identified one, and a single global scale cannot cover both natural-parameter indices at once.

**Publishers MUST use stochastic rounding when quantizing.** Federated summation is unusually quantization-tolerant — with N members, independent zero-mean errors add in quadrature (~√N) while the signal grows ~N, so relative error in the composed prior falls as 1/√N. That property depends entirely on unbiasedness. Deterministic round-to-nearest is biased, and bias accumulates linearly alongside the signal rather than cancelling.

Two consequences worth stating plainly:

- Quantization noise near `η₂ ≈ 0` can push the composed result out of its domain, which the *Validity* constraint above turns into a refusal to compose. Prefer a wider encoding for groups whose precision sits near zero.
- **The protocol sets no precision floor.** `ΔF` (below) is computed on the quantized `Δη` — the exact bytes a consumer would sum — so an over-quantized site stops helping, its `ΔF` goes negative, and its p falls. Each consumer prices the publisher's bandwidth/accuracy trade-off independently. Precision is a publisher choice, not a conformance requirement.

Wire precision and compute precision are separate concerns: `ΔF` is a difference of differences of `A(η)` and is prone to catastrophic cancellation, so clients MUST dequantize to f64 and evaluate `A()` in f64 regardless of how a site arrived.

### Publishing order and availability

**Publishers MUST upload the blob before publishing the event that references it.** A Nostr event is durable and signed the moment a relay accepts it; its blob is not, so publishing in the other order creates a signed reference to bytes nobody serves.

Publishers SHOULD mirror to more than one server and list each as a hint. Model creators SHOULD maintain the model card's `b` list as a fallback, since per-event hints go stale as servers come and go while the model card is replaceable.

**An unresolvable blob is not an error state.** A client that cannot fetch a member's blob treats that member as absent and composes without them. This needs no special handling: it is the same outcome as declining to fetch a member whose `(a, b)` prior did not justify the bandwidth, and the member returns to the composition as soon as its bytes are retrievable.

Relays store references, not payloads, so a relay's size limits bound tag count and metadata only. Blob size is a matter between publisher and Blossom server.

---

## Trust Layer

p is computed locally and privately by every client. Publishing an attestation is a separate, voluntary act. A node that publishes no attestations and reads none is a fully functional participant — the social layer is a cold-start and bandwidth convenience, never a dependency.

### The model

Peer inclusion is a Bernoulli indicator with a Beta prior:

```
z_{n,g} ~ Bernoulli(π_{n,g})     does peer n's group-g site enter my prior?
π_{n,g} ~ Beta(a, b)             prior over that, from attestations
ΔF_{n,g}                          log Bayes factor, from BMR, recomputed each round
```

Posterior odds are prior odds times the Bayes factor, so:

```
β_{n,g} = E[π_{n,g}]            = a / (a + b)                        prior inclusion probability
p_{n,g} = E[z_{n,g} | my data]  = 1 / (1 + (b/a)·exp(−ΔF_{n,g}))     posterior inclusion probability
```

**`β` and `p` are different quantities and this document keeps them apart.** `β` is the *prior* inclusion probability, fixed by the Beta parameters `(a, b)` and hence by attestations alone; it is what a client believes before looking at its own data. `p` is the *posterior*, `P(z = 1 | my data)`, and it is the number that weights the composition — `η₀ + Σ p_n Δη_n`. Earlier revisions of this document wrote `β` for both, which made statements like "`β = σ(ΔF)`" incoherent: a prior cannot depend on `ΔF`. Where a quantity is a prior, it is `β` or `a/(a+b)`; everywhere else it is `p`.

The two inputs have cleanly separated jobs:

- **`ΔF` is recomputed every round and never stored.** Sites are replaceable, so `Δη_n` at round `t+1` is a different object than at round `t`; evidence about a superseded site is evidence about something that no longer exists. BMR's `ΔF` is exact given the inputs, with no sampling noise to pool away, so nothing justifies an accumulator here. This also removes the earn-then-defect attack: a defecting peer's p collapses the same round, with no banked history to spend.
- **`(a, b)` accumulates across *attesters*, not across time.** It summarises other participants' reports, which a client cannot recompute for itself. Kind 30102 is addressable, so a reader holds at most one live attestation per `(attester, target, scope)`; `(a, b)` is the weighted sum over the attestations currently live, recomputed when any of them is replaced. A reader MUST NOT bank superseded attestations — that would let an attester spend the same opinion twice by republishing it, and would make a retracted judgement permanent. A publisher signs no pseudo-counts at all — it publishes a single p and the reader forms `(a, b)` from it, so `(a, b)` is a reader-side derivation throughout, never a ledger and never a signed quantity.

  It is a prior only. A few nats of direct evidence swamps a handful of pseudo-counts from strangers, which is the correct ordering — and measurably so: at `κ_r = 4`, recovery error runs 0.49, 0.38, 0.34, 0.30, 0.34 as the attesting fraction goes 0, 25, 50, 75, 100 %. It improves sharply at first and then flattens and wanders, because `ΔF` runs to hundreds of nats while an attestation contributes `ln(a/b)`, a couple of nats at most. Attestations are a cold-start convenience, exactly as claimed above, and cannot be made decisive by publishing more of them. (3 seeds; the non-monotone tail is noise, and per *What has been tested* the last two entries should not be ranked against each other. An earlier revision reported 0.46 → 0.25 for this sweep, measured under the superseded publisher-side `(a, b)` scheme.)

With no attestations at all, `(a, b) = (1, 1)` and `p = σ(ΔF)` — pure self-evaluation.

Because the logistic never reaches its asymptotes, `p ∈ (0,1)` strictly, and no peer is ever permanently excluded. Re-testing does not depend on that, however: `ΔF` is evaluated at unit weight (Mechanism 1), so every member is re-tested at *full* weight every round regardless of how low its current p has fallen, and a peer that starts contributing usefully again recovers immediately. Clients MAY hard-threshold `p < ε` to zero to avoid fetching a large blob, accepting that the member goes untested until the threshold is revisited.

Note that this is a statement about a peer's p recovering. It is not a claim that the client's p table as a whole is self-correcting: a peer that has already distorted the prior can leave the client's judgements of *other* peers wrong in a way no later round revisits. See *Bounding claimed confidence*.

### Mechanism 1 — Bayesian model reduction

`p` is the probability that a peer belongs in the pool, so the question to ask about peer n is which of `z_n = 1` and `z_n = 0` the local evidence prefers. Both are judged against the **same** reference, which holds every *other* peer at its current `p` and n at nothing. Write `L_A` for the client's local likelihood contribution, so that each posterior is its prior plus `L_A`:

```
η_p^{−n},g = η₀,g + Σ_{m ∈ M, m ≠ A, m ≠ n}  p_{A→m,g} · Δη_{m,g}
η_p^{+n},g = η_p^{−n},g + Δη_{n,g}
η_q^{±n},g = η_p^{±n},g + L_A,g

ΔF_{n,g}   = A(η_q^{+n},g) + A(η_p^{−n},g) − A(η_q^{−n},g) − A(η_p^{+n},g)
```

`ΔF > 0` means including n **at full weight** raises the log evidence.

The two models differ by exactly `Δη_n`, never by `p_n · Δη_n`. An implementation may reach `η_p^{−n}` by subtracting `p_n·Δη_n` from the working prior it already holds — the same quantity, one rank update instead of a re-sum — but it must then re-add the peer at weight 1, not leave the working prior as the "included" arm. Doing the latter compares `z_n = p_n` against `z_n = 0`, which is a comparison between one real model and one that does not exist.

**Score at unit weight, not at the peer's current p.** Scaling the contribution by `p_{n,g}` — as earlier drafts of this document did — is that mistake, and it is a degenerate fixed point: as `p_n → 0` the contribution vanishes, so `ΔF → 0`, so `p` returns to the prior mean `β = a/(a+b)` no matter what the peer published. A rejected peer drifts back to neutral and cannot be stably excluded. Measured in [test/fior_sim.py](test/fior_sim.py): under the p-scaled form a poisoned peer scores `ΔF = +0.0` and climbs, while at unit weight the same peer scores `ΔF = −3110`.

**The reference is mean-field, and that is an approximation.** `η_p^{−n}` holds every other peer `m` at `p_{A→m} ∈ (0,1)`. No configuration of the indicators does that — each `z_m` is 0 or 1 — so the quantity above is the log Bayes factor evaluated at *one* reference, the mean of the inclusion posterior, rather than its expectation over the `2^{|M|−1}` configurations. `A(η)` is not linear, so the two differ. Exact marginalisation is exponential in the member count and not a practical option; this is the same mean-field step that expectation propagation already takes when it forms a cavity from other sites' approximate factors, and it is recorded here rather than hidden.

Measured by brute-force marginalisation on a 10-node federation (`d = 20`, one seed, so read it as a size not a constant): with every peer at `p = 0.5`, the widest ambiguity available, the mean-field `ΔF` departs from `E_z[ΔF]` by up to 67 nats out of ~100, a relative error up to 0.91. At a converged assignment — in-cluster 0.97, out-of-cluster 0.02 — the gap falls to at most 6.8 nats. **No sign flipped in either case**, which is what matters most, since the sign is what decides whether `p` moves above or below `β`. The error is therefore concentrated in the transient early rounds and shrinks as the assignment sharpens, but it is not negligible while it lasts: a peer scoring `ΔF = 5.2` against an exact `2.3` is the difference between `p = 0.994` and `p = 0.908`.

Unit-weight scoring is also what actually delivers the re-testing property described above. The question asked each round is "would this peer earn full weight, given who else I currently include?" — a question whose answer does not depend on the answer given last round, so a peer that starts contributing usefully again recovers immediately and one that defects collapses immediately.

`A(η)` is the log-partition function, already tabulated for every supported family in [distributions.md](distributions.md), so this introduces no new mathematics. Where a model card's groups are independent blocks, `A` decomposes as a sum over them and `ΔF_{n,g}` follows by restricting to that group's slice — which is what gives trust its per-group resolution. **This decomposition is a property of the group assignment, not of the method.** A model card that assigns coupled parameters to separate groups — a Normal-inverse-gamma over coefficients and noise, whose blocks couple through `−a·ln b`, is the standard case — breaks it. `ΔF_{n,g}` remains well defined there, since the formula above never requires `A` to factor; only the claim that it can be evaluated on that group's slice alone is lost, and the four `A` evaluations must be taken over the full parameter vector with group g's contribution varied.

The cost is **four evaluations of `A(η)` per peer per group**, on parameters the client already holds. All four are peer-specific under unit-weight scoring, where the p-scaled form shared two of them across peers — the correctness fix doubles the cost. It remains far cheaper than the alternative: no holdout split, no retraining, no second pass over data. One local fit per round serves both inference and the evaluation of every peer.

Across rounds this is a fixed-point iteration: `p⁽⁰⁾ = β = a/(a+b)`, then each round's `ΔF` yields the next `p`, which shapes the next round's prior. Clients SHOULD NOT damp the p update; damping leaves residual weight on peers the evidence rejects, and measurably worsens the result.

Scoring a peer against the base prior alone — without the other peers present — is not a usable substitute for the leave-one-out form. Against a vague `η₀`, any added precision raises the evidence regardless of where it points, so unrelated peers and closely matched ones score alike. The discriminating signal comes from the comparison against a prior that already contains everyone else.

### Bounding claimed confidence

BMR asks whether a peer helps. It never asks how sure that peer is entitled to be, and nothing else in this document bounds it either: a site may claim arbitrary precision at no cost, and `Δη` carries no evidence of the data behind it.

This is exploitable, and the exploit does not require ever being trusted. Every site publishes its precision block, which for a Normal group *is* the publisher's observed subspace — so an attacker can read off, from public data, which directions the federation constrains least. A site that agrees with the consensus everywhere else and plants one confident falsehood along such a direction is close to optimal against this scoring rule: along a direction the evaluator has no data about, moving the mean costs the evaluator no evidence while the added precision raises it.

The damage arrives through **transient trust**. `p⁽⁰⁾ = β = a/(a+b)` is `1/2` for a stranger, so before it can be rejected the attacker sits in every client's prior at half weight and, at sufficient claimed precision, dominates it. Every `ΔF` computed in those rounds — including each honest peer's judgement of every *other* honest peer — is measured against a corrupted prior, and because `p = σ(ΔF)` with `ΔF` in the hundreds of nats, those judgements saturate. The attacker is then correctly rejected and the corrupted judgements about everyone else remain. In [test/fior_sim.py](test/fior_sim.py) this costs **a mean of 44 nats per held-out test point over 8 seeds, with a median of 12 and a worst case of 181** — the damage is severe but wildly variable, and any single figure for it is misleading. Meanwhile the attacker ends at `p = 0.012`, trusted by 0.08 of 16 nodes; pinning its p to zero from round 0 reproduces the attacker-free run exactly, which isolates transient trust as the whole of the effect.

Harm is measured here as negative log predictive density on held-out data, not as error in the posterior mean, and the distinction is not cosmetic. This attack manufactures posteriors that are *confidently* wrong: it collapses the variance along a direction while displacing the mean there, so residuals land where the posterior asserts they cannot. An error measure on the mean alone is blind to that by construction, and reports the same attack as a sixfold error inflation while predictive loss goes up by tens of nats.

There is a second, sharper observation. The victim's own free energy is about 1000 nats worse than it should be, so the damage **is** plainly visible in its own evidence — but it is not *attributable*. Per-peer BMR asks whether removing a peer improves matters, and removing the attacker does not restore the p values it corrupted for everyone else. A client can therefore see clearly that its inference has gone wrong and have no way to identify who did it. A trust layer built only on per-peer scoring has no vocabulary for this; detecting it needs an "my evidence is anomalously poor" trigger separate from scoring any individual peer, which this document does not currently specify.

Two natural defences do not work, and are recorded so they are not re-attempted. Setting `p⁽⁰⁾ = 0` only delays exposure by one round, because a stranger scored against `η₀` alone scores positively (see above) and is admitted at high weight immediately after. Capping `trace(Λ)` does not see a rank-one spike: such a site's trace sits well inside any cap loose enough to admit honest nodes.

What works is a bound in the same shape as the attack. Because the falsehood is directional, the constraint must be directional. For a Normal group, writing `Λ = −2η₂`, clients SHOULD bound each peer against what their *other* peers supply:

```
Λ_n  ⪯  c · Λ_others          (Loewner order)
Λ_others = −2 η₂ of  η₀,g + Σ_{m ≠ A, m ≠ n} p_{A→m,g} · Δη_{m,g}
```

applied by clipping the eigenvalues of `Λ_n` in the metric of `Λ_others` and rescaling `η₁` to preserve the peer's implied mean — the peer still says exactly what it thinks, with bounded confidence. **Excluding the peer from its own reference is essential**; a spike that counts toward its own budget bounds nothing. `c ≈ 3` is the most effective single measure in this document against a lone tailored attacker: over 8 seeds it cuts the excess above from a mean of 44 nats per test point to 1.4, and — more tellingly — the worst case from 181 to 3.4. It also restores in-cluster p from 0.803 to 0.977.

This is a client-side policy, like every other part of p. It changes no wire format and requires no agreement between participants.

**It is much weaker against a coalition.** When several peers push the same direction together, the coalition *is* the other peers there and so inflates its own budget. Over 8 seeds the cap reduces a coalition's excess from a mean of 10.4 nats per test point to 6.1, against the 44 → 1.4 it achieves on a lone attacker. The cap is necessary and not sufficient; see *Corroboration between sites*.

For families whose natural parameters carry no matrix precision block, no analogue is specified here.

### The invariant

Every defence that works here obeys one rule, and every design that fails violates it:

> **A peer's influence must be bounded only by quantities it does not control.**

Declared precision is free to fabricate — a site can claim any `Λ` at no cost, backed by no data. So it may never set voting weight, never set a cap's reference, and never scale a discrepancy. The cap above satisfies this because it measures a peer against *other* peers' precision, never its own. Two natural-looking designs that violate it were measured and both are capturable by a coalition: using a per-direction quantile over peers as the cap's reference, and the precision-weighted consensus described next.

**"Does not control" means the claim, not the identity.** Excluding peer n from its own reference set is necessary and is not enough, because a peer may copy its reference rather than influence it: a site that republishes what its reference contains is measured against itself no matter whose pubkey signed it. Every mechanism in this document enforces the identity-level reading only. The consequences are measured under *Replay*, in *Relation to the federated learning attack literature* below, and closing this is unsolved.

### Corroboration between sites

A client's own `ΔF` has zero power in the directions its data does not span, which is exactly where a tailored site puts its falsehood. But that blindness is per-client and blind spots differ — measured, a claim tuned to one group of nodes is markedly more visible to nodes outside it. And nothing needs to be shared to exploit this: a site's precision block already *is* its publisher's observability, so the map of who can see what is public.

Clients therefore SHOULD, for each direction a peer asserts something in, weigh that assertion against what their already-trusted peers assert in the same direction, and scale the peer's precision there by its agreement. This deliberately does not distinguish malice from divergence: a peer from a genuinely different data distribution fails the same test as a fabricator, because it is asserting something the client's trusted peers contradict, and the consequence for the client's inference is the same.

Two details are load-bearing, both instances of the invariant above:

- **One peer, one vote, weighted only by trust already earned** — never by declared precision. Weighting the consensus by claimed precision was measured to invert the mechanism completely: a minority claiming precision two orders of magnitude above everyone else *becomes* the consensus, honest peers are then penalised for disagreeing with it, and the coalition's p rises to 0.88.
- **Scale discrepancy by the pool's observed dispersion**, not its claimed precision. This is also unfakeable, and self-calibrating: where trusted peers genuinely disagree the tolerance widens on its own, and where they agree it tightens.

The tolerance — how far from consensus, in units of the pool's own observed spread, a peer may sit before its confidence there is halved — is the one free constant this introduces, and it MUST NOT be fixed by fiat. The quantity to threshold is a peer's **worst** direction, `max_g |P_{n,g} − median_g| / MAD_g`, not its typical one: a rank-one lie agrees everywhere except where it lies, so any statistic averaged over directions hides it completely.

**Separation is a function of how large the lie is, and small lies are not separated.** Measured over 3 seeds, worst-direction discrepancy by attacker displacement:

| displacement | same-cluster honest | cross-cluster honest | fabricator |
|---|---|---|---|
| 3 | median 10.6, max 29.1 | median 10.4, max 15.5 | median 21.0, max 62.6 |
| 10 | median 8.5, max 20.4 | median 7.5, max 25.6 | median 80.0, max 211.6 |
| 30 | median 6.6, max 17.8 | median 6.4, max 10.4 | median 248.7, max 637.0 |

At a displacement of 30 the two populations are two and a half orders of magnitude apart and any tolerance in the gap works. At a displacement of 3 they **overlap** — the honest maximum, 29.1, exceeds the fabricator median, 21.0 — so no threshold separates them, and a client that picks one anyway will penalise honest peers to catch nothing. An earlier revision of this document reported "honest around 2 with a tail to 17 against a fabricated claim at 180" as though separation were the general case; it is not, and those figures did not reproduce.

This is why the constant must come from the data rather than the specification. Each client can compute its own distribution from public sites alone, at no cost and with no coordination. Clients SHOULD place the tolerance in the gap **when there is one**, and MUST treat an unseparated distribution as evidence the mechanism has nothing to say here rather than as grounds for picking a threshold anyway. The consolation is that the attacks corroboration fails to separate are the small ones, which are also the ones that do least damage.

Note that this discrepancy does **not** distinguish the client's own cluster from another: same-cluster and cross-cluster peers measure alike at every displacement above — 10.6 against 10.4, 8.5 against 7.5, 6.6 against 6.4. That is the division of labour — `ΔF` decides who is useful to you, corroboration decides who is fabricating — and it is why the two are complementary rather than redundant.

Corroboration alone is weaker than the cap against a lone attacker, because honest disagreement widens the tolerance exactly where such an attacker hides. Against a coalition it is the only thing that works. The two compose and SHOULD be used together: the cap bounds how *surely* a peer may assert, corroboration bounds *what* it may assert, and escaping one tightens the other — shedding precision to slip the cap also sheds the weight needed to move a consensus.

Measured as excess negative log predictive density on held-out data, in nats per test point, against the same configuration with no attacker present. **These distributions are heavy-tailed and must not be quoted as point estimates** — over 8 seeds the undefended tailored attack ranges from +0.14 to +181. Mean and worst case over 8 seeds:

| scenario | no defence | cap only | cap + corroboration | + novelty |
|----------|-----------|----------|---------------------|-----------|
| tailored attacker | +44.4 *(max 181)* | **+1.4** *(max 3.4)* | +1.7 *(max 6.9)* | +4.1 *(max 20.4)* |
| colluding coalition | +10.4 *(max 23.1)* | +6.1 *(max 14.5)* | +4.0 *(max 28.4)* | **+2.7** *(max 9.5)* |

An earlier revision of this table reported +97.3 and +1.0 for the first row and +11.1 / +9.0 / +2.1 for the second, from 3 seeds. Those figures were not reproducible: they were single draws from a distribution whose spread is larger than the differences being claimed. Two conclusions drawn from them were wrong and are withdrawn — that the cap achieves only a "19% reduction" against a coalition, and that corroboration is "not a refinement there, it is the whole defence". Over 8 seeds the cap takes a coalition from +10.4 to +6.1 and corroboration takes it to +4.0; both contribute, neither dominates.

Read four things from the corrected table.

*Against a lone tailored attacker, the cap alone is best* and it is best on the worst case as well as the mean — its maximum over 8 seeds is +3.4, where no defence reaches +181. Adding anything else to it makes that row worse, not better.

*Against a coalition, all three together is best*, and the number that matters is the worst case rather than the mean: +9.5 against corroboration's +28.4, a threefold cut in the tail. The mean improvement, +4.0 to +2.7, is well inside the spread and would not be worth claiming on its own.

*There is no single best configuration.* The cap is best on one attack and the full stack on the other, and the gap between them is real in both directions. A client must choose against a threat model rather than enable everything and assume monotone improvement.

*Naive fabricators (3 seeds only, not re-measured): +3.2 / +4.4 / +4.5.* Neither defence helps, because `ΔF` already handled them, and the excess even rises slightly since the defences improve the attacker-free baseline faster than the attacked case. *Absolute NLPD with no attacker at all (3 seeds) is 4.35 / 3.22 / 2.31*, which remains the reason to run the cap and corroboration regardless of threat: they lower predictive loss on an entirely honest network by sharpening how sites from genuinely divergent peers are weighted.

**Corroboration can also rescue a peer that `ΔF` correctly rejected.** Agreement with the trusted pool is exactly what a peer that *republishes* the pool has in abundance. Measured, a free-rider claiming the whole federation's pooled evidence as its own is rejected outright by `ΔF` at p = 0.000, and adding corroboration restores it to 0.167 at a cost of 3.3 nats per test point. Corroboration never adds precision — its weight is at most 1 — so the effect is relative: it shrinks the *disagreeing* honest peers, which changes the cavity every `ΔF` is computed against, and a peer that mirrors the consensus gains from that.

This is the invariant above failing in a form it was not stated strongly enough to catch. Excluding a peer from the reference it is measured against is necessary but not sufficient, because a replayer's content *is* the reference; identity-level exclusion does not make the reference independent of it. Stated correctly, the reference must be independent of the peer's **claim**, not merely of its pubkey — and no mechanism in this document establishes that. See *Relation to the federated learning attack literature*.

**Collusion is narrowed, not closed.** A coalition concentrating on the federation's least-observed direction still costs a mean of 4.0 nats per test point under the cap and corroboration, with a worst case of 28.4 over 8 seeds. Adding the novelty weight of *Defending against replay* is the best measured configuration — mean 2.7, worst case 9.5 — but that is tail control rather than elimination, and it costs accuracy against a lone attacker. Tightening corroboration further trades the coalition case against lone attackers and honest participants rather than improving both. This remains the principal open problem in the trust layer.

### Mechanism 2 — Hierarchy

Attestations address p at three levels — `peer`, `peer:model`, `peer:model:group`. Because the kind is addressable and the `d` values differ, all three coexist rather than replacing each other.

Coarse levels are not a first-hit fallback; they are a **hierarchical prior**, with the parent supplying discounted pseudo-counts to the child. In Beta natural parameters `η^β = (a−1, b−1)`:

```
η^p_eff(B,m,g) = η^p_own(B,m,g) + γ · η^p_eff(B,m)
η^p_eff(B,m)   = η^p_own(B,m)   + γ · η^p_eff(B)
η^p_eff(B)     = η^p_own(B)     + η^p_0
```

`γ ∈ [0,1]` is the pooling factor: `γ = 0` is fully qualified per-group with no cold-start help, `γ = 1` is full pooling. Partial pooling lets a peer with a track record on one model start a new one with a real but discounted prior, while per-group evidence overrides it as it arrives.

This applies only to `(a, b)`. Direct evidence is exact per group and shares no strength across levels.

### Mechanism 3 — Transitivity

Attestations from peers a client already trusts enter as prior pseudo-counts, discounted by that trust:

```
η^β_trans(A→C) = Σ_B  p_{A→B} · κ_r · ( p_{B→C},  1 − p_{B→C} )
```

which adds into the hierarchy above as another prior term. Depth 1 is the default and is cycle-free by construction; multi-hop propagation with product discounting is local policy, not protocol.

No per-contributor clip appears here, and none is needed: an attestation is a single scalar in `(0,1)`, so every contributor supplies at most `κ_r` pseudo-counts by construction. An earlier revision published `(a, b)` and clipped them at `κ_max`; see *An attestation carries a belief, not a weight* for why that was both unfounded and non-monotone.

Discounting by `p_{A→B}` is what makes the channel safe to open. In [test/fior_sim.py](test/fior_sim.py) a colluding minority attests maximally against every honest node in every round; the slander has no measurable effect at any honest-attester fraction, because it is weighted by a trust the attackers never earned. An attester's testimony carrying only the weight it has earned is the property doing the work here.

A known simplification, recorded rather than solved: `p_{A→B}` measures trust in B's *parameters*, and is used here as a proxy for trust in B's *judgment about others*. These are not the same quantity. Separating them would require a second Beta per edge.

### Mechanism 4 — Attestations

Binary likes and dislikes are the endpoints `p = 1` and `p = 0` — the degenerate special case of the same event, requiring no separate kind. A reader that wants a click to count for less than a computed p applies a smaller `κ_r` to it, keyed off the `basis` field in the content; that is reader policy, and the paragraphs below are why.

That is the whole of it. An earlier revision added a second path — a benchmarked leave-one-out pair measured against a shared descriptor, feeding the same `(a, b)` — and it is **removed**; see *Changes from v2* for why.

#### An attestation carries a belief, not a weight

The sentence above — "`a + b` encodes confidence" — described an earlier design in which the publisher sent both numbers. It was wrong, and the current wire format carries **a single scalar p**.

The problem is that on the common path there is nothing to put in `a + b`. `ΔF` is recomputed each round and never accumulated, so a client forming an attestation from its own p holds no round count, no sample count, and no evidence tally from which a confidence could be derived. Any strength it publishes is invention. [test/fior_sim.py](test/fior_sim.py) made this visible by construction: its honest attesters published `(1 + κp, 1 + κ(1−p))` with `κ` a fixed constant, so every attestation carried the same `a + b` and the pair had exactly one degree of freedom — the mean, which is p itself.

Worse, a publisher-supplied strength is a **publisher-supplied weight on the publisher's own testimony**, which is the one quantity a reader must never accept on assertion. Mechanism 3's per-contributor clip `κ_max` existed to claw it back, and clipping an unnormalised pair does not preserve its mean. For one attester at `p = 0.9`, `κ_max = 4`:

| published `κ` | pseudo-counts | after clip | offset (nats) |
|---|---|---|---|
| 2   | (1.80, 0.20)  | (1.80, 0.20) | 0.847 |
| 4   | (3.60, 0.40)  | (3.60, 0.40) | **1.190** |
| 8   | (7.20, 0.80)  | (4.00, 0.80) | 1.022 |
| 32  | (28.80, 3.20) | (4.00, 3.20) | 0.174 |
| 64  | (57.60, 6.40) | (4.00, 4.00) | **0.000** |

Above `κ = κ_max/p` the numerator saturates while the denominator keeps growing, so publishing *more* confidence bought *less* influence; above `κ = κ_max/(1−p)` both sides pinned and the attestation was read as exactly neutral — an attester asserting certainty heard as having no opinion. End to end this showed up as a non-monotone optimum sitting on the hard-coded default: trust in the attackers went 0.213, 0.172, 0.158, 0.126, 0.191, 0.215 for `κ` = 0.5, 1, 2, 8, 32, 128.

#### The reader supplies the weight

A publisher emits its belief `p ∈ (0,1)`. Each reader forms pseudo-counts with **its own** strength `κ_r`:

```
(a, b) = (1, 1) + Σ_B  p_{A→B} · κ_r · (p_{B→C},  1 − p_{B→C})
```

Nothing is lost. `ln(a/b)` is the only quantity the trust layer ever consumes — the Beta variance is never read anywhere — so `a + b`'s sole function was as a per-attestation weight. Under reader-derived `κ_r` the recoverable offset is

```
ln( (1 + κ_r·p) / (1 + κ_r·(1−p)) )  ∈  ( −ln(1+κ_r), +ln(1+κ_r) )
```

monotone in `p`, onto that interval, with `p = 0.5 ↦ 0`. Every offset a reader is *willing to grant* is reachable from `p` alone; the two-parameter form's extra reach was exactly the part `κ_max` existed to remove. The bound is now structural rather than patched, each contributor supplies at most `κ_r` pseudo-counts by construction, and `κ_max` is therefore **removed** — one reader-side knob replaces two that fought each other.

This is the same invariant the rest of this document runs on: the publisher states a claim, the reader prices it. It is the attestation-layer counterpart of *Defending against replay*'s requirement that ordering be unforgeable by the publisher.

Measured, the change is neutral. 5 seeds, 25 rounds:

| configuration | targeted →bad | recov | NLPD | collude →bad | recov | AUC | random →bad |
|---|---|---|---|---|---|---|---|
| publisher `κ=8`, `κ_max=4` (old) | 0.133 | 0.228 | 2.43 | 0.353 | 0.334 | 0.951 | 0.000 |
| reader `κ_r = 1` | 0.176 | 0.245 | 2.47 | 0.353 | 0.351 | 0.947 | 0.000 |
| reader `κ_r = 2` | 0.161 | 0.244 | 2.44 | 0.363 | 0.329 | 0.940 | 0.000 |
| reader `κ_r = 4` | 0.134 | 0.249 | 2.61 | 0.337 | 0.363 | 0.923 | 0.000 |
| reader `κ_r = 16` | 0.181 | 0.233 | 2.30 | 0.364 | 0.353 | **0.847** | 0.000 |
| *no attestations* | 0.551 | 0.380 | 2.71 | 0.709 | 0.687 | 0.596 | 0.000 |

At matched budget (`κ_r` = the old `κ_max` = 4) the two forms are indistinguishable. Every difference down the `κ_r` column is small against the gap to the no-attestation row: the channel matters, its parameterisation barely does, and at 5 seeds with this file's spread the column should be read as "no difference" rather than ranked.

`κ_r` MUST stay small. At 16 a colluding minority's mutual praise is bounded by nothing and ranking AUC falls from 0.951 to 0.847. Clients SHOULD use `κ_r ∈ [1, 4]`. Note also that `κ_r = 16` simultaneously gives the best NLPD and the worst p→attacker on the targeted attack — the same warning as *Defending against replay*, that trust figures and predictive loss can move in opposite directions.

The channel remains second-order by design. Its whole range at `p = 0.9` is 0.32 to 2.20 nats, against a measured `|ΔF|` with median 4.12 and 95th percentile 58.17. Nothing in the attestation layer can make the social channel decisive, which is the intended ordering.

With benchmarks removed, this holds without exception: **no publisher supplies a weight on its own testimony anywhere in the protocol.** The benchmark path was the last mechanism that carried a publisher-asserted evidence count, and removing it makes the rule uniform rather than merely usual.

### Where evidence comes from

```
my own data ──(BMR)──▶ ΔF ──┐
                            ├──▶ p = σ(ΔF + ln(a/b))
attestations ──▶ (a, b) ────┘
```

Two inputs, and only one of them is mine to verify. `ΔF` is recomputed from my own data
every round; `(a, b)` summarises what others report, weighted by trust they have already
earned with me. There is no third channel, and no loop back from any published measurement —
an earlier revision had one, through benchmarked composites, and it is removed.

### Sybil resistance

Direct evidence `ΔF` grows in influence without bound as rounds accumulate, while the transitive channel is discounted by `p_{A→B} ≤ 1`, bounded at `κ_r` per contributor by construction, and open only to peers already trusted. Direct evidence therefore dominates asymptotically.

More fundamentally, minting N identities buys nothing *through the trust channel*: each is scored on its own contribution to the evaluator's free energy, so a Sybil earns p only by actually helping. Earn-then-defect is bounded to a single round by per-round recomputation of `ΔF`.

That argument has one hole, and it is not a small one: a Sybil can help without contributing anything, by republishing a site somebody else published — or any linear combination of several. Each identity then earns p honestly, on the merits, having held no data, and measured it earns *more* than the peer whose work it copied. Nothing specified above closes this. *Defending against replay* below gives a mechanism that does, at the cost of requiring a publication order the publisher cannot backdate.

It does not follow that identities are free of consequence. A fabricated site influences every client's prior during the rounds before it is scored, whatever p it eventually receives, and that influence is bounded only by the precision it claims — so N identities buy N times the transient influence even though they earn no lasting trust. See *Bounding claimed confidence* above: this is the channel that matters, it is not closed by anything in the trust model itself, and against a coordinated coalition it is currently only narrowed rather than closed.

### What has been tested, and what has not

The quantitative claims in this section come from [test/fior_sim.py](test/fior_sim.py): 20 nodes, fully connected, Bayesian linear regression with 100 coefficients and 20 observations per node, so that no node can identify the model alone. Two honest clusters of 8 with different coefficients and noise, plus 4 colluding fabricators. It runs the composition rule and the trust layer directly, with no Nostr, relays, or blobs.

It supports: that p segregates the honest clusters and suppresses fabricators — at the default 6 rounds, in-cluster 0.92, cross-cluster 0.10, fabricator 0.010, ranking AUC 0.96, and recovery error 0.89 → 0.34 against an oracle's 0.10; run to 25 rounds those become 0.94, 0.10, 0.000, 0.95 and 0.89 → 0.25, so the error figure depends strongly on how long the fixed point is allowed to run and should always be quoted with its round count. It also supports the unit-weight correction in Mechanism 1; the leave-one-out requirement; the attestation ceiling and the transitive channel's resistance to collusive slander; the attacks and defences in *Bounding claimed confidence* and *Corroboration between sites*; and the replay results in *Relation to the federated learning attack literature*, including the two negative ones — that replay is undetected by everything else here and misassigns credit, and that corroboration can restore weight `ΔF` withheld. The span test in *Defending against replay* is measured against verbatim copies, blends of 4 and 16 peers, the pooled sum of every honest site, and camouflaged blends, together with its two failure regimes (unordered peers, and member count approaching a group's ambient dimension). It has since been run inside the round loop as a p weight, which is what produced the negative result that it must not be used without a confidence bound. It remains the one mechanism here proposed *after* the simulation rather than derived from it.

Harm figures are quoted as held-out negative log predictive density rather than error in the posterior mean. Where both are available the mean-error figure is consistently the more flattering one, because it cannot see a posterior whose *variance* has been corrupted, which is precisely what a fabricated precision does. Segregation figures (p block means, AUC) are unaffected by this choice — but they are not a substitute for it, and one mechanism here (the novelty weight used alone) improves every p figure while tripling predictive loss.

**A methodological caution that applies to every number in this section.** Predictive loss under a confidence-fabrication attack is heavy-tailed across seeds: the undefended tailored attack spans +0.14 to +181 nats over 8 seeds, so its spread is an order of magnitude larger than most of the differences between defences. An earlier revision of this document quoted 3-seed means as point estimates, and two of the conclusions drawn that way did not survive re-measurement at 8 seeds and have been withdrawn in place. Figures given with a worst case attached are 8-seed; the span-residual and `κ_r` tables are 5-seed; everything else is 3-seed and should be treated as indicative of order of magnitude only. Comparisons between defences whose means differ by less than the spread are not claims this simulation can support.

It does **not** exercise several things this section specifies. Mechanism 2 is untested — the simulation has one model and one group, so the hierarchy is degenerate and `γ` is never exercised. Per-group p resolution is untested, for the same single-group reason, which is also why the group-decomposition caveat in Mechanism 1 rests on analysis rather than measurement.

One further caveat on scope: everything here is measured on a conjugate linear-Gaussian model, where a node's site contribution is exactly its local likelihood and therefore identical whatever prior it composed. Real models are not conjugate, sites move between rounds, and the fixed-point iteration has more to converge than p alone.

### Relation to the federated learning attack literature

The attacks studied in federated learning mostly assume a central server running a fixed aggregation rule over gradients, and a single global model whose accuracy is the thing being protected. Three differences change which of them apply here:

- **There is no aggregator**, so there is no aggregation rule to reverse-engineer and no global model to corrupt. Every client composes privately, and an attack must succeed against each victim separately.
- **What is exchanged is a natural-parameter site, not a gradient.** Confidence is an explicit, forgeable field rather than an implicit property of an update's magnitude. This makes some attacks easier — a precision claim costs nothing to fabricate — and some defences sharper, since precision is also exactly what a directional bound can constrain.
- **Weight is subjective.** The question is not "is this peer honest" but "does this peer raise *my* evidence", which is a different and, as noted below, considerably more tractable question.

With those substitutions, the literature maps onto this design as follows.

| Attack or result | Form it takes here | Status |
|---|---|---|
| No update rule based on a **linear combination** of worker contributions survives one Byzantine worker (Blanchard et al. 2017) | Composition `η₀ + Σ p·Δη` is a linear combination, and at `p⁽⁰⁾ = β = a/(a+b)` a stranger enters at a weight it did not earn | **Applies directly.** This is *Bounding claimed confidence*, arrived at independently. Bounded by the Loewner cap; not eliminated |
| **Robust aggregation** by selection or coordinate-wise robust statistics (Krum; Yin et al. 2018) | No server to run it. The nearest analogue is corroboration's weighted median and MAD, applied per client, per direction | Convergent design, different locus |
| **Optimised local model poisoning** against a known defence (Fang et al. 2020; Shejwalkar & Houmansadr 2021) | The `targeted` attacker: constructed against BMR specifically, using the public precision blocks | Mitigated by the cap. The methodology — attack the defence, not the model — is the one used to derive it |
| **"A little is enough"**: colluders perturbing *within* the honest population's variance evade distance- and median-based defences (Baruch et al. 2019) | A coalition that stays inside the pool's observed dispersion widens the MAD tolerance that is supposed to catch it | **Open.** This is the same phenomenon as the residual coalition cost in *Corroboration between sites*, and the literature indicates it is not a tuning problem |
| **Sybil poisoning detected by update similarity** (FoolsGold; Fung et al. 2020) | N keypairs publishing near-identical sites | Partly, and with an inversion — see below |
| **Backdoor by model replacement**, mitigated by norm clipping (Bagdasaryan et al. 2020; Sun et al. 2019) | A scalar bound on the size of `Δη` | Insufficient here: measured, `trace(Λ)` does not see a rank-one spike. The Loewner cap is the directional generalisation of norm clipping |
| **Robust aggregation fails under non-IID data** (Karimireddy et al. 2022) | Honest cluster divergence is not distinguishable from attack | **Structurally sidestepped** — see below |
| **Trust bootstrapped from a clean root dataset** (FLTrust; Cao et al. 2021) | The client's own local data is its root of trust, and `ΔF` is its trust score | Adopted by construction. The rejected open-benchmark design was a shared root dataset, discarded because the data is private |
| **Free-riding by fabricated updates** (Lin et al. 2019) | Republishing another peer's site, or any linear combination of several, as one's own | **Not caught by anything currently specified**, and `ΔF` rewards it. A span test against strictly older sites does catch it, with two orders of magnitude of margin — see *Replay* and *Defending against replay* below |
| **Cheap identities** (Douceur 2002) | Nostr keypairs are free | Acknowledged, not solved; see *Sybil resistance* |
| **Data reconstruction from shared updates** (Zhu et al. 2019; Geiping et al. 2020) | For a conjugate family, no reconstruction is needed — the site *is* the sufficient statistic. For a deep model it is the same hard inverse problem as in the cited work | **Not addressed by the trust layer at all**; see *What a published site discloses* |

**The similarity inversion.** FoolsGold suppresses clients whose updates are unusually *similar* to each other, reasoning that sybils share a target while honest clients' data does not. Corroboration here does the opposite: it *rewards* agreement with the pool. The two are reconcilable, because corroboration weights the pool only by trust already earned, so agreement among a bloc of strangers buys nothing — the mechanisms disagree about unearned similarity, not about similarity as such. But the reconciliation fails against one adversary in particular: a coalition that earns trust honestly and then turns together. Such a coalition defects while holding `p ≈ 1` rather than the stranger's `1/2`, so its transient influence is larger than a fresh attacker's, and its members corroborate each other at full weight for the round it takes `ΔF` to collapse. Unit-weight scoring bounds this to one round, which is the same bound as everywhere else, but at a higher price per round. A similarity penalty of FoolsGold's kind is the obvious candidate defence and is untested here.

**Why the non-IID impossibility does not bind.** The most relevant negative result in this literature is that with heterogeneous data no aggregator can separate a Byzantine contribution from an honest but unusual one, since both look like outliers, and worst-case convergence to a joint optimum is therefore out of reach. This document does not need that guarantee, because it is not computing a joint optimum. Each client's p answers "useful to me", and a peer whose data comes from a genuinely different distribution is correctly downweighted whether it is malicious or merely different — which is why *Corroboration between sites* declines to distinguish the two cases, and why that is a design decision rather than a limitation. What is given up in exchange is the object the literature is trying to protect: there is no single model here that is correct for everyone, and none is claimed. This is the same trade the *Trust is not consensus* section states from the other direction.

**Replay.** The attack this trust layer handles worst is also the least sophisticated. A peer holding no data at all can copy another peer's published site and publish it as its own. Nothing about the claim is false: it is a real posterior from real observations, correctly formed, pointing where the evaluator's data supports. `ΔF` therefore scores it positively, the Loewner cap admits it because its precision is by construction a genuine peer's, and corroboration endorses it because it agrees exactly with a peer the client already trusts. Every mechanism specified here passes it, and each is *right* to, since each is asking a question the replayer answers honestly.

Measured in [test/fior_sim.py](test/fior_sim.py) with four free-riders copying four of the sixteen honest nodes verbatim, the result is not what a poisoning framing would predict:

| | p to the copier | p to the author it copied | p to uninvolved honest peers | NLPD |
|---|---|---|---|---|
| no free-riders | — | — | 0.983 | 4.354 |
| 4 free-riders | **0.957** | **0.118** | 0.468 | 3.301 |
| 8 free-riders | 0.963 | 0.026 | — | 2.990 |
| 4, with cap + corroboration | 0.963 | 0.110 | 0.502 | 3.358 |

Three things follow, and the first two are the reason this is recorded as an open problem.

*Credit lands on the copier, not the author.* Leave-one-out scoring asks whether removing a peer costs the evaluator evidence. With a duplicate present the answer is no for **either** member of the pair, so the two are interchangeable to the test, and `p = σ(ΔF)` with `ΔF` large is a saturating map with no stable interior point — it resolves the tie rather than splitting it. Which way it resolves is not something this design controls, and in the simulation it consistently resolves against the original: the author of the data falls from 0.983 to 0.118, and to 0.026 once every peer is copied. There is no principle anywhere in this document that assigns credit to the party that actually holds the data, and for a protocol whose purpose is a marketplace in contributions that is a substantive gap, independent of any inference-quality concern.

*The defences are blind, as expected.* The cap and corroboration move the numbers by less than 0.01. Neither is asking a question a replayer fails.

*Predictive accuracy does not suffer at these levels — it improves.* Composition sums `Δη` on the assumption that members' data are disjoint, so a replayed site does enter the same observations twice and the composed posterior is over-precise. But the redundant precision points where the evaluator's own data already supports, so held-out predictive loss falls (4.35 → 3.30 → 2.99) rather than rising. This document earlier asserted the opposite on general grounds; the measurement does not support it. Double counting is a real defect of the composition rule, and the classical treatment of it is the conservative-fusion literature (covariance intersection and successors), but at these levels it is not what makes replay harmful. It should not be assumed harmless at higher duplication, or where the duplicated evidence is *not* aligned with the evaluator's own — neither has been measured.

And note that none of this requires an attacker: two honest clients holding overlapping data produce the same double counting and the same mutual trust collapse.

### Defending against replay

An equality test between different authors' `Δη` is the obvious first move and it is far too weak. A free-rider need not copy anyone verbatim: it can publish **any linear combination** of the sites it has read — an average of its neighbours, an arbitrary weighted blend — and contribute nothing while duplicating nobody. That is a continuum, not a set of cases to enumerate, so the defence has to be stated over the whole span at once.

**The test.** For a candidate site, ask how much of it is *not* reachable as a linear combination of the other published sites. Flatten each site to a vector, solve one least-squares problem, and take the residual norm relative to the site's own norm:

```
r_n = ‖ Δη_n − Σ_{m ∈ R} c*_m Δη_m ‖ / ‖ Δη_n ‖ ,     c* = argmin over all coefficient vectors
```

`r_n = 0` means the peer published nothing the rest of the network had not already published. This searches every coefficient vector simultaneously rather than guessing which subset and which weights, so it subsumes the average-of-neighbours case and every other blend. It costs one least-squares solve per peer over public data, needs no local data at all, and — for *exact* dependence — is metric-free, so it does not inherit the threshold problem that the tolerance in *Corroboration between sites* has.

**Symmetric, it does not work, and the reason matters.** Measured in [test/fior_sim.py](test/fior_sim.py), against sixteen honest nodes at `d = 100`:

| | free-riders | honest peers whose site was blended | honest peers not blended |
|---|---|---|---|
| relative residual | ~4·10⁻¹⁵ | ~2·10⁻¹⁴ | 0.90 – 0.92 |

The free-rider is flagged, and so is every peer it copied from — at the same level. Linear dependence is a property of a *set*, not a direction: if the copier's site lies in the author's span, the author's lies equally in the copier's. This is the same failure that made `ΔF` unable to tell author from copier, reappearing in the detector, and it is fatal on its own — when the free-riders blend widely enough that every honest node is used by someone, all twenty sites score zero and the test says nothing at all.

**A publication order breaks the symmetry, and that is the whole fix.** Project each site only onto the span of *strictly older* sites. Nothing older can explain a genuine contribution; everything a copier published is explained by what preceded it. Measured, with the same worlds:

| free-rider strategy | honest min | honest median | free-rider max |
|---|---|---|---|
| verbatim copy | 0.911 | 0.956 | 4·10⁻¹⁵ |
| blend of 4 peers | 0.911 | 0.956 | 5·10⁻¹⁵ |
| blend of 16 peers | 0.911 | 0.956 | 1·10⁻¹⁵ |
| the pooled sum of every honest site | 0.911 | 0.956 | 3·10⁻¹⁵ |
| blend of 4, plus 1% camouflage noise | 0.911 | 0.956 | 9.9·10⁻³ |

Two orders of magnitude of margin even against a camouflaged blend, and fourteen without. The honest floor does not move, because the ordering only ever removes candidate explanations.

**Camouflage does not pay, and this is what makes the defence hold.** A free-rider escapes only by adding something genuinely new, and what it adds is what it is then scored on. Sweeping the perturbation:

| camouflage noise | residual | p earned |
|---|---|---|
| 0 | 0.000 | 0.20 |
| 0.01 | 0.010 | 0.23 |
| 0.1 | 0.098 | 0.12 |
| 1.0 | 0.680 | 0.34 |
| 3.0 | 0.906 | 0.14 |

Clearing the honest floor takes a perturbation three times the norm of the blend itself, at which point the site is a fabrication that `ΔF` rejects on its own merits (p = 0.14, held-out loss an order of magnitude worse than any honest configuration). The two mechanisms close on the attacker from opposite sides: `ΔF` prices what a site *asserts*, the residual prices what it *adds*, and there is no perturbation that is small enough to keep the first happy and large enough to satisfy the second.

Worth noting for calibration that blending is already the weaker attack — a blend of four peers earns p = 0.20 where a verbatim copy earns 0.957, because leave-one-out finds a blend redundant against the four originals that are all still present. The free-rider's best strategy under `ΔF` is exact copying, which is the easiest thing in this family to detect.

**Where the order comes from.** A self-asserted `created_at` is worthless here: backdating is free, and priority is exactly what a free-rider wants to claim. Two sources are usable, and they serve different purposes.

- **A client's own first-seen order** is unforgeable by the publisher — a peer cannot cause you to have seen its site earlier than you did — costs nothing, and is available immediately. It is subjective, and different clients will order the same two sites differently, but every other quantity in this trust layer is subjective already, so this fits rather than strains the design. **Clients SHOULD use first-seen order for the residual test.** It is unavailable only to a client that joins after both sites already exist.
- **An anchored timestamp** covers that gap and settles credit between clients. [NIP-03](https://github.com/nostr-protocol/nips/blob/master/03.md) defines exactly this primitive — a `kind:1040` event carrying an OpenTimestamps proof for a target event id, anchored in Bitcoin. It proves anteriority only, which is the correct direction: a replayer cannot anchor a site before it has seen it. Two caveats. Bitcoin confirmation takes hours, so anchors cannot gate a per-round p and are a settlement-layer mechanism, not an inference-layer one. And NIP-03 is currently marked *unrecommended — vulnerable to one specific attack, needs update* in the NIPs repository, so this document states the requirement (an anteriority proof the publisher cannot backdate) rather than hard-wiring that NIP.

  Anchoring alone is also not sufficient, because it can be gamed by withholding: anchor early, publish late, and claim priority over someone who published in the interim. If anchors are used, priority SHOULD require the anchor to be recent relative to publication, so that a site cannot be held back and produced later as evidence of precedence.

**Apply it as a weight, not as a substitution.** It is tempting to compose with residuals in place of sites, which would make composition rank-correct and eliminate double counting outright. Do not: the orthogonal projection of a positive semi-definite precision block is not in general positive semi-definite, so the result need not be a valid site. Clients SHOULD instead scale a peer's p by its residual — treating `r_n` as the fraction of that peer's claim that is its own — which changes no wire format and stays inside the existing client-side-policy boundary.

**Where it stops working.** The test has power only while the parameter space is large relative to the number of publishers. Honest sites are linearly independent by accident of having different data, and that accident runs out. Measured on a clean network of sixteen honest nodes with no free-riders at all, the minimum honest residual falls as the ambient dimension of a site — `d + d(d+1)/2` for a Normal group — approaches the number of peers:

| `d` | ambient dimension | min honest residual, causal | same, symmetric |
|---|---|---|---|
| 100 | 5150 | 0.922 | 0.905 |
| 20 | 230 | 0.638 | 0.604 |
| 6 | 27 | 0.247 | 0.179 |
| 3 | 9 | 0.000 | 0.000 |
| 2 | 5 | 0.000 | 0.000 |

Means over 5 seeds. The causal column is the one that governs deployment, since it is the
test clients are told to run; the symmetric column is shown because it is the sharper
statement of the underlying geometry and because an earlier revision of this table quoted
it by mistake, as though it described the deployed test.

Below saturation every honest peer is a linear combination of the others and the test reports everyone as a free-rider. Clients MUST NOT apply it when the number of members approaches a group's ambient dimension. This is a mild constraint for real models, where `d` is large, and a hard one for small groups — which is a reason to prefer the test at model-card scope rather than per small group.

The same effect appears as a **first-mover advantage**: with everything else equal, the sixteenth publisher scores lower than the first, because more explanations exist by the time it arrives. At `d = 100` the decay is mild (1.00 → 0.94 across sixteen publishers), at `d = 20` moderate (1.00 → 0.70), and at `d = 6` severe (1.00 → 0.34). A client that scales p by the raw residual is therefore penalising late joiners for arriving late as well as free-riders for contributing nothing, and these are not the same thing. Normalising against the decay observed among peers already trusted is the obvious correction and is untested.

**Measured inside the round loop.** The figures above are residuals; scaling p by them and running the full fixed point gives, over 8 rounds and 3 seeds, for cluster-A evaluators:

| attack | defence | p → free-rider | p → the author it copied | p → uninvolved peers |
|---|---|---|---|---|
| verbatim replay | none | 0.957 | 0.118 | 0.468 |
| verbatim replay | cap + corroboration | 0.963 | 0.110 | 0.502 |
| verbatim replay | **novelty** | **0.000** | **0.987** | **0.980** |
| blended free-rider | none | 0.200 | 0.885 | 0.870 |
| blended free-rider | **novelty** | **0.000** | 0.977 | 0.980 |

Credit is fully restored: the robbed author returns to 0.987 against an attacker-free value of 0.983, and the free-rider goes to zero. Novelty also repairs the corroboration regression recorded above — the pooled-sum free-rider that corroboration had rehabilitated to p = 0.167 returns to 0.000, and that scenario's excess predictive loss falls from +2.8 to +0.3 nats. On an honest network with no free-riders at all the cost is small and possibly negative: p within-cluster moves 0.983 → 0.976 and ranking AUC is unchanged, while the three mechanisms together give the best absolute predictive loss and the best AUC measured anywhere in this document (2.38 nats, AUC 0.991).

**Novelty MUST NOT be used on its own.** This is the sharpest result in this section and it is a negative one. Used alone it makes *both* confidence-fabrication attacks substantially worse — over 8 seeds the lone tailored attacker goes from a mean excess of +44 nats to **+76**, and a coalition from +10.4 to **+14.6**, with a worst case of +197 and +60 respectively.

The reason is structural rather than incidental, and it is worth stating plainly: **a tailored attack is novel by construction.** It plants its falsehood along the direction the federation observes least, which is precisely the direction no existing site can explain, so the attacker earns the *highest* novelty score in the network — 0.993, above every honest peer. Novelty rewards exactly what the cap exists to punish. The two mechanisms are in tension, not in alignment, and a client running novelty without a confidence bound has built an amplifier.

Note also what this does to the trust metrics: with novelty alone, p → attacker falls to 0.003, the best value in any configuration measured, while predictive loss triples. A mechanism can optimise every trust number in this document and wreck the inference, which is the strongest argument yet for scoring harm by predictive loss rather than by p.

Composed with the cap and corroboration it is instead the best defence measured against a coalition, and mainly through the tail: worst case +9.5 against corroboration's +28.4. On a lone attacker the composition is worse than the cap by itself (+4.1 against +1.4). **Clients SHOULD enable novelty only alongside a confidence bound, and SHOULD NOT assume that enabling every mechanism is monotonically safer.**

**What this does not solve.** It bounds influence and, given an order, assigns credit — but only against free-riders who *republish*. A peer that holds real data and simply also happens to duplicate a neighbour's is penalised identically, which for double counting is correct and for credit is not. And nothing here addresses a coalition contributing genuine but coordinated evidence, which is the separate open problem above.

**Replaying the pooled sum is caught, and the defences partly undo the catch.** A free-rider that republishes the *pooled* sum of every honest site — claiming the whole federation's evidence — is rejected outright by `ΔF` alone, at p = 0.000, because the pool carries the other cluster's evidence as well and that half lowers the evaluator's log evidence. Over-reaching is what exposes it. But adding the cap and corroboration **rehabilitates it to p = 0.167 and raises predictive loss from 2.08 to 5.36 nats per point.** The reason is structural rather than incidental: corroboration scales a peer's precision by its agreement with the trusted pool, and a republisher of the pool is by construction in maximal agreement with it. Corroboration can therefore restore weight to a peer that `ΔF` correctly rejected. This is a cost of the mechanism recommended in *Corroboration between sites*, it was not anticipated there, and it is the one measured case in this document where a defence makes an attack materially worse.

### Trust is not consensus

There is no global trust score and no snapshot event. Each client's p table is its own, derived from its own data. Two honest clients with different data will legitimately assign different p to the same peer, and neither is wrong. Nothing in this protocol publishes a score, so there is nothing for a marketplace to rank by — what it can show is a viewer's own p, clearly labelled as local, and the derivation graph from site provenance.

---

## 30102 — Trust Attestation

An optional, voluntary publication of a client's posterior inclusion probability `p` for a peer. Carries a *belief about a peer* and nothing else: one scalar, no confidence, no member set, no η, no version pinning.

### Tags

| Tag          | Required | Description                                                    |
|--------------|----------|-----------------------------------------------------------------|
| `d`          | yes      | `<target-hex>`, `<target-hex>:<model-id>`, or `<target-hex>:<model-id>:<group>` |
| `p`          | yes      | Target pubkey (indexed)                                        |
| `a`          | no       | Model card coordinate, marker `model`, when model-scoped       |
| `i`       | yes      | The attester's posterior inclusion probability, one float in `(0,1)` |
| `E` | no       | NIP-40 expiration; RECOMMENDED                                 |

### `i` Tag Format

```
["i", "<p>"]
```

A single float in the open interval `(0,1)`: the attester's own **posterior** inclusion probability `p` for the target — `P(z = 1 | the attester's data)` — and nothing about how strongly anyone else should hold it. A like is `["i", "1"]`, a dislike `["i", "0"]`; readers clamp the endpoints into `(0,1)` before use.

Note what the reader does with it: one client's *posterior* becomes an observation feeding another client's *prior* `(a, b)`, hence its `β`. The two clients' quantities are not the same object, which is why they do not share a name.

Readers form Beta pseudo-counts locally, with their own strength `κ_r`, as specified in *The reader supplies the weight*. A publisher does not choose the weight its testimony carries, and there is consequently nothing to clip.

**Renamed from `beta`.** Earlier revisions called this tag `beta` and, before that, gave it two values. Both were wrong: the value is a posterior, so naming it for the Beta *prior* inverted the model, and `beta` already names an exponential-family distribution in model card `D` tags. Readers SHOULD accept a legacy `beta` tag, taking a single value as `p` and a pair as `a/(a+b)`.

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

The attester weights this peer's `fc-layers` contribution at 0.82. A reader using `κ_r = 4` and holding `p = 0.6` for the attester adds `0.6·4·(0.82, 0.18) = (1.97, 0.43)` to its pseudo-counts for that target — an offset of about 0.7 nats, against a `ΔF` that is routinely tens.

---

## Relay Conformance

A FIOR relay advertises support in its NIP-11 document. This is the entire mechanism by which a relay declares it supports the marketplace — no new role or handshake is introduced.

```json
{
  "supported_nips": [1, 9, 11, 40, 42, 45],
  "limitation": { "max_message_length": 262144, "max_event_tags": 2000 },
  "fior": {
    "version": "3",
    "kinds": [30100, 30101, 30102],
    "models": ["pump-failure-v1"],
    "validates": ["structure"],
    "blossom": ["https://blossom.example"]
  }
}
```

`models` omitted means any model is accepted; present means an allowlist.

### Validation split

- **Clients MUST** verify decoded vector lengths against group dimensions derived from ONNX initializer shapes times the family's value count; verify blob SHA-256 on fetch; verify the model card `v` of every site they compose.
- **Relays MAY** perform cheap structural checks only: tag arity, SHA-256 hex well-formedness, `d` well-formedness, known `enc` token. A relay cannot be expected to parse ONNX or fetch blobs, and per the roles above it must never touch the numbers.

---

## Query Patterns

| Need                          | Filter                                                       |
|-------------------------------|--------------------------------------------------------------|
| Browse models                 | `{kinds:[30100]}`                                            |
| All sites for a model         | `{kinds:[30101], "#d":["pump-failure-v1"]}`                  |
| One peer's current site       | `{kinds:[30101], authors:[B], "#d":["pump-failure-v1"]}`     |
| Sites built on B's            | `{kinds:[30101], "#p":[B]}`                                  |
| Attestations about B          | `{kinds:[30102], "#p":[B]}`                                  |

Only single-letter tags are indexed under NIP-01, which is why event and coordinate references use `e`, `a`, and `p` rather than descriptive names.

The `#p` query is the derivation graph: every site tags each member of its cavity, so asking for sites that tag B returns everyone who built on B's work. NIP-01 filters OR within a key and cannot AND across keys, so anything narrower — sites built on both B *and* C — is a fetch-by-one, filter-client-side operation.

---

## Flow Summary

```
1. Creator exports the model to ONNX, uploads it to a Blossom server
2. Creator publishes a Model Card (30100) with the blob hash, group map, and η₀
3. A node fetches the model card, the ONNX graph, and the η₀ blob
4. The node fetches the Δη blobs of whichever peers it judges worth the
   bandwidth, and composes its prior: η₀ + Σ p Δη
   (first round: no peers, so the prior is η₀)
5. The node trains locally on private data
6. The node uploads its Δη blob, then publishes its site (30101) referencing
   that blob and pinning the cavity it trained against
7. Each client independently recomputes ΔF per peer per group via BMR,
   updates p, and recomposes its prior
8. Repeat from step 4
```

Steps 4 and 7 happen independently on every client. Nothing in this loop requires any participant to wait on any other.

Optional, in parallel:

```
A. Any node publishes Trust Attestations (30102)
```

---

## Security

- All events are Nostr-signed by the publishing keypair.
- Clients verify site vector lengths against ONNX initializer shapes from the model card, and reject on mismatch.
- Clients verify blob integrity by SHA-256 before use. A Blossom server is therefore untrusted infrastructure: it can withhold bytes but cannot substitute them.
- Blob availability is a liveness concern, not a safety one. A withheld or lost blob makes a member absent from a composition — the same outcome as a low p — and never produces a wrong result.
- Clients verify a site's model card `v` before composing it.
- No participant is obliged to accept any other participant's parameters. p is local, and `p → 0` costs nothing to apply.
- Poisoned parameters are suppressed by the same mechanism that weights honest ones: a site that lowers the evaluator's log evidence gets `ΔF < 0` and a `p` below the prior mean `β`, in the round it is published. Read this narrowly. It is a statement about sites that lower the evaluator's evidence, and the attacks that matter are the ones that do not — a fabricated claim in a direction the evaluator has no data about, and a replayed site whose content is true. Both are covered in the trust layer, and neither is fully closed.
- Trust is subjective and per-client. There is no consensus score to capture.
- A site's cavity is fully reproducible from the event ids its `m` tags pin, so a claimed `Δη` is checkable by anyone who refetches its members.

### What a published site discloses

This protocol is designed so that data never leaves the client, and the trust layer above is built on the premise that a peer's data is private. **That premise is about custody, not confidentiality, and the difference matters.**

A large body of work reconstructs training data from shared gradients ([Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935); [Inverting Gradients](https://arxiv.org/abs/2003.14053)), by solving for the input that would have produced the observed update. **How much of that applies here depends entirely on the model, and the two ends of the range are very far apart.**

**Conjugate models: exact, and no attack is required.** Where the likelihood is conjugate to the family — linear-Gaussian regression being the case this document's simulation uses — a site contribution `Δη` *is* the sufficient statistic of the publisher's data. That is precisely what makes the composition rule a sum. The statistics are therefore published directly rather than inferred, and everything below follows by algebra rather than by optimisation.

For a Normal group with known noise precision `τ`, the site is exactly `Λ = τ·XᵀX` and `η₁ = τ·Xᵀy`. Consequently:

- **`XᵀX` is recovered exactly** by any reader who knows `τ`, which the model card publishes. Every feature variance and every pairwise feature correlation in the publisher's sample is public.
- **The observations themselves are determined up to an `n × n` orthogonal mixing of the publisher's own rows**, where `n` is the number of local observations. That mixing is not published, so for large `n` individual records are not identified.
- **For small `n` this protection disappears.** At `n = 1` the orthogonal group is `{+1, −1}`, so a single-observation site discloses that observation exactly, up to a global sign. Verified in [test/site_disclosure.py](test/site_disclosure.py).
- **Differences between consecutive sites are themselves sites.** Kind 30101 is addressable, so a reader keeps the replacement and can subtract it from the version it replaced. If a publisher adds one observation between publications, the difference is a one-observation site — the fully-disclosing case above. A client that republishes frequently against a slowly growing dataset leaks its data record by record.

**Non-conjugate and deep models: none of the above transfers.** For the models this protocol actually targets, a site is a variational natural-parameter update from an approximate posterior over a nonlinear map — not a sufficient statistic, and generally not an invertible function of the data at all. Recovering inputs there is the hard, empirical inverse problem the gradient-inversion literature studies, and its results are strongly architecture-, batch-size- and initialisation-dependent rather than exact. Nothing above should be read as a claim about those models. What survives the generalisation is only the weak and unsurprising statement that a published update carries *some* information about the data behind it, and that the amount is a property of the model, not of this protocol.

The practical consequence is that the disclosure budget is a per-model-card question. A card whose family and likelihood are conjugate is publishing sufficient statistics and should be treated accordingly; a card wrapping a deep network is not, and the honest position is that its exposure is unquantified rather than either exact or negligible.

Clients that consider their raw observations sensitive SHOULD therefore accumulate a substantial number of observations before publishing a site, SHOULD NOT republish on every new datapoint, and MAY add calibrated noise to `Δη` before publishing — at a cost in the `ΔF` others compute for them, since a noisier site is a genuinely less useful one. This document specifies no differential-privacy mechanism, and the interaction between such a mechanism and the confidence bounds in the trust layer is unexamined: noise added for privacy is, to a peer applying the Loewner cap, indistinguishable from honest imprecision.

---

## Changes from v2

### Structural

| v2                                     | v3                                                      |
|----------------------------------------|----------------------------------------------------------|
| Aggregator computes and broadcasts the global prior | Every client composes its own prior locally |
| 30101 Prior Broadcast (aggregator)     | removed — there is no shared prior to broadcast          |
| 30102 Posterior Submit (full posterior)| 30101 Site Contribution (`Δη`), carrying its own provenance |
| 30103 Node Registration                | removed — publishing a site is joining                   |
| 30102 Trust Evaluation (back-test)     | removed — see *Benchmarks are removed* below             |
| 30105 Trust Snapshot (aggregator)      | removed — no consensus score exists                      |
| —                                      | 30102 Trust Attestation (one scalar p)                   |
| `round` counter                        | removed — no global rounds                               |
| `agg` tag                              | removed                                                  |

Three kinds remain: 30100, 30101, 30102.

### Benchmarks are removed

An intermediate revision of v3 specified three further kinds — **30102 Composite**, a
reproducible weighted combination; **30103 Benchmark Result**, a measurement of a site or
composite; and **30105 Benchmark Descriptor**, defining what made results comparable. All
three are gone. **The numbers are retired, not reassigned**, so a client encountering them
on a relay is seeing an obsolete revision, never a different meaning.

The reasoning is that a benchmark cannot be both verifiable and informative here:

- **If the evaluation set is public**, any node can run it locally against parameters it
  already fetched. Publishing the result adds no information, and inviting peers to trust a
  number they could have computed is a net loss — it substitutes an assertion for a
  computation.
- **If the evaluation set is private**, the result is unverifiable by construction. The
  earlier text conceded as much: results from a private descriptor were comparable "only
  within one publisher's series" and had to be attributed as "measured by npub…". That is a
  reputation claim wearing the costume of a measurement.

Since data is private and unshareable — the premise the whole protocol rests on — every
benchmark falls into one case or the other. What replaces it is not a better measurement
but a different kind of evidence: **distributed agreement**. Corroboration and the trust
layer ask whether peers *independently* concur, which is unfakeable in a way a published
score is not. See *Corroboration between sites*.

There was also an adversarial cost. A published benchmark is a public objective, and a
public objective is a gradient an attacker can follow: it tells a poisoner exactly how
close it is to passing, turning a blind attack into a guided one. The federated-learning
literature calls this the optimised-local-model-poisoning setting, and it is strictly
harder to defend than the blind case.

Two consequences elsewhere in this document. Mechanism 4 loses its second path, so `(a, b)`
now has exactly one source — attestations — and the rule that *no publisher supplies a
weight on its own testimony* holds without exception. And the cavity, which 30102 used to
pin by reference, is now carried inline on the site itself as `m` tags, which is strictly
more auditable: provenance travels with the thing it describes and cannot go missing.

### Nostr conformance fixes

| v2                              | v3                       | Reason                                                                 |
|---------------------------------|--------------------------|------------------------------------------------------------------------|
| 30102 treated as an accumulating list | one site per (author, model), latest wins | 30000–39999 is addressable; a conformant relay drops the others |
| 30105 declared non-replaceable  | n/a                      | Same — the kind range makes that impossible                            |
| `["post", "<event-id>"]` provenance | `["m", ...]` + indexed `["p", ...]` on the site | Provenance belongs on the event it describes, and `#p` makes the derivation graph queryable |
| `["η", ...]`                    | removed                  | Payload moved to blobs; the non-ASCII tag key was also a cross-implementation hazard and was never indexable |
| `["p", "<event-id>"]`           | `["a", ...]` / `["e", ...]` | `p` is a pubkey reference in NIP-01 and relays index it as one       |
| `["post", "<event-id>"]`        | `["e", ...]` + `["a", ...]` | Multi-letter tags are not indexed, so evaluations were unqueryable  |
| `["agg", "npub1..."]`           | removed                  | Tags carry 32-byte hex; npub is NIP-19 display encoding                |
| `["o", "<nostr event id>"]`  | `["o", "<sha256>", ...]` | Blossom addresses blobs by SHA-256, not event id                    |
| "NIP-33"                        | NIP-01                   | NIP-33 was merged into NIP-01                                          |
| `η` hex-encoded in event tags   | `η` always a Blossom blob; events carry only the SHA-256 | Typical relay caps are 64–256 KiB, which a real model exceeds immediately; and an inline payload cannot be declined by a subscriber |
| hex-only encoding               | raw binary, `f32le`/`i16le`/`i8` available | Half the size of hex at no loss; up to 8× with quantization |

### Trust layer

v2 scored contributions by back-testing: split local data, retrain against the peer's posterior, evaluate both on a holdout, publish per-group deltas. v3 replaces it with Bayesian model reduction — a closed form in `A(η)`, computable from parameters already held, with no holdout split and no retraining. Back-testing does not survive in any form; see *Benchmarks are removed*.

v2 averaged trust scores across evaluators using different private holdouts, which are not on a common scale. v3 publishes no scores to average.

v2 planned consensus-weighted aggregation at the aggregator. v3 has no aggregator and no consensus: weighting is local, per-client, and per-group by construction.

---

## References

### Federated inference

- [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach](https://arxiv.org/abs/2302.04228) — Guo, Greengard, Wang, Gelman, Kim, Xing
- [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning](https://arxiv.org/abs/2202.12275) — Ashman, Bui, Nguyen, Markou, Weller, Swaroop, Turner
- [Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/abs/1811.11206) — Bui, Nguyen, Swaroop, Turner
- [Bayesian model reduction](https://arxiv.org/abs/1805.07092) — Friston, Parr, Zeidman
- [FedGVI: Federated Generalised Variational Inference](https://arxiv.org/abs/2502.00846) — Mildner, Hamelijnck, Giampouras, Damoulas (ICML 2025). Robustifies PVI by replacing the KL objective with a robust divergence, which is the alternative to the client-side bounds specified here: it prices misspecified sites in the inference objective rather than in a separate trust layer.
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) — Kairouz et al. §5 surveys adversarial attacks on model performance.
- [distributions.md](distributions.md) — supported exponential families and their log-partition functions

### Attacks and defences, and where this document stands relative to them

Discussed in *Relation to the federated learning attack literature* above.

- [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://arxiv.org/abs/1703.02757) — Blanchard, El Mhamdi, Guerraoui, Stainer (NIPS 2017). No update rule based on a linear combination of worker contributions tolerates a single Byzantine worker; introduces Krum.
- [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/abs/1803.01498) — Yin, Chen, Ramchandran, Bartlett (ICML 2018). Coordinate-wise median and trimmed mean.
- [A Little Is Enough: Circumventing Defenses For Distributed Learning](https://arxiv.org/abs/1902.06156) — Baruch, Baruch, Goldberg (NeurIPS 2019). Colluding workers perturbing within the honest population's variance evade distance- and median-based defences.
- [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/abs/1911.11815) — Fang, Cao, Jia, Gong (USENIX Security 2020). Attacks optimised against a known defence rather than against the model.
- [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) — Shejwalkar, Houmansadr (NDSS 2021).
- [Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning](https://arxiv.org/abs/2108.10241) — Shejwalkar, Houmansadr, Kairouz, Ramage (IEEE S&P 2022). Argues that under production threat models, simple defences suffice and poisoning is over-stated.
- [Mitigating Sybils in Federated Learning Poisoning](https://arxiv.org/abs/1808.04866) / [The Limitations of Federated Learning in Sybil Settings](https://www.usenix.org/conference/raid2020/presentation/fung) — Fung, Yoon, Beschastnikh (RAID 2020). FoolsGold penalises *similarity* between client updates.
- [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459) — Bagdasaryan, Veit, Hua, Estrin, Shmatikov (AISTATS 2020). Model-replacement backdoors.
- [Can You Really Backdoor Federated Learning?](https://arxiv.org/abs/1911.07963) — Sun, Kairouz, Suresh, McMahan. Norm clipping and weak DP as defences — the scalar ancestor of the directional bound in *Bounding claimed confidence*.
- [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://arxiv.org/abs/2006.09365) — Karimireddy, He, Jaggi (ICLR 2022). Under non-IID data, honest heterogeneity and attack are not separable by a robust aggregator.
- [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/) — Cao, Fang, Liu, Gong (NDSS 2021). Trust must be anchored in data the evaluator holds.
- [Free-riders in Federated Learning: Attacks and Defenses](https://arxiv.org/abs/1911.12560) — Lin, Du, Liu. Fabricated updates that earn reward without contributing data.
- [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) — Zhu, Liu, Han (NeurIPS 2019); [Inverting Gradients — How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) — Geiping, Bauermeister, Dröge, Moeller (NeurIPS 2020). Training data reconstructed from shared updates.
- [Poisoning Bayesian Inference via Data Deletion and Replication](https://arxiv.org/abs/2503.04480) — Carreau, Naveiro, Caballero (2025). Steering a Bayesian posterior toward a target while leaving other inferences undisturbed.
- [Data Poisoning Attacks in Gossip Learning](https://arxiv.org/abs/2403.06583) — Pham, Potop-Butucaru, Tixeuil, Fdida; [RepuNet: A Reputation System for Mitigating Malicious Clients in DFL](https://www.sciencedirect.com/science/article/pii/S1389128626002549). Serverless and reputation-based settings, the closest system-level relatives to this design.
- [The Sybil Attack](https://link.springer.com/chapter/10.1007/3-540-45748-8_24) — Douceur (IPTPS 2002). Without a central authority, cheap identities are always possible.
- [A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations](https://ieeexplore.ieee.org/document/609105) — Julier, Uhlmann (ACC 1997). Covariance intersection: conservative fusion when the correlation between two estimates is unknown, the classical treatment of the double-counting problem in *Replay*.
- NIP-01 (events, addressable kinds, filters), NIP-09 (deletion), NIP-11 (relay information), NIP-40 (expiration), NIP-42 (authentication), NIP-45 (COUNT)
- Blossom BUD-01 (blob retrieval by SHA-256), BUD-02 (upload)
