# FIOR Architecture

## Brief

**FIOR (Federated Inference Over Relays)** is a Nostr protocol for collaborative machine learning without sharing raw data. Multiple parties train the same model on their own private data and exchange only the statistical parameters describing what each learned. Every party then combines those contributions locally, weighting each peer by how much that peer measurably improves its own inference.

There is no aggregator. Nothing in the protocol requires a participant to wait on, or trust the state of, any other participant.

### Repository Layout

| Part | Language | Location | Status |
|------|----------|----------|--------|
| Protocol spec | Markdown | `protocol.md` | v3 |
| Client interface | Markdown | `interface.md` | v3 |
| UI integration guide | Markdown | `ui-integration.md` | v3 |
| Distribution reference | Markdown | `distributions.md` | Done |
| **Composition and trust** | **Python** | **`test/fior_sim.py`** | **v3, working — no Nostr** |
| Python library (wire protocol + client) | Python | `fior-python/fior/` | **v3, working — tested end-to-end against a relay-like stub** |
| CLI | Python | `fior-python/fior/cli/` | **v3, working** |
| Marketplace UI | React | — | Not started |

The wire protocol is implemented in the Python library: event construction and
BIP-340 signing (pure Python, no external crypto dependency), the blob codec
for all families and encodings, discovery, composition, the full trust layer
(BMR unit-weight scoring, the Loewner confidence bound, corroboration, novelty,
attestation priors), and publication.  It is exercised end-to-end by
`fior-python/tests/` against an in-process relay and Blossom server, and the
arithmetic is cross-checked against `test/fior_sim.py`
(`tests/test_sim_consistency.py`).

An earlier Go and Python implementation exists on the `feat/test-model` branch of the
`forum` repository. **It implements v2 and is superseded**: v2 had a central aggregator,
published posteriors rather than site contributions, carried parameters inline as hex, and
had no trust layer at all. Five of its six event kinds have changed meaning. Treat it as
prior art for Nostr plumbing, not as a starting point — see *What is reusable* below.

### How It Works

1. A model creator exports the model definition to ONNX, uploads it to a Blossom server, and publishes a **Model Card** (kind 30100) carrying the blob hash, the map from ONNX initializer tensors to exponential family distributions, and the base prior `η₀`.

2. A node fetches the model card, the ONNX graph, and `η₀`.

3. The node fetches the site contributions of whichever peers it judges worth the bandwidth, and **composes its own prior**:

   ```
   η_A^prior = η₀ + Σ_{n ≠ A} p_{A→n} · Δη_n
   ```

   `p_{A→n}` is A's **inclusion probability** for peer n: `P(z_n = 1 | A's data)`, where
   `z_n` indicates whether n's site belongs in A's pool. It doubles as the composition
   weight because the weight *is* `E[z_n]`. Its prior, from attestations alone, is written
   `β = a/(a+b)` and is `0.5` for a stranger; `protocol.md` keeps the two apart throughout.

   On the first round there are no peers and the prior is just `η₀`.

4. The node trains on its own local data. Raw data never leaves the machine. Training produces a posterior; subtracting the prior it trained against yields the node's **site contribution** `Δη` — its likelihood approximation, and the quantity that composes.

5. The node uploads `Δη` as a Blossom blob, then publishes a **Site Contribution** (30101) referencing that blob and pinning the cavity it trained against. Publishing the site *is* joining; there is no registration step.

6. Each user independently scores every peer via **Bayesian model reduction**, a closed-form log Bayes factor between including that peer **at full weight** and leaving it out entirely, both judged against the same reference holding every other peer at its current `p`. This sets `p` per peer per parameter group, and the client recomposes its prior and consequently its posterior.

7. Repeat from step 3. Steps 3 and 6 happen independently on every client.

The site also carries its **provenance**: one `m` tag per member of the cavity it was fitted against, pinning each by event id with the `p` applied, plus an indexed `p` tag per member pubkey. That makes `Δη = η_post − η_cavity` checkable by anyone, and makes "which sites were built on mine" a relay query.

In parallel and entirely optionally, any node may publish **Trust Attestations** (30102), sharing its inclusion probability `p` for a peer as a single scalar.

### Why It Works

A global aggregation is transformed into local addition. The exponential family is closed under addition of natural parameters, and Nostr's addressable event semantics store exactly one event per `(kind, pubkey, d)`. Keying sites by `(author, model)` therefore gives "the latest site replaces the previous one" for free — precisely the expectation-propagation invariant that makes summing latest-per-author double-count nothing.

That is the whole reason no coordinator is needed: the relay's replacement rule *is* the consistency mechanism.

---

## 1. The reference implementation — `test/fior_sim.py`

This is the only code that implements v3. It is a single self-contained Python file,
runnable with `uv run test/fior_sim.py`, and it is the source of every quantitative claim
in `protocol.md`. Read it before writing the Nostr layer: the wire format is
straightforward, and the arithmetic is where the design actually lives.

**It deliberately has no Nostr, no Blossom, and no ONNX.** Sites are `Eta` objects in a
list, peers are array indices, and a "round" is a loop iteration. The implementation isolates the algorithm's logic for easy testing from a realisation as the nostr protocol.

### What it implements

| Concern | Where | Spec section |
|---|---|---|
| Natural parameters, `A(η)`, domain check | `Eta`, `log_partition`, `in_domain` | *Mathematical Model*, `distributions.md` |
| Composition `η₀ + Σ p·Δη` | `compose_prior` | *Composition* |
| BMR log Bayes factor | `bmr_delta_f` | *Mechanism 1* |
| `p = σ(ΔF + ln(a/b))` | `p_from` | *The model* |
| Attestations → `(a, b)`, with `κ_r` | `resolve_attestation_prior` | *Mechanism 3*, *The reader supplies the weight* |
| Loewner confidence bound | `spectral_clip_site`, `robust_clip_site` | *Bounding claimed confidence* |
| Directional corroboration | `Corroboration` | *Corroboration between sites* |
| Causal span / novelty test | `causal_span_residual`, `novelty_weight` | *Defending against replay* |
| Attacks to test against | `make_attack_sites` | *Relation to the federated learning attack literature* |
| Harm measures | `recovery`, `free_energy` | *What has been tested* |

Its module docstring carries fourteen numbered findings, several of them negative, and
they are the reason the spec says what it says. Findings 2, 12, 13 and 14 each record a
mechanism that looked correct and was not. `test/dfcheck2.py` and `test/meanfield.py`
verify the ΔF definition against brute force; `test/site_disclosure.py` measures what a
published site leaks.

### What it does not implement

The simulation uses one model, one parameter group, one exponential family
(Normal). So the per-group resolution of `p`, the hierarchical attestation prior's pooling
factor `γ`, and every family in `distributions.md` other than Normal are exercised by
the simulation directly but not by its numbers. The Python library implements
all of them (and `tests/test_trust.py` covers the families), but the library's
coverage is newer than the simulation's, so
`protocol.md`'s *What has been tested, and what has not* remains the authoritative
list; do not assume a mechanism is validated because it is specified.

---

## 2. The Nostr layer

This is what the Python library (`fior-python/fior/`) implements. `protocol.md`
is normative for all of it; `interface.md` gives the client-side call surface,
and the library's method names match it.

### Transport and encoding

- **Event construction, signing, relay I/O** for three addressable kinds — 30100, 30101,
  30102. Kind numbers may be reassigned between pre-release revisions (30102 carried
  Trust Evaluation in v2 and carries Trust Attestation in v3); 30103/30105 from an
  intermediate revision are retired. See *Changes from v2*.
  Standard Nostr JSON, BIP-340 Schnorr (pure-Python, `fior/bip340.py`, validated
  against the official test vectors), WebSocket. Addressable semantics are load-bearing,
  not incidental: one event per `(kind, pubkey, d)`, latest supersedes. Keying a local
  store by event id instead of by coordinate will accumulate superseded sites and silently
  double-count members.
- **Blossom upload and fetch**, with SHA-256 verification on fetch. Upload the blob
  *before* publishing the event that references it. An unresolvable blob is a normal state,
  not an error — the member is absent from composition and returns when the bytes resolve.
- **Blob codec** — `f64le`, `f32le`, `i16le`, `i8`, with a header carrying per-block scales
  for the integer forms (one scale per natural-parameter index block: 2 for
  normal/mvnormal/gamma/beta, 1 for dirichlet/cat). Quantizing encodings use stochastic
  rounding; deterministic rounding biases every consumer's composed prior in the same
  direction, so the error accumulates across members instead of cancelling.
- **Validation** on every fetch: decoded vector length against ONNX initializer shapes
  times the family's value count, blob SHA-256, model card `version` on every site before
  composing it, and blob header `group_count` against the model card. Mismatches fail
  loudly (`fior/blob.py` raises) rather than misparsing.

### Arithmetic — ported from `test/fior_sim.py`

- `A(η)` for every family in `distributions.md`, not just Normal, evaluated in f64
  regardless of the encoding a site arrived in. `tests/test_sim_consistency.py` pins the
  library's log-partition, BMR, and full round loop to the simulation's numbers.
- Composition, BMR `ΔF`, and `p` resolution. Four `A()` evaluations per peer per group.
- `(a, b)` resolution from live attestations at the reader's own `κ_r`, plus the
  transitive channel discounted by `p_{A→B}`. The hierarchical pooling across
  `peer` / `peer:model` / `peer:model:group` with `γ` is specified here but its pooling
  factor is untested in the simulation — see section 1.

### Defences — all three, and the ordering matters

- The **Loewner confidence bound** is not optional. Without it a single site can dominate
  every client's prior in the rounds before it is scored. The library implements the
  eigen-clip in the metric of the other peers for `mvnormal` groups and the elementwise
  form for scalar `normal`.
- **Corroboration** as one-peer-one-vote weighted median with MAD scale. Weighting votes by
  declared precision instead is broken and measurably so; the library never does.
- The **novelty weight** uses the client's per-site **first-seen** order (recorded in the
  client when sites are listed; `created_at` is self-asserted and unusable). It scales `p`
  *after* scoring, never the contribution BMR tests, and should be skipped when the member
  count approaches a group's ambient dimension `d + d(d+1)/2`. **It must never ship
  without the confidence bound** — alone it makes fabrication attacks worse; the library
  leaves it off by default.

### Model backend

Bayesian linear regression with closed-form updates is what the simulation uses. Real
models need ONNX import for arbitrary graphs, a Pyro or NumPyro backend, stochastic
variational inference where the posterior is not analytically tractable, and local GPU
training. The library draws the boundary at `initializer_dims` (client-side ONNX shape
extraction) and at `fit_local`/`compute_site` (posterior minus cavity). Neither ONNX
parsing nor a training backend ships in the library — those are integration points, and
the protocol constrains only that a site is `η_post − η_cavity`.

### What is reusable from `forum`

Little, and it should be lifted deliberately rather than migrated. From `fiorgo`: the
six kind constants as integers, `Aggregate` and `Diff` (elementwise add and subtract on
natural parameters, which carry over unchanged), and the orly Nostr plumbing — keypair
generation, WebSocket connection, event signing. From `testmodel`: `generator.py`'s
synthetic data and `model.py`'s mean-field VI fit. Everything else — the hex codec, the
aggregation flow, and the semantics of every kind except 30100 — describes a protocol that
no longer exists.

### Integration targets

- **Marketplace UI.** Wire a client to the v3 kinds per `ui-integration.md`.
- **NIP submission.** Kinds 30100–30105 were verified unused against `nostr-protocol/nips`
  for v2; only three are used now and their semantics have changed entirely, so repeat the
  check. Include the `fior`
  NIP-11 extension object, since that is how a relay declares support.

---

## 3. Open problems in the design

These are unsolved in the *design*, not merely unbuilt. Section 2 is the build list; this
section is what building it will not fix.

**A coalition perturbing within the honest spread.** The principal open problem. A minority
concentrating on the federation's least-observed direction still costs a mean of 4.0 nats
per test point under the confidence bound and corroboration, worst case 28.4 over 8 seeds.
Adding the novelty weight is the best measured configuration at mean 2.7 — tail control,
not elimination. Tightening corroboration further trades the coalition case against lone
attackers and honest participants rather than improving both.

**Transitive trust conflates two quantities.** Attestations from a peer are discounted by `p_{A→B}`, which measures trust in that peer's *parameters* and is used as a proxy for trust in its *judgement about others*. These are not the same quantity. Separating them would require a second Beta per edge.

**Novelty and the confidence bound pull against each other.** Replay itself is solved:
scoring a site by the fraction of it unreachable from **strictly older** sites sends
free-riders to `p = 0.000` and returns the robbed author to 0.987 against an attacker-free
0.983. But novelty rewards exactly what the confidence bound punishes — a falsehood planted
along an unobserved direction is maximally novel by construction, and scores above every
honest peer. Used alone the novelty weight makes fabrication attacks substantially *worse*
(mean excess NLPD +76 against +44 undefended). Only the composition of the two is safe, and
no configuration measured here wins against both a lone attacker and a coalition.

**Disclosure budget.** A site is the sufficient statistic of its publisher's data, so `XᵀX` is exactly public and small-`n` sites approach full disclosure of individual records — as does the difference between consecutive publications by the same author. The spec now states this (*What a published site discloses*) but specifies no mechanism. Adding one means deciding how noise added for privacy interacts with the confidence bounds in the trust layer, which currently cannot distinguish it from honest imprecision.
