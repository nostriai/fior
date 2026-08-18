# FIOR Architecture

## Brief

**FIOR (Federated Inference Over Relays)** is a Nostr protocol for collaborative machine learning without sharing raw data. Multiple parties train the same model on their own private data and exchange only the statistical parameters describing what each learned. Every party then combines those contributions locally, weighting each peer by how much that peer measurably improves its own inference.

There is no aggregator. Nothing in the protocol requires a participant to wait on, or trust the state of, any other participant.

### Repository Layout

| Part | Language | Location | Status |
|------|----------|----------|--------|
| Protocol spec | Markdown | `protocol.md` | v3 |
| Architecture | Markdown | `architecture.md` | v3 |
| UI integration guide | Markdown | `ui-integration.md` | v3 |
| Distribution reference | Markdown | `distributions.md` | Done |
| Python reference library | Python | `fior-python/` | In progress |
| JavaScript reference library | JavaScript | `fior-js/` | Not started |
| **Composition and trust** | **Python** | **`test/fior_sim.py`** | **v3 reference** |

### How It Works

1. A model creator exports the model definition to ONNX, uploads it to a Blossom server, and publishes a **Model Card** (kind 30100) carrying the blob hash, the map from ONNX initializer tensors to exponential family distributions, and the base prior η₀.

2. A node fetches the model card, the ONNX graph, and η₀.

3. The node fetches the site contributions of whichever peers it judges worth the bandwidth, and **composes its own prior**:

   ```
   η_A^prior = η₀ + Σ_{n ≠ A} p_{A→n} · Δη_n
   ```

   p_{A→n} is A's **inclusion probability** for peer n. On the first round there are no peers and the prior is just η₀.

4. The node trains on its own local data. Raw data never leaves the machine. Training produces a posterior; subtracting the prior it trained against yields the node's **site contribution** Δη.

5. The node uploads Δη as a Blossom blob, then publishes a **Site Contribution** (30101) referencing that blob and pinning the cavity it trained against. Publishing the site *is* joining; there is no registration step.

6. Each user independently scores every peer via **Bayesian model reduction**, a closed-form log Bayes factor. This sets p per peer per parameter group, and the client recomposes its prior.

7. Repeat from step 3.

---

## Reference Implementations

### Python Library (`fior-python/`)

The Python library provides the complete FIOR client API for ML developers and backend services.

**Structure:**
```
fior-python/
├── fior/
│   ├── __init__.py
│   ├── types.py          # Eta, Site, ModelCard
│   ├── math.py           # A(η), composition, BMR
│   ├── trust.py          # p resolution, corroboration, novelty
│   ├── nostr.py          # Event construction, signing
│   ├── blob.py           # Blossom upload/fetch
│   └── client.py         # High-level client API
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

### JavaScript Library (`fior-js/`)

The JavaScript library provides the same API for frontend developers and browser-based clients.

**Structure:**
```
fior-js/
├── src/
│   ├── index.js
│   ├── types.js
│   ├── math.js
│   ├── trust.js
│   ├── nostr.js
│   ├── blob.js
│   └── client.js
├── tests/
├── package.json
└── README.md
```

### Algorithm Reference (`test/fior_sim.py`)

The single source of truth for the algorithm. Implements:
- Natural parameters, A(η), domain checks
- Composition: η₀ + Σ p·Δη
- BMR log Bayes factor
- p resolution: p = σ(ΔF + ln(a/b))
- Loewner confidence bound
- Corroboration (one-peer-one-vote with MAD)
- Novelty weight
- Attack simulations and harm measures

---

## Event Kinds

Three addressable kinds. One per `(kind, pubkey, d)`; latest supersedes.

| Kind | Name | Purpose |
|------|------|---------|
| 30100 | Model Card | ONNX blob, distribution map, η₀ |
| 30101 | Site Contribution | A node's Δη |
| 30102 | Trust Attestation | One scalar p for a peer |

---

## Trust Layer

### Components

1. **BMR scoring** (Mechanism 1): ΔF per peer per group
2. **Loewner cap**: Bound claimed precision against peers
3. **Corroboration**: One-peer-one-vote with MAD scale
4. **Novelty weight**: First-seen tracking for replay defense

### Key Properties

- p is local and per-client
- No consensus score exists
- Trust is subjective
- Every mechanism is optional but recommended

---

## To Be Done

### Python Library

- [ ] Port core math from fior_sim.py
- [ ] Implement TrustTable and Corroboration classes
- [ ] Add Nostr event construction
- [ ] Add Blossom blob handling
- [ ] Write tests matching fior_sim.py test vectors
- [ ] Add examples

### JavaScript Library

- [ ] Port from Python library
- [ ] Use @noble/hashes and @noble/secp256k1
- [ ] Mirror Python tests exactly
- [ ] Cross-language parity tests

### Protocol

- [ ] Submit as NIP to nostr-protocol/nips
- [ ] Verify kind numbers unused

### Production

- [ ] Pyro/NumPyro backend for arbitrary ONNX models
- [ ] GPU-accelerated local training
- [ ] Persistent client daemon
