# FIOR Client Interface

## Overview

The FIOR client API provides methods for participating in federated inference. All operations are local - no aggregator or coordinator is required.

## Methods

### fetch_model_card

Fetch a model card from the relay.

**Input:**
- `model_id` - Model identifier (the `d` tag value)
- `relay_url` - Optional relay URL (uses default if omitted)

**Output:**
- `ModelCard` object with groups, distributions, η₀, ONNX blob reference

**Behavior:** Queries kind-30100 events, returns the latest for the given model_id.

---

### fetch_sites

Fetch site contributions for a model.

**Input:**
- `model_id` - Model identifier
- `relay_url` - Optional relay URL

**Output:**
- Array of `Site` objects, each with author, delta_eta, cavity_members

**Behavior:** Queries kind-30101 events filtered by `d` tag.

---

### fetch_attestations

Fetch trust attestations for a peer.

**Input:**
- `target_pubkey` - Target peer's public key (hex)
- `relay_url` - Optional relay URL

**Output:**
- Array of `Attestation` objects, each with scope and inclusion probability

**Behavior:** Queries kind-30102 events filtered by `p` tag.

---

### compose_prior

Build a local prior from trusted peers.

**Input:**
- `eta0` - Base prior (from model card)
- `sites` - Array of Site objects to include
- `p_values` - Array of inclusion probabilities (same order as sites)

**Output:**
- `Eta` object (natural parameters of the composed prior)

**Behavior:** Computes `η₀ + Σ p_n · Δη_n`. This is the prior used for training.

---

### compute_delta_f

Compute log Bayes factor for a peer using BMR.

**Input:**
- `site` - Site to evaluate
- `cavity` - Prior without this site
- `local_likelihood` - Client's local data contribution

**Output:**
- `float` - ΔF (log Bayes factor)

**Behavior:** Evaluates whether including this peer at full weight improves log evidence. Four A(η) evaluations.

---

### resolve_p

Convert ΔF and prior belief to inclusion probability.

**Input:**
- `delta_f` - Log Bayes factor from BMR
- `beta` - Prior inclusion probability from attestations (a/(a+b))

**Output:**
- `float` - p in (0,1)

**Behavior:** Computes `p = σ(ΔF + ln(β/(1-β)))`.

---

### publish_site

Publish a site contribution to the relay.

**Input:**
- `private_key` - Node's private key (nsec or hex)
- `model_id` - Model identifier
- `model_version` - Version of the model card used
- `delta_eta` - Site contribution (Δη)
- `cavity_members` - Members of the cavity this was trained against
- `blob_data` - Raw binary of Δη for Blossom upload

**Output:**
- `event_id` - Published event ID

**Behavior:**
1. Upload Δη blob to Blossom server
2. Construct kind-30101 event with tags: d, a, v, m, p, x
3. Sign with private key
4. Publish to relay

---

### publish_attestation

Publish a trust attestation to the relay.

**Input:**
- `private_key` - Attester's private key
- `target_pubkey` - Target peer's public key (hex)
- `scope` - Scope: "peer", "peer:model", or "peer:model:group"
- `inclusion_probability` - p value in (0,1)

**Output:**
- `event_id` - Published event ID

**Behavior:**
1. Construct kind-30102 event with tags: d, p, i
2. Sign with private key
3. Publish to relay

---

## Types

### Eta

Natural parameters for an exponential family distribution.

```python
class Eta:
    h: ndarray    # η₁ (mean parameters)
    Lam: ndarray  # η₂ (precision parameters)
```

### Site

A published contribution.

```python
class Site:
    author: str           # pubkey hex
    model_id: str         # model identifier
    model_version: int    # model card version
    delta_eta: Eta        # Δη
    cavity_members: list  # m tags (pubkey, event_id, p_vector)
    event_id: str         # Nostr event ID
```

### ModelCard

Model definition.

```python
class ModelCard:
    id: str               # model identifier
    title: str            # human-readable name
    version: int          # model card version
    groups: list[Group]   # parameter groups
    eta0: Eta             # base prior
    onnx_blob: str        # SHA-256 of ONNX file
    blossom_servers: list # Blossom server URLs
```

### Attestation

Trust rating for a peer.

```python
class Attestation:
    target: str           # target pubkey hex
    scope: str            # "peer", "peer:model", or "peer:model:group"
    p: float              # inclusion probability (0,1)
    event_id: str         # Nostr event ID
```

### Group

Named set of initializers sharing a distribution.

```python
class Group:
    name: str             # group identifier
    family: str           # distribution family
    initializers: list    # ONNX initializer names
```

---

## Constants

### Event Kinds

```python
KIND_MODEL_CARD = 30100
KIND_SITE_CONTRIBUTION = 30101
KIND_TRUST_ATTESTATION = 30102
```

### Distribution Families

```python
FAMILY_NORMAL = "normal"
FAMILY_GAMMA = "gamma"
FAMILY_BETA = "beta"
FAMILY_DIRICHLET = "dirichlet"
FAMILY_CATEGORICAL = "cat"
```

### Blob Encodings

```python
ENCODING_F64LE = "f64le"  # 8 bytes, default
ENCODING_F32LE = "f32le"  # 4 bytes, opt-in
ENCODING_I16LE = "i16le"  # 2 bytes, requires scales
ENCODING_I8 = "i8"        # 1 byte, requires scales
```
