# FIOR Protocol Specification v2

## Overview

Federated Inference Over Relays. Nodes train Bayesian models on local private data and exchange posterior distribution parameters over Nostr. Model structure is defined using ONNX. Distribution metadata layers on top of ONNX initializer tensors.

## Why ONNX

ONNX (Open Neural Network Exchange) is the standard interchange format for ML models. It defines:

- A directed acyclic computation graph (operators, types, shapes)
- Initializer tensors (the learnable parameters)
- Versioned operator specifications
- Tooling to export from PyTorch, TensorFlow, scikit-learn, JAX, etc.

FIOR does not need to reinvent model description. An `.onnx` file defines the inputs, outputs, parameter names, shapes, and computation graph. FIOR layers on top:

1. **Distribution assignments** -- which exponential family distribution models each initializer tensor
2. **Natural parameter wire format** -- the federated exchange of those distributions' natural parameters
3. **Aggregation rules** -- how the aggregator merges posteriors
4. **Trust evaluation** -- how nodes rate each other's contributions

## Event Kinds

| Kind   | Name               | Description                                      |
|--------|--------------------|--------------------------------------------------|
| 30100  | Model Card         | Points to ONNX file, defines distribution mapping |
| 30101  | Prior Broadcast    | Aggregator publishes current global parameters    |
| 30102  | Posterior Submit   | Node submits locally-trained parameters           |
| 30103  | Node Registration  | Node joins or leaves a federation                 |
| 30104  | Trust Evaluation   | Node rates a posterior submission via back-testing |
| 30105  | Trust Snapshot     | Aggregator publishes consensus scores for UI display |

---

## 30100 -- Model Card (Replaceable)

Parameterized replaceable event (NIP-33). The `d` tag carries the model identifier. The ONNX file is hosted as a Blossom blob -- the model card references it by the Nostr event ID of the Blossom upload.

### Tags

| Tag       | Required | Description                                        |
|-----------|----------|----------------------------------------------------|
| `d`       | yes      | Model identifier (slug, e.g. `pump-failure-v1`)    |
| `title`   | yes      | Human-readable model name                          |
| `summary` | no       | Short description                                  |
| `version` | no       | Model version, increment on schema changes         |
| `agg`     | yes      | Aggregator npub (hex)                              |
| `onnx`    | yes      | Nostr event ID of the ONNX file (Blossom upload)   |
| `onnxhash`| no       | SHA256 of the `.onnx` file (hex encoded)            |
| `dist`    | yes*     | Distribution assignment for ONNX initializer tensors|
| `group`   | no*      | Named group of initializers sharing a distribution  |

`*` At least one `dist` or `group` tag. If every initializer has the same distribution, a single `group` tag with `*` suffices.

### `dist` Tag Format

Maps an ONNX initializer tensor name to a distribution type and natural parameter encoding.

```
["dist", "<init_name>", "<dist>"]
```

| Field       | Type   | Description                                              |
|-------------|--------|----------------------------------------------------------|
| `init_name` | string | ONNX initializer tensor name (e.g. `fc1.weight`)        |
| `dist`      | string | `normal`, `gamma`, `dirichlet`, `cat` |

### `group` Tag Format

Groups multiple initializer tensors under a shared distribution. This reduces wire overhead when parameters share the same distribution type and can be concatenated into a single vector.

```
["group", "<name>", "<dist>", "<init1>", "<init2>", ...]
```

| Field   | Type   | Description                                   |
|---------|--------|-----------------------------------------------|
| `name`  | string | Group identifier (e.g. `linear-layers`)       |
| `dist`  | string | `normal`, `gamma`, `dirichlet`, `cat`         |
| `init*` | string | ONNX initializer tensor names in this group   |

A special group name `*` means "all initializers not otherwise assigned, with this distribution."

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
    ["version", "1.0"],
    ["agg", "npub1aggregatorhex..."],
    ["onnx", "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"],
    ["onnxhash", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    ["dist", "fc1.weight", "normal"],
    ["dist", "fc1.bias", "normal"],
    ["dist", "fc2.weight", "normal"],
    ["dist", "fc2.bias", "normal"]
  ],
  "content": "{\"framework\":\"pyro\",\"onnx_opset\":18,\"min_samples\":500,\"license\":\"MIT\"}"
}
```

With groups:

```json
{
  "kind": 30100,
  "tags": [
    ["d", "pump-failure-v1"],
    ["title", "Pump Failure Predictor"],
    ["agg", "npub1aggregatorhex..."],
    ["onnx", "abcdef1234567890..."],
    ["onnxhash", "9f86d08188..."],
    ["group", "fc-layers", "normal", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    ["group", "convs", "normal", "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]
  ],
  "content": "{\"framework\":\"pyro\"}"
}
```

---

## 30101 -- Prior Broadcast

Published by the aggregator. The latest (by `created_at`) supersedes all prior broadcasts for the same model.

### Tags

| Tag     | Required | Description                                      |
|---------|----------|--------------------------------------------------|
| `d`     | yes      | Model identifier                                 |
| `round` | no       | Aggregation round number                          |
| `p`     | no       | Prior event ID this was computed from (chain)     |
| `η`     | yes*     | Natural parameter vectors per group/initializer   |

`*` At least one `η` tag.

### `η` Tag Format

```
["η", "<name>", "<encoding>", "<data>"]
```

| Field      | Type   | Description                                              |
|------------|--------|----------------------------------------------------------|
| `name`     | string | Group name or initializer name (matches model card)       |
| `encoding` | string | `hex` for raw LE float64 binary encoded as lowercase hex |
| `data`     | string | Encoded natural parameter vector                          |

**Encoding per distribution:**

For a group/initializer with `d` scalar parameters and distribution `normal`:
- Vector of `2d` float64 LE bytes
- First `d` values: η₁..η_d = μ_i / σ_i² (mean natural parameters)
- Last `d` values: η_d+1..η_2d = -1 / (2 * σ_i²) (precision natural parameters)

For `gamma` with `d` parameters:
- Vector of `2d` float64 LE bytes
- First `d`: η₁..η_d = α_i - 1
- Last `d`: η_d+1..η_2d = -β_i

For `dirichlet` with `k` categories:
- Vector of `k` float64 LE bytes: η_1..k = α_i - 1

For `cat` with `k` categories:
- Vector of `k` float64 LE bytes: η_1..k = log(P(category_i))

### Content

Optional JSON with aggregation metadata.

### Example

```json
{
  "kind": 30101,
  "tags": [
    ["d", "pump-failure-v1"],
    ["round", "42"],
    ["η", "fc-layers", "hex", "a1b2c3d4e5f6..."],
    ["η", "convs",    "hex", "0f1e2d3c4b5a..."]
  ],
  "content": "{\"node_count\":17,\"total_samples\":102400,\"delta\":0.0003}"
}
```

---

## 30102 -- Posterior Submit

Published by a node after local training. The aggregator processes these into the next prior.

### Tags

| Tag     | Required | Description                                            |
|---------|----------|--------------------------------------------------------|
| `d`     | yes      | Model identifier                                       |
| `p`     | yes      | Prior event ID this posterior was trained from         |
| `η`     | yes*     | Natural parameter vectors (same format as prior)        |

`*` At least one `η` tag. Must match the groups/initializers in the model card.

### Content

JSON with training metadata:

```json
{
  "samples": 8234,
  "elbo": -1247.3,
  "duration_sec": 42.1,
  "framework_version": "pyro-1.9.0"
}
```

---

## 30103 -- Node Registration

Published by a node to announce joining or leaving a model's federation.

### Tags

| Tag      | Required | Description            |
|----------|----------|------------------------|
| `d`      | yes      | Model identifier       |
| `action` | yes      | `join` or `leave`      |

---

## 30104 -- Trust Evaluation

Published by a node that has back-tested another node's posterior submission against their own local holdout data. Carries a per-group variance/improvement score so that parameter contributions can be weighted by empirical accuracy rather than taken at face value.

The evaluator:
1. Downloads a posterior (30102) from Node B
2. Splits their own local data into train/holdout
3. Trains on the train split using B's posterior as prior → gets their own "control" posterior
4. Evaluates B's posterior directly on the holdout split → measures prediction error
5. Evaluates their control posterior on the holdout split → measures baseline error
6. For each parameter group, computes the **ELBO improvement** (or log-likelihood delta) attributable to B's parameter offsets vs the prior
7. Publishes per-group deltas

### Tags

| Tag     | Required | Description                                            |
|---------|----------|--------------------------------------------------------|
| `d`     | yes      | Model identifier                                        |
| `post`  | yes      | Event ID of the posterior being evaluated               |
| `group` | yes*     | Per-group evaluation result                             |
| `metric`| yes      | Top-level metric: `elbo_delta`, `log_lik_delta`, or `accuracy_delta` |
| `samples`| no      | Number of holdout samples used for evaluation            |

`*` At least one `group` tag, one per group/initializer in the model being evaluated.

### `group` Tag Format

```
["group", "<name>", "<score>", "<variance>"]
```

| Field      | Type   | Description                                                  |
|------------|--------|--------------------------------------------------------------|
| `name`     | string | Group/initializer name (matches model card)                  |
| `score`    | string | Improvement metric for this group (float as string)          |
| `variance` | string | Variance/uncertainty of the improvement estimate (float as string) |

**Interpretation:**

- `score > 0` -- B's parameters improved predictions on the evaluator's holdout data
- `score ≈ 0` -- B's parameters made no difference
- `score < 0` -- B's parameters degraded predictions
- `variance` -- how variable this improvement is across the holdout set (high variance = inconsistent, low trust)

### Content

JSON with evaluation methodology:

```json
{
  "holdout_samples": 1647,
  "holdout_start": "2025-06-01T00:00:00Z",
  "holdout_end": "2025-09-30T23:59:59Z",
  "control_elbo": -1247.3,
  "tested_elbo": -1198.7,
  "notes": "tested against Q3 2025 sensor data from plant B line 3"
}
```

### Example

```json
{
  "kind": 30104,
  "tags": [
    ["d", "pump-failure-v1"],
    ["post", "deadbeef1234..."],
    ["metric", "elbo_delta"],
    ["samples", "1647"],
    ["group", "fc-layers", "48.6", "12.3"],
    ["group", "convs", "-3.2", "8.1"]
  ],
  "content": "{\"holdout_samples\":1647,\"control_elbo\":-1247.3,\"tested_elbo\":-1198.7}"
}
```

Interpretation: `fc-layers` group improved ELBO by 48.6 ± 12.3 points (strong positive). `convs` group degraded slightly (-3.2) with moderate variance. A consumer would weight `fc-layers` parameters heavily and discount `convs`.

### Trust Consensus

Per-group scores from multiple evaluators form a **consensus signal**:

1. Node B submits posterior with new parameters
2. Nodes A, C, D, E independently back-test B's posterior on their own holdout data
3. Each publishes a 30104 with per-group scores
4. Any consumer (including the aggregator) queries all 30104 events referencing B's posterior
5. For each group, the consumer computes a weighted consensus: evaluators with historically consistent scores get higher weight (meta-evaluation)
6. Groups with positive consensus improve their weight in aggregation; groups with negative or highly variant consensus get suppressed

This is subjective -- each node chooses which evaluators to trust and how to weight them. The aggregator may apply consensus weighting when merging posteriors, but the consensus is verifiable by anyone replaying the evaluation chain.

---

## 30105 -- Trust Snapshot

Published by the aggregator (or any node computing the summary). Aggregates all 30104 Trust Evaluations for a specific posterior submission into per-group consensus scores. This gives the marketplace UI a fast lookup without replaying the full evaluation chain.

A client can verify the snapshot by fetching the referenced 30104 events and recomputing.

### Tags

| Tag     | Required | Description                                            |
|---------|----------|--------------------------------------------------------|
| `d`     | yes      | Model identifier                                        |
| `post`  | yes      | Event ID of the posterior being summarized              |
| `round` | no       | Aggregation round when this snapshot was computed       |
| `group` | yes*     | Per-group consensus result                              |

`*` At least one `group` tag, one per group/initializer in the model.

### `group` Tag Format

```
["group", "<name>", "<mean_score>", "<std_score>", "<evaluator_count>"]
```

| Field             | Type   | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| `name`            | string | Group/initializer name (matches model card)              |
| `mean_score`      | string | Mean improvement score across all evaluators (float as string) |
| `std_score`       | string | Standard deviation of scores across evaluators (float as string) |
| `evaluator_count` | string | Number of evaluators that back-tested this posterior     |

**Interpretation for UI:**

| Signal                 | Display                                    |
|------------------------|--------------------------------------------|
| mean_score >> 0, low std | "Strong consensus: improves predictions" |
| mean_score > 0, high std  | "Mixed reviews, trending positive"       |
| mean_score ≈ 0          | "No significant improvement"             |
| mean_score < 0          | "Consensus: degrades predictions"         |
| evaluator_count = 0      | "No evaluations yet"                      |

### Content

JSON with computation metadata:

```json
{
  "computed_by": "npub1aggregatorhex...",
  "eval_event_ids": ["eventid1", "eventid2", "eventid3"],
  "method": "unweighted_mean",
  "computed_at": "2025-08-08T12:00:00Z"
}
```

### Example

```json
{
  "kind": 30105,
  "tags": [
    ["d", "pump-failure-v1"],
    ["post", "deadbeef1234..."],
    ["round", "42"],
    ["group", "fc-layers", "48.6", "12.3", "3"],
    ["group", "convs", "-3.2", "8.1", "3"]
  ],
  "content": "{\"computed_by\":\"npub1agghex...\",\"eval_event_ids\":[\"id1\",\"id2\",\"id3\"],\"method\":\"unweighted_mean\"}"
}
```

### Refresh Policy

The aggregator publishes a new snapshot when:
- N new evaluations arrive for the posterior (configurable, e.g., every 3 evaluations)
- A configurable time window elapses

Snapshots are non-replaceable -- each is a distinct event. Consumers take the latest by `created_at`.

---

## Aggregation Algorithm (Stage 1)

For each posterior submission at the aggregator:

```
For each group g:
    if posterior.prior_ref == current_prior:
        η_diff[g] = η_local[g] - η_prior_ref[g]
    else:
        η_diff[g] = η_local[g] - fetch_prior(posterior.prior_ref)[g]

    η_global[g] = η_global[g] + η_diff[g]
```

After a configurable interval or batch size, the aggregator publishes a new Prior Broadcast (30101) with `η_global`.

### Future: Consensus-Weighted (Stage 2)

Once trust data has accumulated, the aggregator may weight contributions by per-group consensus scores from 30105 snapshots. This is deferred to stage 2, where adversarial actors are introduced and the system must route around poisoned parameters.

### Prior Initialization

Before any training, the aggregator publishes a prior with all natural parameters set to zero:

```
η[g] = [0.0] * len(g)        (zero vector → μ=0, σ²=∞ → "no information")
```

---

## Flow Summary

```
1. Creator exports model to ONNX, uploads via Blossom
2. Creator publishes Model Card (30100) referencing ONNX event ID
3. Aggregator publishes initial Prior (30101): η = zero vector
4. Node publishes Registration (30103, action=join)
5. Node fetches ONNX file, latest Prior (30101)
6. Node trains locally on private data
7. Node publishes Posterior Submit (30102)
8. Other nodes back-test the posterior, publish Trust Evaluations (30104)
9. Aggregator publishes Trust Snapshot (30105) summarizing evaluations
10. Aggregator sums posteriors into η_global, publishes updated Prior (30101)
11. All nodes fetch new prior, retrain, repeat from step 6
```

## Security

- All events are Nostr-signed by the publishing keypair.
- Aggregator verifies posterior η lengths match ONNX initializer shapes from model card.
- Aggregator verifies node is registered before accepting posteriors.
- Aggregator may enforce minimum sample counts and outlier rejection.
- Trust evaluations are per-evaluator, subjective, not a centralized score.
- ONNX file integrity is verifiable via `onnxhash` tag against the Blossom blob.
