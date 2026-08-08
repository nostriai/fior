# FIOR Protocol -- UI Integration Guide

## Event-to-UI Mapping

### Model Discovery (Marketplace Browse)

The marketplace home page lists available models.

**Query:** Fetch all kind-30100 events with the target relay filter.

**Data extracted from 30100 tags:**

| Tag        | Maps to                              |
|------------|--------------------------------------|
| `d`        | `model.id`                           |
| `title`    | `model.displayName`                  |
| `summary`  | `model.description`                  |
| `version`  | `model.version`                      |
| `agg`      | `model.aggregatorNpub`               |
| `onnx`     | `model.onnxBlossomEventId`           |
| `onnxhash` | `model.onnxHash` (verify integrity)  |
| `dist`     | `model.parameterGroups[].name`       |
| `group`    | `model.parameterGroups[].name`       |

**UI state computed from 30100:**

```
model.totalParameters = sum of all group/initializer dimensions
model.groupCount       = count of distinct groups
model.framework        = content.framework
model.minSamples       = content.min_samples
```

---

### Prior Display (Per-Model View)

When a user selects a model, show the current global parameters.

**Query:** Fetch latest kind-30101 event for this model `d` tag (order by `created_at` descending, limit 1).

**Data extracted from 30101 tags:**

| Tag     | Maps to                          |
|---------|----------------------------------|
| `round` | `prior.aggregationRound`         |
| `d`     | `prior.modelId`                  |
| `η`     | `prior.groupParameters[name]`    |

**UI state computed from 30101:**

```
for each η tag:
    buf = hex.decode(η.data)
    floats = float64LE.decode(buf)
    prior.groupParams[η.name] = {
        naturalParams: floats,
        dim: len(floats) / 2,  // for Normal; varies by distribution
    }
```

**Display:** Show round number, node count, total samples. Do NOT show raw `η` vectors in the UI -- they are not human-readable. Show derived summary:
```
prior.groupSummary[η.name] = {
    paramCount: dim,
    distribution: from model card,
    // Derived from natural params (example for Normal):
    meanMagnitude: sqrt(mean(μ_i²)),  // rough signal of "how active" this group is
    precisionMagnitude: mean(-2 * η_d+i),  // average certainty
}
```

---

### Marketplace Listing (Per-Posterior View)

List posterior submissions alongside trust data.

**Query:** Fetch kind-30102 events for the model. For each, fetch latest 30105 snapshot.

**Data extracted from 30102:**

| Tag | Maps to                        |
|-----|--------------------------------|
| `d` | `posterior.modelId`            |
| `p` | `posterior.priorEventId`       |
| `η` | `posterior.groupParams[name]`  |

**Metadata from 30102 content:**
```
posterior.samples       = content.samples
posterior.elbo          = content.elbo
posterior.durationSec   = content.duration_sec
posterior.frameworkVer  = content.framework_version
```

**Data extracted from 30105 (Trust Snapshot):**

| Tag group field | Maps to                                    |
|-----------------|--------------------------------------------|
| `name`          | `trust.groupScores[name].name`             |
| `mean_score`    | `trust.groupScores[name].meanScore`        |
| `std_score`     | `trust.groupScores[name].stdScore`         |
| `evaluator_count` | `trust.groupScores[name].evaluatorCount` |

**UI state computed from 30102 + 30105:**

```
posterior.trustSummary = {
    overall: {
        evaluatorCount: max(group.evaluatorCount),
        // Weighted average of group mean scores, weighted by group dim
        aggregateScore: sum(score * dim) / sum(dim),
    },
    perGroup: trust.groupScores,  // direct pass-through from 30105
}
```

**Display -- Marketplace Grid Item:**

```
┌─────────────────────────────────────┐
│                                     │
│  npub1abc1234... (author)           │
│  8,234 samples | 42s training       │
│  ELBO: -1247.3                      │
│                                     │
│  ┌─ Trust ────────────────────────┐│
│  │ 3 evaluators                    ││
│  │ fc-layers:  Δ+48.6 (strong)    ││
│  │ convs:      Δ-3.2  (weak)     ││
│  └─────────────────────────────────┘│
│                                     │
│  [Download] [Evaluate] [Use as Prior]│
└─────────────────────────────────────┘
```

---

### Trust Evaluation Workflow (User Initiates Back-Test)

A user clicks "Evaluate" on a posterior. The client:

1. Fetches the posterior's `η` data
2. Loads the user's local private data (client-side only, never leaves the machine)
3. Splits into train/holdout
4. Trains a control model (user's prior + train split)
5. Evaluates both control and tested posterior on holdout
6. Computes per-group ELBO delta / log-likelihood delta
7. Publishes kind-30104

**UI state during evaluation:**

```
eval.status  = "running" | "complete" | "error"
eval.progress = { currentGroup, groupsDone, totalGroups }
eval.results = {
    postEventId: "...",
    metric: "elbo_delta",
    holdoutSamples: 1647,
    controlElbo: -1247.3,
    testedElbo: -1198.7,
    perGroup: [
        {name: "fc-layers", score: 48.6, variance: 12.3},
        {name: "convs",    score: -3.2, variance: 8.1},
    ]
}
```

**Display -- Evaluation Result:**

```
┌───────────────────────────────────────┐
│ Evaluation complete                   │
│                                       │
│ Holdout: 1,647 samples                │
│ Baseline ELBO: -1247.3                │
│ Tested ELBO:   -1198.7  (+48.6)      │
│                                       │
│ Per-group breakdown:                  │
│ fc-layers   +48.6 ± 12.3  ████████    │
│ convs        -3.2 ±  8.1  █░░░░░░░    │
│                                       │
│ [Publish Evaluation to Relay]         │
└───────────────────────────────────────┘
```

---

### Trust Snapshot Display (Aggregated Consensus)

When displaying a posterior in the marketplace, the UI queries the latest 30105 snapshot for that posterior. This avoids replaying all individual 30104 events.

**If no 30105 exists yet:** Show "No evaluations yet. Be the first to evaluate."

**If 30105 exists:**

```
┌─────────────────────────────────────────┐
│ Trust Consensus (3 evaluators)          │
│                                         │
│ Group        Score      Std    Signal   │
│ ─────────────────────────────────────  │
│ fc-layers    +48.6     ±12.3  ████░░   │
│ convs         -3.2     ± 8.1  █░░░░░   │
│                                         │
│ Aggregate: +22.7 (weighted by dim)      │
└─────────────────────────────────────────┘
```

**UI logic for signal strength indicator:**

```
function signalStrength(meanScore, stdScore):
    if meanScore <= 0:
        return "negative"
    z = abs(meanScore) / max(stdScore, epsilon)
    if z >= 2.0:
        return "strong_positive"
    if z >= 1.0:
        return "moderate_positive"
    return "weak_positive"
```

---

### Registration & Federation Status

**User's registered models:**

Query kind-30103 events by the user's npub, filter `action=join`, for each model `d`. If the latest event for that model has `action=leave`, the user is not registered.

**Per-model federation status in UI:**

```
model.nodeCount = count of distinct npubs with latest 30103 action=join
// This information is typically in the prior broadcast content.node_count
```

---

## Summary of UI Data Dependencies

| UI View             | Events Queried                        | Freshness Strategy     |
|---------------------|---------------------------------------|------------------------|
| Marketplace browse  | 30100 (all)                           | On mount + subscription|
| Model detail        | 30100 (latest for `d`), 30101 (latest)| On mount + subscription|
| Posterior list      | 30102 (all for `d`), 30105 (per post) | On mount + subscription|
| Evaluate posterior  | 30102 (target), local data            | Fetch once + compute   |
| Trust breakdown     | 30105 (latest for post)               | On mount               |
| My registrations    | 30103 (by author npub)                | On mount + subscription|

## Relay Subscription Pattern

```
// On marketplace mount -- subscribe to all FIOR events for the relay
const sub = relay.subscribe([
    {kinds: [30100, 30101, 30102, 30103, 30104, 30105]},
]);

// Client maintains an in-memory event store:
//   modelCards:     Map<d_tag, 30100_event>
//   priors:         Map<d_tag, 30101_event>    (latest only)
//   posteriors:     Map<eventId, 30102_event>
//   evaluations:    Map<postEventId, 30104_event[]>
//   snapshots:      Map<postEventId, 30105_event>  (latest only)
//   registrations:  Map<d_tag, {action, npub, timestamp}>
```

Subscription provides live updates: new posteriors and evaluations appear in the UI without polling.
