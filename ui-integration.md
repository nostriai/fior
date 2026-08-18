# FIOR Protocol - UI Integration Guide

## Event-to-UI Mapping

### Model Discovery (Marketplace Browse)

The marketplace home page lists available models.

**Query:** Fetch all kind-30100 events.

**Data extracted from 30100 tags:**

| Tag | Maps to |
|-----|---------|
| `d` | `model.id` |
| `t` | `model.displayName` |
| `s` | `model.description` |
| `v` | `model.version` |
| `o` | `model.onnxBlob` (SHA-256) |
| `D` | `model.parameterDistributions[]` |
| `g` | `model.parameterGroups[]` |

**UI state computed from 30100:**

```
model.totalParameters = sum of all group/initializer dimensions
model.groupCount = count of distinct groups
model.framework = content.framework
model.minSamples = content.min_samples
```

---

### Site Listing (Per-Model View)

When a user selects a model, show all site contributions.

**Query:** Fetch kind-30101 events for this model `d` tag.

**Data extracted from 30101:**

| Tag | Maps to |
|-----|---------|
| `d` | `site.modelId` |
| `a` | `site.modelCardRef` |
| `v` | `site.modelVersion` |
| `m` | `site.cavityMembers[]` |
| `p` | `site.memberPubkeys[]` |
| `x` | `site.deltaEtaBlob` |

**Metadata from 30101 content:**
```
site.samples = content.samples
site.freeEnergy = content.free_energy
site.durationSec = content.duration_sec
site.frameworkVer = content.framework_version
```

**Display - Site Grid Item:**

```
┌─────────────────────────────────────┐
│                                     │
│  npub1abc1234... (author)           │
│  8,234 samples | 42s training       │
│  Free energy: -1247.3               │
│                                     │
│  Built on: 3 peers                  │
│  Model version: 3                   │
│                                     │
│  [View] [Use as Prior]              │
└─────────────────────────────────────┘
```

---

### Trust Attestations

Display trust attestations for a peer.

**Query:** Fetch kind-30102 events where `p` tag matches the target pubkey.

**Data extracted from 30102:**

| Tag | Maps to |
|-----|---------|
| `d` | `attestation.scope` (target, model, or group) |
| `p` | `attestation.targetPubkey` |
| `i` | `attestation.inclusionProbability` |

**Display - Trust Panel:**

```
┌─────────────────────────────────────┐
│ Trust in npub1abc1234...            │
│                                     │
│ Overall:           0.82 ████████░░  │
│ fc-layers group:   0.91 █████████░  │
│ convs group:       0.64 ██████░░░░  │
│                                     │
│ Based on 5 attestations             │
└─────────────────────────────────────┘
```

---

### Publishing a Trust Attestation

User clicks "Trust" or "Distrust" on a peer's site.

**Client action:**

1. Compute local p for this peer using BMR
2. Optionally adjust based on own judgment
3. Publish kind-30102 with:
   - `d`: scope (peer, peer:model, or peer:model:group)
   - `p`: target pubkey
   - `i`: inclusion probability (0 to 1)

**UI state:**

```
attestation.status = "publishing" | "published" | "error"
attestation.scope = "peer" | "model" | "group"
attestation.value = 0.82  // user-adjusted p
```

---

## Summary of UI Data Dependencies

| UI View | Events Queried | Freshness Strategy |
|---------|----------------|-------------------|
| Marketplace browse | 30100 (all) | On mount + subscription |
| Model detail | 30100 (latest for `d`) | On mount + subscription |
| Site list | 30101 (all for `d`) | On mount + subscription |
| Trust panel | 30102 (for target pubkey) | On mount + subscription |

## Relay Subscription Pattern

```javascript
// On marketplace mount - subscribe to all FIOR events
const sub = relay.subscribe([
    {kinds: [30100, 30101, 30102]},
]);

// Client maintains an in-memory event store:
//   modelCards:    Map<d_tag, 30100_event>
//   sites:         Map<event_id, 30101_event>
//   attestations:  Map<target_pubkey, 30102_event[]>
```

Subscription provides live updates: new sites and attestations appear in the UI without polling.

---

## Trust Display Logic

### Computing Local p

The client computes p for each peer using BMR:

```javascript
function computeP(site, trustedSites, eta0) {
    // Build cavity (prior without this peer)
    const cavity = composePrior(eta0, trustedSites, pValues);
    
    // Compute ΔF at full weight
    const deltaF = bmrDeltaF(
        cavity + site.deltaEta,  // with peer at full weight
        cavity,                  // without peer
        cavity + localLikelihood, // with local data
        cavity + site.deltaEta + localLikelihood // both
    );
    
    // Resolve p from ΔF and prior β
    return pFrom(deltaF, beta);
}
```

### Displaying p

p is a probability in (0,1). Display as:
- **Bar**: visual width proportional to p
- **Color**: red (0) → yellow (0.5) → green (1)
- **Label**: "distrusting" (< 0.3), "neutral" (0.3-0.7), "trusting" (> 0.7)

### Attestations Feed

Show recent attestations in a feed:

```
┌─────────────────────────────────────┐
│ Recent Trust Updates                │
│                                     │
│ npub1abc... trusts npub1def... 0.82 │
│ npub1ghi... trusts npub1jkl... 0.45 │
│ npub1mno... distrusts npub1pqr...   │
│                                     │
└─────────────────────────────────────┘
```
