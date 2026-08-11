# FIOR Protocol -- UI Integration Guide

## What changed from v2

The v2 marketplace was a list of posterior submissions ranked by a consensus trust score
published by the aggregator. Neither exists in v3:

- Sites are **addressable** — one per `(author, model)`, latest wins. There is no growing
  list of submissions to page through. The marketplace shows a **membership roster**, not
  a feed.
- There is **no consensus trust score**. β is local to each client and derived from its
  own data, so two honest clients will legitimately disagree about the same peer. The UI
  shows *your* β, and separately shows benchmark results, which are comparable and
  verifiable.
- Parameters are never in events. Every η payload is a Blossom blob, so the UI must
  distinguish "descriptor loaded" from "parameters fetched" and let the user choose when
  to spend the bandwidth.

---

## Model Discovery (Marketplace Browse)

**Query:** `{kinds:[30100]}`

**Data extracted from 30100 tags:**

| Tag       | Maps to                                 |
|-----------|-----------------------------------------|
| `d`       | `model.id`                              |
| `title`   | `model.displayName`                     |
| `summary` | `model.description`                     |
| `version` | `model.version`                         |
| `onnx`    | `model.onnx = {hash, bytes, servers}`   |
| `dist`    | `model.parameterGroups[].name`          |
| `group`   | `model.parameterGroups[].name`          |
| `x`       | `model.basePrior = {hash, enc, bytes}`  |
| `blossom` | `model.blossomServers[]`                |
| `ttl`     | `model.siteTtlSeconds`                  |

**Computed:**

```
model.totalParameters = sum of group dimensions from the ONNX graph
model.groupCount      = count of distinct groups
model.memberCount     = count of live 30101 sites (see below)
```

`memberCount` requires a second query. Use NIP-45 `COUNT` where the relay supports it
rather than fetching every site descriptor just to display a number.

---

## Membership Roster (Per-Model View)

**Query:** `{kinds:[30101], "#d":[modelId]}`

The relay returns exactly one site per author. Sites are descriptors — the `x` tag
carries a hash, not parameters.

| Tag          | Maps to                                     |
|--------------|---------------------------------------------|
| `d`          | `site.modelId`                              |
| `v`          | `site.modelCardVersion`                     |
| `x`          | `site.blob = {hash, enc, bytes, servers}`   |
| `a` (cavity) | `site.cavityRef`                            |
| `expiration` | `site.expiresAt`                            |

**Metadata from content** — advisory only, never an input to composition or trust:

```
site.samples, site.freeEnergy, site.durationSec, site.frameworkVersion
```

**Client-side filtering the UI must apply:**

```
site.stale     = expiresAt < now, or created_at + model.siteTtlSeconds < now
site.mismatched = site.modelCardVersion !== model.version
```

Stale and mismatched sites are excluded from composition. Show them greyed with the
reason rather than hiding them — a roster that silently shrinks is confusing.

**Roster row:**

```
┌──────────────────────────────────────────────────────┐
│ npub1abc1234…                            8,234 samples│
│ i8 · 24.6 KB · not fetched                            │
│ β  —  (fetch to evaluate)                [Fetch]      │
└──────────────────────────────────────────────────────┘
```

---

## Parameter Fetch and Local Trust

This is the interaction v2 had no equivalent for. Because η lives in blobs, the client
decides what to download, and β cannot be computed for anything not downloaded.

**Per-site fetch state:**

```
site.fetchState = "descriptor" | "fetching" | "resolved" | "unresolvable"
```

`unresolvable` is **not an error**. The blob could not be retrieved from any hinted
server or the model card's `blossom` list. The member is simply absent from composition,
and returns if the bytes become retrievable. Display it as a neutral state with a retry
affordance, not a failure.

**After fetch, the client scores every resolved peer locally:**

```
deltaF[peer][group] = A(η_q) + A(η_p − βΔη) − A(η_q − βΔη) − A(η_p)
beta[peer][group]   = 1 / (1 + (b/a)·exp(−deltaF[peer][group]))
```

`(a, b)` comes from the local attestation store; absent any attestations it is `(1, 1)`
and β reduces to `σ(ΔF)`.

**Trust panel — note this is per-viewer, not consensus:**

```
┌─ Your weighting of npub1abc1234… ────────────────────┐
│ Computed from your data. Other nodes will differ.    │
│                                                      │
│ Group        ΔF        β                             │
│ ────────────────────────────────────────────────     │
│ fc-layers   +48.6    0.94  ████████░░                │
│ convs        −3.2    0.28  ██░░░░░░░░                │
│                                                      │
│ Prior: Beta(14.2, 3.1) from 2 attestations           │
│                                              [Attest]│
└──────────────────────────────────────────────────────┘
```

Label the panel explicitly as local. The single largest way to mislead a user here is to
present a per-viewer weight as though it were a community verdict.

β must be recomputed whenever a peer's site changes, since `ΔF` describes the current
`Δη` and nothing else. A site event arriving over the subscription invalidates that
peer's β immediately.

---

## Composition State

```
composition = {
    members: [{pubkey, siteEventId, beta: {perGroup}}],
    rho: 1.0,
    eta: {perGroup},
    excluded: [{pubkey, reason: "stale" | "mismatched" | "unresolvable" | "unfetched"}],
}
```

**Show `rho` when it is below 1.** Backtracking means some composed group left its valid
natural parameter domain — with Normal groups, that a precision term went non-negative.
It is usually the first visible symptom of an over-quantized or adversarial site, and
silently damping hides the signal.

```
┌─ Your prior ─────────────────────────────────────────┐
│ 12 of 17 members included                            │
│ ρ = 0.5  ⚠ damped — a group left its valid domain    │
│ Excluded: 3 unfetched, 1 stale, 1 version mismatch   │
│                          [Publish as Composite]      │
└──────────────────────────────────────────────────────┘
```

---

## Composites

**Query:** `{kinds:[30102], "#p":[pubkey]}` for composites including a given member, or by
author for a node's own.

| Tag       | Maps to                                            |
|-----------|----------------------------------------------------|
| `d`       | `composite.id`                                     |
| `m`       | `composite.members[] = {pubkey, siteEventId, beta[]}` |
| `rho`     | `composite.rho`                                    |
| `purpose` | `composite.purpose`                                |
| `x`       | `composite.blob`                                   |

A composite is a receipt, not a belief. `purpose` drives the label, and the distinction
matters: in an `ablation`, `β = 0` for a member is the second arm of an experiment, not
an accusation. Render ablations as experiments and never as negative judgements.

**Offer verification.** Every member is pinned by event id, so the client can refetch,
recompute `η₀ + ρ Σ βΔη`, and compare against the published blob:

```
composite.verification = "unverified" | "verifying" | "matches" | "mismatch"
```

`mismatch` is a strong, publishable signal — the author's arithmetic does not reproduce.

---

## Benchmarks

Benchmark results are the only cross-client comparable numbers in the system.

**Descriptor** — `{kinds:[30105], "#a":[modelCoord]}`:

```
descriptor.id, descriptor.title
descriptor.set     = {type: "public" | "private", hash?, servers?, protocol?}
descriptor.metrics = [{name, direction: "higher" | "lower"}]
```

**Leaderboard** — `{kinds:[30103], "#a":["30105:<pk>:<benchId>"]}`:

```
result.target  = {coord, eventId, authorPubkey}
result.metrics = [{name, value, stderr}]
result.samples
```

**Two rules the UI must enforce:**

1. **Never pool results across descriptors.** A log-likelihood on one holdout and one on
   another are not on the same scale. Group the leaderboard by descriptor; a combined
   ranking is meaningless.
2. **Attribute `private` descriptors to their publisher.** A private holdout supports
   comparison only within one publisher's series. Show it as "measured by npub1…", not as
   a community result.

```
┌─ Pump sensors, Q3 2025 holdout ──────────────────────┐
│ public set · 1b4f0e… · 16,470 rows          verifiable│
│                                                       │
│  log_lik ↑        target                     samples  │
│  −1198.7 ±14.2    composite trust-weighted     1,647  │
│  −1211.4 ±13.8    composite uniform            1,647  │
│  −1247.3 ±15.1    composite ablate-C           1,647  │
└───────────────────────────────────────────────────────┘
```

The ablation reading positive here is exactly the evidence loop the trust layer runs on:
dropping member C cost 48.6 nats, so C is earning its weight.

---

## Attestations

**Query:** `{kinds:[30104], "#p":[peerPubkey]}` — who has attested about this peer.

```
attestation.beta   = {a, b}
attestation.mean   = a / (a + b)
attestation.sd     = sqrt(a·b / ((a+b)²·(a+b+1)))
attestation.scope  = "peer" | "peer:model" | "peer:model:group"
```

**Publishing is voluntary and must be presented that way.** A node that publishes none is
fully functional. Do not gate features on attesting, and do not nag.

Publishing an attestation reveals a trust judgement about a named peer permanently and
publicly. Confirm before publishing, and show the scope plainly — a peer-level
attestation is a much broader statement than a group-level one.

```
┌─ Attest to npub1abc1234… ────────────────────────────┐
│ Your local evidence: ΔF +48.6 on fc-layers           │
│                                                       │
│ Scope:  ( ) this peer, everywhere                     │
│         ( ) this peer on pump-failure-v1              │
│         (•) this peer on pump-failure-v1 / fc-layers  │
│                                                       │
│ Publishing is public and permanent.   [Cancel][Publish]│
└───────────────────────────────────────────────────────┘
```

---

## Data Dependencies

| UI View            | Events Queried                              | Blobs Fetched        | Freshness              |
|--------------------|---------------------------------------------|----------------------|------------------------|
| Marketplace browse | 30100 (all), 30101 COUNT                    | none                 | On mount + subscription|
| Model detail       | 30100 (latest for `d`)                       | ONNX, η₀             | On mount + subscription|
| Membership roster  | 30101 (latest per author for `d`)            | none                 | On mount + subscription|
| Local trust panel  | 30104 (`#p` per peer)                        | per-site Δη, on demand| Recompute on site change|
| Composite detail   | 30102, plus pinned 30101s to verify          | composite η + members| On open                |
| Leaderboard        | 30105 (per model), 30103 (`#a` per descriptor)| none                | On mount + subscription|

---

## Relay Subscription Pattern

```js
const sub = relay.subscribe([
    {kinds: [30100, 30101, 30102, 30103, 30104, 30105]},
]);

// In-memory store. Note the addressable kinds are keyed by coordinate,
// not by event id — a new event for the same coordinate REPLACES.
//   modelCards:   Map<"30100:<pk>:<d>", event>
//   sites:        Map<"30101:<pk>:<d>", event>     // one per author per model
//   composites:   Map<"30102:<pk>:<d>", event>
//   benchmarks:   Map<"30103:<pk>:<d>", event>
//   attestations: Map<"30104:<pk>:<d>", event>
//   descriptors:  Map<"30105:<pk>:<d>", event>

// Separate, because blobs are not events and are not pushed:
//   params:       Map<blobHash, Float64Array>      // fetched on demand
```

Two subscription behaviours that differ from v2:

- **Replacement, not accumulation.** All six kinds are addressable. Keying the store by
  event id will accumulate superseded events and produce a roster with duplicate members
  and inflated counts.
- **A site event invalidates derived state.** When a 30101 arrives for a coordinate the
  store already holds, discard that peer's cached `Δη`, `ΔF`, and β. The blob hash has
  changed, and evidence about the old site describes something that no longer exists.
