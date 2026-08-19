# FIOR Protocol -- UI Integration Guide

## What changed from v2

The v2 marketplace was a list of posterior submissions ranked by a consensus trust score
published by the aggregator. Neither exists in v3:

- Sites are **addressable** — one per `(author, model)`, latest wins. There is no growing
  list of submissions to page through. The marketplace shows a **membership roster**, not
  a feed.
- There is **no consensus trust score**. p is local to each client and derived from its
  own data, so two honest clients will legitimately disagree about the same peer. The UI
  shows *your* p, and never a total. There are no published scores to rank by.
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
│ p  —  (fetch to evaluate)                [Fetch]      │
└──────────────────────────────────────────────────────┘
```

---

## Parameter Fetch and Local Trust

This is the interaction v2 had no equivalent for. Because η lives in blobs, the client
decides what to download, and p cannot be computed for anything not downloaded.

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
// the peer enters the full model at weight 1, not at its current p;
// the reference holds every OTHER peer at its current p
deltaF[peer][group] = A(η_q⁺) + A(η_p⁻) − A(η_q⁻) − A(η_p⁺)
p[peer][group]   = 1 / (1 + (b/a)·exp(−deltaF[peer][group]))
```

`(a, b)` is derived locally from the attestation store — each attestation is a scalar
`p_{B→C}` contributing `p_{A→B}·κ_r·(p_{B→C}, 1−p_{B→C})` at the viewer's own `κ_r`.
Absent any attestations it is `(1, 1)` and `p` reduces to `σ(ΔF)`.

**Two quantities, two names.** `β = a/(a+b)` is the *prior* inclusion probability, set by
attestations alone and equal to `0.5` for a stranger. `p` is the *posterior*,
`P(z = 1 | this viewer's data)`, and it is what weights the composition. Do not label a
`p` as "β" in the UI: the panel below shows both, and showing a posterior under the name
of its prior is the specific confusion this guide previously had.

**Trust panel — note this is per-viewer, not consensus:**

```
┌─ Your weighting of npub1abc1234… ────────────────────┐
│ Computed from your data. Other nodes will differ.    │
│                                                      │
│ Group        ΔF        p  (posterior)                │
│ ────────────────────────────────────────────────     │
│ fc-layers   +48.6    0.94  ████████░░                │
│ convs        −3.2    0.28  ██░░░░░░░░                │
│                                                      │
│ Prior β = 0.77   Beta(3.0, 0.9), 2 attestations, κᵣ=4│
│                                              [Attest]│
└──────────────────────────────────────────────────────┘
```

Label the panel explicitly as local. The single largest way to mislead a user here is to
present a per-viewer weight as though it were a community verdict.

p must be recomputed whenever a peer's site changes, since `ΔF` describes the current
`Δη` and nothing else. A site event arriving over the subscription invalidates that
peer's p immediately.

---

## Composition State

```
composition = {
    members: [{pubkey, siteEventId, p: {perGroup}}],
    eta: {perGroup},
    invalid: [groupName],
    excluded: [{pubkey, reason: "stale" | "mismatched" | "unresolvable" | "unfetched"}],
}
```

**Surface a group that leaves its natural parameter domain** — with Normal groups, a
precision term going non-negative. Sites are differences, so a weighted sum of valid
sites need not be valid. It is usually the first visible symptom of an over-quantized or
adversarial site, and the protocol specifies no recovery, so the composition simply
cannot be used until the user drops a member or refetches at a wider encoding. Say that
plainly rather than presenting a silently unusable prior.

```
┌─ Your prior ─────────────────────────────────────────┐
│ 12 of 17 members included                            │
│ ⚠ group `convs` left its valid domain — cannot compose│
│ Excluded: 3 unfetched, 1 stale, 1 version mismatch   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Provenance

Every site carries the member list of the cavity it was fitted against — one `m` tag per
member, plus an indexed `p` tag per member pubkey.

```
site.members = [{pubkey, siteEventId, p: [perGroup]}]   // from `m` tags
```

**Two views fall out of this, and both are worth building.**

*Upstream* — what this peer built on. Render the member list as named peers with the `p`
each was given, so a user can see whose work a site rests on. A site with no `m` tags was
composed against `η₀` alone; label it "first round", not "no sources".

*Downstream* — who built on this peer. `{kinds:[30101], "#p":[peerPubkey]}` returns every
site that pinned this one as a member. This is the derivation graph, and it is the only
place in the protocol where a peer's influence on others is publicly visible rather than
private to each evaluator.

**Offer verification.** Members are pinned by event id, so the client can refetch each one,
recompute `η₀ + Σ p·Δη`, and check it against the site's own `Δη` plus the poster's
claimed posterior:

```
site.verification = "unverified" | "verifying" | "matches" | "mismatch"
```

`mismatch` means the published `Δη` is not the difference the author claims it is. Surface
it plainly — it is one of the few objectively checkable failures in the system.

Do not present the derivation graph as a ranking. Being built upon is not endorsement:
a site can be widely used and still be scored badly by everyone using it, because `p` is
per-evaluator and never published as a total.

---

## Attestations

**Query:** `{kinds:[30102], "#p":[peerPubkey]}` — who has attested about this peer.

```
attestation.p      = <scalar in (0,1)>   // the attester's POSTERIOR p, from `incl`
attestation.scope  = "peer" | "peer:model" | "peer:model:group"
```

An attestation is one number. It carries no confidence, because the publisher has none to
report — `ΔF` is never accumulated — and because a weight a publisher puts on its own
testimony is not something a reader should honour. The viewer's own `κ_r` supplies the
weight. **Do not render an uncertainty for an attestation**; earlier revisions exposed an
`sd` from a Beta pair, and there is no longer a pair, nor was the spread ever read by the
trust layer. A legacy `beta` tag displays as its single value, or `a/(a+b)` if a pair.

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
| Models by group    | 30100 (`#G` per group name)                 | none                 | On search              |
| Models by family   | 30100 (`#F` per distribution)               | none                 | On search              |
| Membership roster  | 30101 (latest per author for `d`)            | none                 | On mount + subscription|
| Local trust panel  | 30102 (`#p` per peer)                        | per-site Δη, on demand| Recompute on site change|
| Provenance (upstream)| the site's own `m` tags                     | members' Δη, to verify| On open                |
| Provenance (downstream)| 30101 (`#p` per peer)                    | none                 | On mount + subscription|
| Site citations     | 30101 (`#e` per event_id)                   | none                 | On open                |

---

## Relay Subscription Pattern

```js
const sub = relay.subscribe([
    {kinds: [30100, 30101, 30102]},
]);

// In-memory store. Note the addressable kinds are keyed by coordinate,
// not by event id — a new event for the same coordinate REPLACES.
//   modelCards:   Map<"30100:<pk>:<d>", event>
//   sites:        Map<"30101:<pk>:<d>", event>     // one per author per model
//   attestations: Map<"30102:<pk>:<d>", event>

// Separate, because blobs are not events and are not pushed:
//   params:       Map<blobHash, Float64Array>      // fetched on demand
```

Two subscription behaviours that differ from v2:

- **Replacement, not accumulation.** All six kinds are addressable. Keying the store by
  event id will accumulate superseded events and produce a roster with duplicate members
  and inflated counts.
- **A site event invalidates derived state.** When a 30101 arrives for a coordinate the
  store already holds, discard that peer's cached `Δη`, `ΔF`, and p. The blob hash has
  changed, and evidence about the old site describes something that no longer exists.
