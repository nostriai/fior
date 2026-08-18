Interface: FederatedInferenceOverRelays

All methods run client-side. Composition and trust are computed locally from events any relay can serve. See
[protocol.md](protocol.md) for the wire format.

`signer` denotes a signing capability (NIP-07 handle, hardware signer, or an in-process
key). Private keys are never passed between components.

**Notation.** Peer inclusion is a Bernoulli indicator `z` with a Beta prior on its rate.
Two quantities, `β` and `p`, and this document keeps them apart:

- `β = a/(a+b)` — the **prior** inclusion probability, from attestations alone. What the
  caller believes before looking at its own data; `0.5` for a stranger.
- `p = P(z = 1 | my data)` — the **posterior**, after folding in `ΔF`. This is also the
  weight multiplying a peer's `Δη` in the composition. The prior is `η₀ + Σ z_n·Δη_n` for some unknown inclusion vector, and its expectation over `z_n` is `η₀ + Σ p_n·Δη_n`, so the weight *is* `E[z_n]`.

## Discovery

* Method: read_model_card
    - Input:
        - model_id - Model identifier (the `d` tag of a kind 30100 event).
        - relays - Relays to query.
    - Output: Model card: ONNX blob reference, group-to-family map, base prior `η₀`
      reference, model card version, Blossom fallback servers, site TTL.
    - Behavior: Fetches the latest kind 30100 for the model. Verifies the ONNX blob
      SHA-256 on fetch. The returned version pins every subsequent call for this model.

* Method: list_sites
    - Input:
        - model_id - Model identifier.
        - relays - Relays to query.
    - Output: One site descriptor per peer: author pubkey, event id, blob hash, encoding,
      blob size, server hints, declared model card version, advisory metadata.
    - Behavior: Fetches the latest kind 30101 per author for the model. Sites are
      addressable, so exactly one is current per `(author, model)`. Returns descriptors
      only — no parameter bytes are fetched. Sites whose declared version disagrees with
      the model card, or that have expired, are excluded.

* Method: fetch_site_params
    - Input:
        - site_descriptor - A descriptor returned by `list_sites`.
        - blossom_servers - Additional server roots to try after the descriptor's hints.
    - Output: `Δη` as a dense per-group vector, dequantized to f64.
    - Behavior: Fetches the blob, verifies its SHA-256, and decodes it against the model
      card's group order. Returns absent rather than raising when the blob cannot be
      resolved from any server — an unresolvable blob makes a member absent, not an error.

## Composition

* Method: compose_prior
    - Input:
        - model_card - As returned by `read_model_card`.
        - members - Peer contributions, each with its `Δη`, source event id, and per-group `p`.
    - Output: The local prior `η₀ + Σ p Δη` per group, and the member list that produced
      it — pubkey, site event id and `p` vector per member.
    - Behavior: Sums trust-weighted site contributions into a prior. Excludes the caller's
      own site. Raises rather than returning a group that falls outside the natural
      parameter domain of its family — sites are differences, so a weighted sum of valid
      sites need not be valid, and the specification defines no recovery from it. The
      returned member list is what `publish_site` must carry as provenance, so it pins each
      member by the event id actually summed, never by coordinate.

* Method: fit_local
    - Input:
        - prior - The composed local prior.
        - data - Local private data. Never leaves the node.
    - Output: The local posterior `η_q`, and the local free energy.
    - Behavior: Runs variational inference against the composed prior. Backend-specific;
      this is the boundary at which Pyro, NumPyro, or a closed-form solver plugs in.

* Method: compute_site
    - Input:
        - posterior - The local posterior from `fit_local`.
        - cavity - The prior it was fitted against.
    - Output: `Δη = η_post − η_cavity`.
    - Behavior: Forms the node's site contribution — the quantity that composes.

## Trust

* Method: score_peers
    - Input:
        - prior - The working prior currently in use.
        - posterior - The local posterior fitted against it.
        - members - Peer contributions with their current `p`.
    - Output: `ΔF` per peer per parameter group.
    - Behavior: Evaluates, in closed form via Bayesian model reduction, how each peer's
      contribution changes the local log evidence. Write `η_p^{−n} = η₀ + Σ_{m ≠ n}
      p_{A→m}·Δη_m` — peer n absent, every *other* peer still at its current `p` — and
      `η_p^{+n} = η_p^{−n} + Δη_n`, the same reference with n present at weight 1. With
      `η_q^{±n}` the corresponding posteriors:
      `ΔF = A(η_q^{+n}) + A(η_p^{−n}) − A(η_q^{−n}) − A(η_p^{+n})`.
      `ΔF > 0` means the peer would earn full weight. The peer must enter at unit weight,
      not at its current `p` — scaling by `p` makes `ΔF → 0` as `p → 0`, so a rejected
      peer drifts back to the prior mean `β` and can never be stably excluded. Requires no
      holdout split and no retraining. Evaluates `A()` in f64 regardless of the encoding
      a site arrived in.

* Method: bound_confidence
    - Input:
        - site_params - A peer's `Δη`.
        - others - The composed contribution of the caller's other peers, excluding both
          the caller and this peer.
        - c - Bound on how much more confident one peer may be than the rest combined.
    - Output: `Δη` with its precision bounded, implied mean unchanged.
    - Behavior: For Normal groups, clips the eigenvalues of the peer's precision in the
      metric of `others` so that `Λ_n ⪯ c·Λ_others`, and rescales `η₁` to preserve the
      implied mean. Bounds the influence a site can exert during the rounds before it is
      scored, which is not otherwise bounded by anything: `score_peers` asks whether a
      peer helps, never how sure it may be. Excluding the peer from `others` is required —
      a site that counts toward its own bound is not bounded. Local policy; changes no
      wire format.

* Method: novelty
    - Input:
        - site_params - A peer's `Δη`.
        - older - The `Δη` of every site the caller saw *before* this one.
    - Output: A scalar in `[0, 1]` — the fraction of the peer's claim that is its own.
    - Behavior: Returns the norm of `site_params` after projecting out the span of
      `older`, relative to its own norm. Zero means the peer republished a linear
      combination of what it had already read and contributed nothing; `score_peers`
      rewards exactly that, so this is the complement to it — `score_peers` prices what a
      site asserts, `novelty` prices what it adds. Costs one least-squares solve over
      public data and needs none of the caller's own data.
    - Notes: `older` must be ordered by something the publisher cannot backdate. The
      caller's own first-seen order is the cheap and correct choice; `created_at` is
      self-asserted and unusable. Ordering is what makes the result directional — against
      an unordered peer set the test flags the copied author exactly as hard as the
      copier. Callers SHOULD scale `p` by this value rather than substituting the residual
      for the site: the projection of a positive semi-definite precision block need not be
      positive semi-definite. Callers MUST skip the test when the member count approaches
      a group's ambient dimension (`d + d(d+1)/2` for a Normal group), where honest sites
      become linearly dependent by accident and every peer scores zero.
    - Warning: callers MUST NOT apply this without also applying `bound_confidence`.
      Measured, it makes confidence-fabrication attacks substantially worse on its own,
      because a site that plants a falsehood along an unobserved direction is maximally
      *novel* by construction — it scores above every honest peer on exactly the test
      meant to catch free-riders. The two methods are in tension and only their
      composition is safe.

* Method: resolve_inclusion
    - Input:
        - peer - Peer public key.
        - model_id - Model identifier.
        - group - Parameter group name.
        - delta_f - The peer's `ΔF` for this group, from `score_peers`.
        - attestation_prior - Beta parameters `(a, b)`, resolved from the local
          attestation store; defaults to `(1, 1)`.
    - Output: `p ∈ (0,1)`.
    - Behavior: Returns the posterior inclusion probability
      `p = 1 / (1 + (b/a)·exp(−ΔF))`. `ΔF` is recomputed each round and never stored;
      only `(a, b)` accumulates, and only from information the client cannot recompute
      for itself.

* Method: resolve_attestation_prior
    - Input:
        - peer - Peer public key.
        - model_id - Model identifier; optional.
        - group - Parameter group name; optional.
        - gamma - Pooling factor across hierarchy levels.
        - kappa_r - Pseudo-counts the caller grants one attestation. `1` to `4`.
    - Output: Beta parameters `(a, b)`.
    - Behavior: Resolves the hierarchical prior `(a, b)` — hence `β = a/(a+b)` — over the
      peer's inclusion, pooling `peer`, `peer:model`, and `peer:model:group` levels
      with discount `gamma`, and folding in transitive attestations from peers the
      caller already trusts. Each attestation `p_{B→C}` contributes
      `p_{A→B}·kappa_r·(p_{B→C}, 1−p_{B→C})`, so one contributor supplies at most
      `kappa_r` pseudo-counts and no per-contributor clip is required. Local policy —
      no part of this is normative.
    - Notes: keep `kappa_r` small.

## Publication

* Method: publish_site
    - Input:
        - signer - Signing capability.
        - model_id - Model identifier.
        - delta_eta - The site contribution from `compute_site`.
        - encoding - Numeric encoding; `f64le`, `f32le`, `i16le`, or `i8`.
        - members - The member list from `compose_prior` — the cavity this site was fitted
          against. Omit only when that cavity was `η₀` alone.
        - blossom_servers - Servers to upload to.
        - relays - Relays to publish to.
    - Output: A success status and the published event id.
    - Behavior: Uploads the `Δη` blob **before** publishing the kind 30101 that references
      it — the event is durable the moment a relay accepts it, and the blob is not.
      Quantizing encodings must use stochastic rounding; deterministic rounding biases
      every consumer's composed prior. Supersedes the caller's previous site for this
      model. Writes `members` as one `m` tag per member plus an indexed `p` tag per member
      pubkey, which is what makes `Δη = η_post − η_cavity` checkable by anyone and what
      makes "who built on my site" a queryable relation.

* Method: withdraw_site
    - Input:
        - signer - Signing capability.
        - model_id - Model identifier.
        - relays - Relays to publish to.
    - Output: A success status.
    - Behavior: Publishes a NIP-09 deletion for the caller's site. No registry is updated
      and no peer needs to act; each peer's next composition omits the caller.

* Method: publish_attestation
    - Input:
        - signer - Signing capability.
        - peer - Peer public key.
        - p - The caller's posterior inclusion probability for the peer, a single
          scalar in `(0,1)`.
        - model_id - Model identifier; omit for a peer-level attestation.
        - group - Parameter group; omit for a model-level attestation.
        - relays
    - Output: A success status.
    - Behavior: Publishes a belief about a peer's inclusion. Entirely optional: `p` is
      always computed and held locally, and a node that publishes no attestations and
      reads none is a fully functional participant. A like is `1`; a dislike is `0`.
    - Notes: the attestation carries the belief only, never how strongly a reader should
      hold it — that is `resolve_attestation_prior`'s `kappa_r`, chosen by the reader.
      There is deliberately no way to publish a confidence. `ΔF` is never accumulated, so
      the caller has no evidence count to put in one, and a publisher-chosen weight on the
      publisher's own testimony is exactly what a reader must not accept on assertion.
