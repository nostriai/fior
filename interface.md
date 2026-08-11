Interface: FederatedInferenceOverRelays

All methods run client-side. No method contacts an aggregator, because there is not one:
composition and trust are computed locally from events any relay can serve. See
[protocol.md](protocol.md) for the wire format.

`signer` denotes a signing capability (NIP-07 handle, hardware signer, or an in-process
key). Private keys are never passed between components.

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
        - members - Peer contributions, each with its `Δη` and per-group `β`.
        - rho - Damping factor; defaults to 1.
    - Output: The local prior `η₀ + ρ Σ β Δη` per group, and the `ρ` actually applied.
    - Behavior: Sums trust-weighted site contributions into a prior. Excludes the caller's
      own site. Backtracks `ρ ← ρ/2` until the result lies within the natural parameter
      domain of each group's family, and reports the final `ρ`.

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
        - members - Peer contributions with their current `β`.
    - Output: `ΔF` per peer per parameter group.
    - Behavior: Evaluates, in closed form via Bayesian model reduction, how each peer's
      contribution changes the local log evidence:
      `ΔF = A(η_q) + A(η_p − βΔη) − A(η_q − βΔη) − A(η_p)`.
      `ΔF > 0` means the peer is earning its weight. Requires no holdout split and no
      retraining. Evaluates `A()` in f64 regardless of the encoding a site arrived in.

* Method: resolve_beta
    - Input:
        - peer - Peer public key.
        - model_id - Model identifier.
        - group - Parameter group name.
        - delta_f - The peer's `ΔF` for this group, from `score_peers`.
        - attestation_prior - Beta parameters `(a, b)`, resolved from the local
          attestation store; defaults to `(1, 1)`.
    - Output: `β ∈ (0,1)`.
    - Behavior: Returns the posterior inclusion probability
      `β = 1 / (1 + (b/a)·exp(−ΔF))`. `ΔF` is recomputed each round and never stored;
      only `(a, b)` accumulates, and only from information the client cannot recompute
      for itself.

* Method: resolve_attestation_prior
    - Input:
        - peer - Peer public key.
        - model_id - Model identifier; optional.
        - group - Parameter group name; optional.
        - gamma - Pooling factor across hierarchy levels.
    - Output: Beta parameters `(a, b)`.
    - Behavior: Resolves the hierarchical prior over the peer's trust weight, pooling
      `peer`, `peer:model`, and `peer:model:group` levels with discount `gamma`, and
      folding in transitive attestations from peers the caller already trusts, discounted
      by that trust and clipped per contributor. Local policy — no part of this is
      normative.

## Publication

* Method: publish_site
    - Input:
        - signer - Signing capability.
        - model_id - Model identifier.
        - delta_eta - The site contribution from `compute_site`.
        - encoding - Numeric encoding; `f64le`, `f32le`, `i16le`, or `i8`.
        - cavity_ref - Coordinate of the composite trained against; optional.
        - blossom_servers - Servers to upload to.
        - relays - Relays to publish to.
    - Output: A success status and the published event id.
    - Behavior: Uploads the `Δη` blob **before** publishing the kind 30101 that references
      it — the event is durable the moment a relay accepts it, and the blob is not.
      Quantizing encodings must use stochastic rounding; deterministic rounding biases
      every consumer's composed prior. Supersedes the caller's previous site for this
      model.

* Method: withdraw_site
    - Input:
        - signer - Signing capability.
        - model_id - Model identifier.
        - relays - Relays to publish to.
    - Output: A success status.
    - Behavior: Publishes a NIP-09 deletion for the caller's site. No registry is updated
      and no peer needs to act; each peer's next composition omits the caller.

* Method: publish_composite
    - Input:
        - signer - Signing capability.
        - model_id - Model identifier.
        - members - Pinned site event ids with the per-group `β` applied to each.
        - rho - The damping actually applied.
        - eta - The resulting parameters.
        - purpose - `trust-weighted`, `uniform`, `ablation`, or free text.
        - blossom_servers, relays
    - Output: A success status and the published event id.
    - Behavior: Publishes a reproducible record of one weighted combination. Members are
      pinned by event id, not by coordinate, because sites are replaceable and a
      benchmark against a moving target means nothing. A composite carries no
      data-dependent term and is never summed as a site.

* Method: publish_benchmark
    - Input:
        - signer - Signing capability.
        - descriptor_ref - Coordinate of the benchmark descriptor measured against.
        - target_ref - Coordinate and event id of the site or composite measured.
        - metrics - Name, value, and standard error per metric.
        - samples - Evaluation sample count.
        - relays
    - Output: A success status and the published event id.
    - Behavior: Publishes a measurement. Metric names must be ones the descriptor
      declares. Results are comparable only within a descriptor — scores measured against
      different evaluation sets are not on a common scale and must not be pooled.

* Method: publish_attestation
    - Input:
        - signer - Signing capability.
        - peer - Peer public key.
        - beta_params - Beta parameters `(a, b)`.
        - model_id - Model identifier; omit for a peer-level attestation.
        - group - Parameter group; omit for a model-level attestation.
        - relays
    - Output: A success status.
    - Behavior: Publishes a belief about a peer's trust weight. Entirely optional: β is
      always computed and held locally, and a node that publishes no attestations and
      reads none is a fully functional participant. A binary like is `(2, 1)`; a dislike
      is `(1, 2)`.
