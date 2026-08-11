# Federated inference over relays (Fior)

## Introduction

A specification for a multi-backend interface for Federated Inference Over Relays: a distributed system that carries out probabilistic inference (learning) and decision making across multiple nodes while preserving data privacy and removing the need for data movement. The framework is based on recent results in probabilistic federated learning derived using variational inference.

There is no aggregator and no coordinator. Every participant computes its own view of the global posterior locally, from whichever peers it chooses to weight and however it chooses to weight them.

## Documents

| Document | Contents |
|----------|----------|
| [protocol.md](protocol.md) | Wire protocol: event kinds, encoding, composition rule, trust layer |
| [interface.md](interface.md) | Client-side API surface |
| [architecture.md](architecture.md) | System overview, implementation status, v3 migration work |
| [ui-integration.md](ui-integration.md) | Mapping protocol events to marketplace UI state |
| [distributions.md](distributions.md) | Supported exponential families and their log-partition functions |

## Functional Requirements

* Inference and decision making are performed locally, on the client side.
* Clients exchange model inference results as parameters of posterior distributions over model parameters.
* Prior and posterior distributions are specified in terms of the natural parameters of an [exponential family](distributions.md).
* A node publishes its **site contribution** — the natural parameter difference `Δη` between its local posterior and the prior it trained against — not its full posterior. Sites compose; posteriors do not.
* Each client builds its own prior by weighted summation over the peers it chooses to include:

  ```
  η_A^prior = η₀ + ρ · Σ_{n ≠ A} β_{A→n} · Δη_n
  ```

* The trust weight `β_{A→n}` is the precision client A assigns to peer n's contribution. It is computed locally and privately. How a client computes it is its own policy; the protocol specifies what β means and how attestations about it are exchanged.
* An interface for nodes to participate in the federated network is specified in the [interface](interface.md) doc. The wire protocol is specified in [protocol.md](protocol.md).

## Non-functional Requirements

* Clients must preserve data privacy by ensuring that raw data does not leave its original node.
* Federation should scale horizontally, allowing more nodes to join without degrading inference results.
* No participant may hold state that another participant must trust in order to make progress.
* A node that ignores the social layer entirely — publishing no trust attestations and reading none — must remain a fully functional participant.

## Roles

The protocol recognises exactly two roles.

* **Relay** — a Nostr relay that carries FIOR events, advertises support via NIP-11, and enforces size and rate policy. It never computes on or alters parameters, and never holds them: parameter payloads live in Blossom blobs and events carry only their hashes.
* **Client** — everything else. Local inference, trust evaluation, prior composition, and publication.

Aggregating, benchmarking, and indexing are things a client may choose to do, not roles the protocol depends on.

## Interfaces

* Fior: an API for nodes to participate in the federated network and exchange natural parameters over the Nostr protocol.

## Use Cases

* Use Case 1: Joining
  - Actors: Node
    - Goal: Contribute to a federated model
    - Precondition: The node has a Fior client installed and can reach a relay carrying the model
    - Main Success Scenario: The node fetches the model card, composes its prior from the peers it chooses to include, trains locally, uploads its `Δη` blob, and publishes a site contribution. Publishing the site *is* joining; no registry is updated and no other participant needs to act.

* Use Case 2: Leaving
  - Actors: Node
    - Goal: Withdraw from a federated model
    - Precondition: The node has published a site contribution
    - Main Success Scenario: The node publishes a NIP-09 deletion for its site, or lets the site expire. Each peer's next composition simply omits it. There is no registry to disagree about who is a member.

* Use Case 3: Ingesting peers
  - Actors: Node
    - Goal: Improve local inference using other nodes' contributions
    - Precondition: One or more peers have published sites for the model
    - Main Success Scenario: The node fetches the sites it judges worth the bandwidth, scores each peer via Bayesian model reduction, sets `β` per peer per parameter group, and recomposes its prior.

* Use Case 4: Benchmarking
  - Actors: Node
    - Goal: Publish a measurement others can compare against
    - Precondition: A benchmark descriptor exists defining a comparable evaluation
    - Main Success Scenario: The node publishes a composite — its own combination, a uniform-weight baseline, or a leave-one-out ablation — measures it against the descriptor, and publishes the result. Anyone can refetch the pinned sites and reproduce the arithmetic.

* Use Case 5: Attesting
  - Actors: Node
    - Goal: Share a trust judgement about a peer
    - Precondition: The node has evidence about that peer
    - Main Success Scenario: The node publishes Beta parameters `(a, b)` for the peer, optionally scoped to a model and parameter group. Publication is voluntary; peers may use it as a prior, and their own direct evidence supersedes it.

## Testing and Acceptance Criteria

* The system is tested with different numbers of nodes, verifying correct operation as the count varies.
* Probabilistic inference tasks complete successfully and results are consistent with expectations.
* Nodes can join and leave without causing disruption to any other participant.
* A node that composes with all `β = 1` recovers unweighted summation, matching the v2 aggregation result.
* A node publishing deliberately poisoned parameters is suppressed by peers' trust weighting in the round it publishes, without any coordinated action.

## Glossary

* **Pyro/NumPyro**: probabilistic programming language frameworks.
* **Inference**: the process of inverting a probabilistic model and forming posterior beliefs over latent random variables.
* **Node**: a participant in the federated network.
* **Site contribution (`Δη`)**: a node's likelihood approximation — the natural parameter difference between its local posterior and the prior it trained against. This is what composes.
* **Cavity**: the prior a node trained against, formed from other participants' sites excluding its own.
* **Composite**: a specific weighted combination of sites, recorded so it can be reproduced and benchmarked. Carries no data-dependent term and is never summed as a site.
* **Trust weight (`β`)**: the precision one client assigns to another's contribution. Scaling natural parameters by β leaves the implied mean unchanged and inflates the implied variance, so β adjusts how much you believe a peer, not what you think they said.
* **Bayesian model reduction (BMR)**: a closed-form evaluation of how a model's log evidence changes when a component is removed, computable from the log-partition function without retraining. Used to score peers.
* **Attestation**: a voluntary publication of Beta parameters expressing a belief about a peer's trust weight.

## References

* [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach](https://arxiv.org/abs/2302.04228) — Han Guo, Philip Greengard, Hongyi Wang, Andrew Gelman, Yoon Kim, Eric P. Xing
* [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning](https://arxiv.org/abs/2202.12275) — Matthew Ashman, Thang D. Bui, Cuong V. Nguyen, Stratis Markou, Adrian Weller, Siddharth Swaroop, Richard E. Turner
* [Personalized Federated Learning via Variational Bayesian Inference](https://proceedings.mlr.press/v162/zhang22o.html) — Xu Zhang, Yinchuan Li, Wenpeng Li, Kaiyang Guo, Yunfeng Shao
* [Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/abs/1811.11206) — Thang D. Bui, Cuong V. Nguyen, Siddharth Swaroop, Richard E. Turner
* [Bayesian model reduction](https://arxiv.org/abs/1805.07092) — Karl Friston, Thomas Parr, Peter Zeidman
