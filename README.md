# Federated inference over relays (Fior)

## Introduction

A specification for a multi-backend interface for Federated Inference Over Relays: a distributed system that carries out probabilistic inference (learning) and decision making across multiple nodes while keeping data in place and removing the need for data movement. What participants exchange is a parameter update rather than raw records; how much that conceals depends on the model, and for conjugate families it is less than the phrase "data privacy" suggests. See *What a published site discloses* in [protocol.md](protocol.md). The framework is based on recent results in probabilistic federated learning derived using variational inference.

There is no aggregator and no coordinator. Every participant computes its own view of the global posterior locally, from whichever peers it chooses to weight and however it chooses to weight them.

## Documents

| Document | Contents |
|----------|----------|
| [protocol.md](protocol.md) | Wire protocol: event kinds, encoding, composition rule, trust layer |
| [interface.md](interface.md) | Client-side API surface |
| [architecture.md](architecture.md) | System overview, implementation status, and what the Nostr layer must implement |
| [ui-integration.md](ui-integration.md) | Mapping protocol events to marketplace UI state |
| [distributions.md](distributions.md) | Supported exponential families and their log-partition functions |
| [test/fior_sim.py](test/fior_sim.py) | **The reference implementation** of composition and trust — no Nostr, no Blossom. Source of every quantitative claim in `protocol.md`, and of fourteen numbered findings including the negative ones |

Nothing yet implements the wire protocol; see `architecture.md` §2.

## Functional Requirements

* Inference and decision making are performed locally, on the client side.
* Clients exchange model inference results as parameters of posterior distributions over model parameters.
* Prior and posterior distributions are specified in terms of the natural parameters of an [exponential family](distributions.md).
* A node publishes its **site contribution** — the natural parameter difference `Δη` between its local posterior and the prior it trained against — not its full posterior. Sites compose; posteriors do not.
* Each client builds its own prior by weighted summation over the peers it chooses to include:

  ```
  η_A^prior = η₀ + Σ_{n ≠ A} p_{A→n} · Δη_n
  ```

* The inclusion probability `p_{A→n} \in [0, 1]` is the weight client A gives peer n's contribution — the posterior probability that n belongs in A's prior pool. It is computed locally and privately. How a client computes it is its own policy; the protocol specifies what `p` means and how attestations about it are exchanged.
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

Aggregating and indexing are things a client may choose to do, not roles the protocol depends on.

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
    - Main Success Scenario: The node fetches the sites it judges worth the bandwidth, scores each peer via Bayesian model reduction, sets `p` (the posterior inclusion probability) per peer per parameter group, and recomposes its prior.

* Use Case 4: Attesting
  - Actors: Node
    - Goal: Share a trust judgement about a peer
    - Precondition: The node has evidence about that peer
    - Main Success Scenario: The node publishes its inclusion probability `p` for the peer as a single scalar, optionally scoped to a model and parameter group. Publication is voluntary; peers may use it to construct a prior, and their own direct evidence supersedes it.

## Testing and Acceptance Criteria

* The system is tested with different numbers of nodes, verifying correct operation as the count varies.
* Probabilistic inference tasks complete successfully and results are consistent with expectations.
* Nodes can join and leave without causing disruption to any other participant.
* A node that composes with all `p = 1` recovers unweighted summation.
* A node publishing deliberately poisoned parameters is suppressed by peers' trust weighting in the round it publishes, without any coordinated action.

## Glossary

* **Pyro/NumPyro**: probabilistic programming language frameworks.
* **Inference**: the process of inverting a probabilistic model and forming posterior beliefs over latent random variables.
* **Node**: a participant in the federated network.
* **Site contribution (`Δη`)**: a node's likelihood approximation — the natural parameter difference between its local posterior and the prior it trained against. This is what composes.
* **Cavity**: the prior a node trained against, formed from other participants' sites excluding its own.
* **Inclusion probability (`p`)**: the *posterior* probability `P(z=1 | my data)` that a peer's contribution enters the client's prior pool. This is the weight in the composition. Distinct from `β`.
* **Prior inclusion probability (`β`)**: `a/(a+b)`, the mean of the Beta prior over that inclusion, set by attestations alone and equal to `0.5` for a stranger. It is what the client believes *before* looking at its own data; `p` is what it believes after.
* **Bayesian model reduction (BMR)**: a closed-form log Bayes factor between two nested models, computable from the log-partition function without retraining. Here the two models are *peer n included at full weight* and *peer n absent*, both evaluated against the same reference containing every other peer at its current `p`. It is a comparison of two hypotheses, not a measurement of what removing a peer from the working prior would do.
* **Attestation**: a voluntary publication of one client's `p` for a peer. It carries no confidence: how heavily to weigh it is the reader's choice, not the publisher's. A reader folds attestations into its own `(a, b)`, so one client's posterior becomes an input to another's prior.

## References

* [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach](https://arxiv.org/abs/2302.04228) — Han Guo, Philip Greengard, Hongyi Wang, Andrew Gelman, Yoon Kim, Eric P. Xing
* [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning](https://arxiv.org/abs/2202.12275) — Matthew Ashman, Thang D. Bui, Cuong V. Nguyen, Stratis Markou, Adrian Weller, Siddharth Swaroop, Richard E. Turner
* [Personalized Federated Learning via Variational Bayesian Inference](https://proceedings.mlr.press/v162/zhang22o.html) — Xu Zhang, Yinchuan Li, Wenpeng Li, Kaiyang Guo, Yunfeng Shao
* [Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/abs/1811.11206) — Thang D. Bui, Cuong V. Nguyen, Siddharth Swaroop, Richard E. Turner
* [Bayesian model reduction](https://arxiv.org/abs/1805.07092) — Karl Friston, Thomas Parr, Peter Zeidman
