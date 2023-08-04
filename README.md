# Federated inference over relays (Fior)

## Introduction
The specification for a multi-backend interface for Federated inference over relays. A distributed system designed to carry out probabilistic inference (learning) and decision making tasks using machine learning models across multiple nodes while preserving data privacy and removing the need for data movement. The inspiration for the framework is based on the recent results on probabilistic federated learning derived using variational inference. 

## Functional Requirements
* The inference and decision making tasks are performed localy on a clinet side.
* Clients exchange model inference results in terms of parameters of a posterior distributions over model parameters.
* Prior and posterior distributions are specified in the terms of natural parameters of an exponential family
* An interface for nodes to join, leave and interact with the federated network is specified in [interface](interface.md) doc.

## Non-functional Requirements
* Clients must preserve data privacy by ensuring that raw data does not leave its original node.
* Federation should scale horizontally, allowing more nodes to join the federation without degrading inference results.

## Interfaces
* Fior: an API for nodes to join or leave the federated network and exchange parameters over nostr protocol.

## Use Cases
* Use Case 1: Node Registration
  - Actors: Node
    - Goal: Join the federated network
    - Precondition: The node has a Fior client installed
    - Main Success Scenario: Node sends a request to an aggregator, aggregator adds the node to its registry, node recieves initial parameter values.

* Use Case 2: Node removal
  - Actors: Node
    - Goal: Leave the federated network
    - Preconditions: The node is a member of the federation
    - Main Success Scenario: Node sends a request to an aggregator, aggregator removes parameters corresponding to that node from the global solution and de-registers the node.

## Testing and Acceptance Criteria
* The system is tested with different numbers of nodes, verifying correct operation with a varying number of nodes.
* Probabilistic inference tasks are completed successfully and results are consistent with expectations.
* Nodes can join and leave the network without causing disruption.

## Glossary
* Pyro/Numpyro: Probabilistic programing language frameworks
* Inference: The process of inverting a probabilistic model and forming posterior beliefs over latent random variables.
* Node: A participant in the federated network.
* Aggregator: A node that recieves the solitions from members of the federation collects all solutions into a global solution and broadcasts the latest results to all members of the federation. 

## References
* [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach
Han Guo, Philip Greengard, Hongyi Wang, Andrew Gelman, Yoon Kim, Eric P. Xing](https://arxiv.org/abs/2302.04228)
* [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning
Matthew Ashman, Thang D. Bui, Cuong V. Nguyen, Stratis Markou, Adrian Weller, Siddharth Swaroop, Richard E. Turner](https://arxiv.org/abs/2202.12275)
* [Personalized Federated Learning via Variational Bayesian Inference
Xu Zhang, Yinchuan Li, Wenpeng Li, Kaiyang Guo, Yunfeng Shao](https://proceedings.mlr.press/v162/zhang22o.html)
* [Partitioned Variational Inference: A unified framework encompassing federated and continual learning
Thang D. Bui, Cuong V. Nguyen, Siddharth Swaroop, Richard E. Turner](https://arxiv.org/abs/1811.11206)
