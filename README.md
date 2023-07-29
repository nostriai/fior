# Federated inference over relays (Fior)

## Introduction
The specification for a multi-backend interface for Federated inference over relays. A distributed system designed to carry out probabilistic inference (learning) and decision making tasks using machine learning models across multiple nodes while preserving data privacy and removing the need for data movement. The inspiration for the framework is based on the recent results on probabilistic federated learning derived using variational inference. 

## Functional Requirements
* Machine learning models specified in ONNX format.
* The inference and decision making tasks are performed localy on a clinet side.
* Clients exchange model inference results in terms of parameters of a posterior distributions over model parameters.
* Prior and posterior distributions are specified in the terms of natural parameters of an exponential family
* The structure and the specification of the prior and approximate posterior distribution are formulated based on [Pyro/Numpyro](pyro.ai) model [format](variational_models.md).
* Clients do not transmit raw data from its original node.
* An interface for nodes to join or leave the federated network is specified in the fior [interface](interface.md).

## Non-functional Requirements
* Clients must preserve data privacy by ensuring that raw data does not leave its original node.
* Federation should scale horizontally, allowing more nodes to join the federation without degrading inference results.

## Interfaces
* The software shall provide a REST API for nodes to join or leave the federated network.
* The software shall also provide an API to submit machine learning models and inference tasks.

## Use Cases
* Use Case 1: Node Registration

Actors: Node
- Goal: Join the federated network
- Precondition: The node has the Fior client installed
- Main Success Scenario: Node sends a request to Fior, Fior adds the node to its registry.
- Use Case 2: Submit Inference Task

Actors: User
- Goal: Submit an inference task
- Preconditions: A valid ONNX model and inference task is available
- Main Success Scenario: User submits the model and task to Fior, Fior validates the input, distributes the task, collects results, and returns the final result to the user.

## Data Specifications
* The clients join the federation via their npubs. 
* Fior will need to manage the registry of nodes.
* It will also need to store metadata about tasks, such as the model used, the status of the task, and the last result.

## Error Handling
* If a submitted model or task is not valid, the system should return an error message.


## Testing and Acceptance Criteria
* The system is tested with different numbers of nodes, verifying correct operation with a varying number of nodes.
* Inference tasks are completed successfully and results are consistent with expectations.
* Nodes can join and leave the network without causing disruption.

## Glossary
* ONNX: Open Neural Network Exchange. An open format to represent deep learning models.
* Pyro/Numpyro: Probabilistic programing language frameworks
* Inference: The process of inverting a probabilistic model and forming posterior beliefs over latent random variables.
* Node: A participant in the federated network.

## References
* [Federated Learning as Variational Inference: A Scalable Expectation Propagation Approach
Han Guo, Philip Greengard, Hongyi Wang, Andrew Gelman, Yoon Kim, Eric P. Xing](https://arxiv.org/abs/2302.04228)
* [Partitioned Variational Inference: A Framework for Probabilistic Federated Learning
Matthew Ashman, Thang D. Bui, Cuong V. Nguyen, Stratis Markou, Adrian Weller, Siddharth Swaroop, Richard E. Turner](https://arxiv.org/abs/2202.12275)
* [Personalized Federated Learning via Variational Bayesian Inference
Xu Zhang, Yinchuan Li, Wenpeng Li, Kaiyang Guo, Yunfeng Shao](https://proceedings.mlr.press/v162/zhang22o.html)
* [Partitioned Variational Inference: A unified framework encompassing federated and continual learning
Thang D. Bui, Cuong V. Nguyen, Siddharth Swaroop, Richard E. Turner](https://arxiv.org/abs/1811.11206)
