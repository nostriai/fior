Interface: FederatedInferenceOverRelays

* Method: register_node
    - Input:
      - node_nsec - A private key of the node.
      - agg_npub - A public key of the aggregator.
    - Output: A success status and any relevant information.
    - Behavior: Registers a new node to the federation.
* Method: remove_node
    - Input:
        - node_nsec - A private key of the node that request removal.
        - agg_npub - A public key of the aggregator.
        - parameters - Parameter values to be removed from the federation. 
    - Output: A success status.
    - Behavior: Removes a node from the federation.
* Method: remove_task
    - Input:
        - node_nsec - A private key of the node that request removal.
        - task_npub - A public key of the task.
        - agg_npub - A public key of the aggregator.
        - parameters - Parameter values of the task to be removed from the federation. 
    - Output: A success status.
    - Behavior: Removes the local solution for a task from the federation.
* Method: submit_posterior
    - Input:
        - node_nsec - A private key of the node.
        - agg_npub - A public key of the aggregator.
        - task_npub - A public key of the task.
        - parameters - Parameters of the posterior distribution to be sent to the aggregator.
        - metadata - Additional metadata describing the solution.
    - Output: A success status.
    - Behavior: Submits local inference solutions to the federation.
* Method: read_prior
    - Input:
        - task_npub - A public key of the task.
        - agg_npub - A public key of the aggregator.
    - Output: Parameters of the global prior distribution.
    - Behavior: Searches for the last broadcasted parameters of the global prior.
