Interface: FederatedInferenceOverRelays

* Method: register_node
    - Input: node_info - Information about the node. 
    - Output: A success status and any relevant information.
    - Behavior: Registers a new node to the federation.
* Method: remove_node
    - Input: node_id - The ID of the node to be removed.
    - Output: A success status.
    - Behavior: Removes a node from the federation.
* Method: submit_model
    - Input: model - A machine learning model in ONNX format.
    - Output: A success status and any relevant information.
    - Behavior: Submits a machine learning model to the federation.
* Method: submit_task
    - Input: task_info - Information about the task.
    - Output: A task ID for tracking purposes.
    - Behavior: Submits an inference task to the federation.
* Method: get_task_status
    - Input: task_id - The ID of the task.
    - Output: Information about the status of the task.
    - Behavior: Returns the current status of the specified task.
* Method: get_task_result
    - Input: task_id - The ID of the task.
    - Output: The result of the task.
    - Behavior: Returns the result of the specified task.