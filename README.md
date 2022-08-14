# autoparallel

## Code Structure

### infrastructure.py
This package defines the hardware topology of the infrastructure
Functions:
- profile_hardware(type_of_device,**args): depending on the type of device, profile the corresponding information of the hardware
- - CPU
- - - computation_power: cores * frequency * 32FLOPS/cycle
- - - - cores: psutils.cpu_count
- - - - frequency: psutils.cpu_frequency
- - - memory_size: psutils.virtual_memory
- - - memory_bandwidth: (not known)
- - GPU
- - - computation_power: cores * frequency * 2FLOPS/cycle
- - - - https://github.com/ekondis/gpuroofperf-toolkit
Classes:
- Node: create the node for each hardware
- - aspects:
- - - node_id(parameter): the identifier of the node
- - - computation_power(parameter): profiled computation_power
- - - memory_size(parameter): profiled memory_size
- - - task_queue: the queue for the tasks waiting for execution
- - - channel_num: (not known)
- - - total_bandwidth: (not known)
- - functions:
- - - execute(): (not implemented)
- Link: create the link between the nodes
- - aspects:
- - - latency(parameter): the latency of the link
- - - bandwidth(parameter): the bandwidth of the link
- Infrastructure:
- - aspects:
- - - num_nodes: the number of nodes included in the infrastructure
- - - nodes(parameter): the nodes included in the infrastructure
- - - links: the two-dimension matrix indicate the links included in the infrastructure
- - functions:
- - - add_link: add the corresponding link to the infrastructure

### task.py
This package defines the task processed in the infrastructure
Classes:
 - LayerPacket: wrap each layer with the input size and corresponding profiling result
 - - aspects:
 - - - layer(parameter): the corresponding layer for this task