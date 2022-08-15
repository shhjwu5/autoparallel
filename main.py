import torch.distributed as dist
import torch.nn as nn
import torch

from task import Task, TaskManager,Task,LayerPacket
from infrastructure import Infrastructure,Node,Link
from optimize import Optimize

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

layers = [nn.Linear(in_features=3, out_features=2).to(torch.device("cuda:0")),
          nn.Linear(in_features=2, out_features=3).to(torch.device("cuda:1")),
          nn.Linear(in_features=3, out_features=1).to(torch.device("cuda:2"))]

sizes = [(1,3),(1,2),(1,3),(1,1)]

batches = [torch.Tensor([1, 2, 3]),
           torch.Tensor([1,2,3])]

task_manager = TaskManager()
for batch_num,_ in enumerate(batches):
    for layer_num,layer in enumerate(layers):
        layer_packet = LayerPacket(layer,sizes[layer_num])
        task_manager.add_task(Task((batch_num,layer_num),[layer_packet],batch_num))

nodes = [Node(torch.device("cuda:%d"%(i)),100,50) for i in range(3)]
infrastructure = Infrastructure(nodes)
for i in range(3):
    for j in range(i,3):
        infrastructure.add_link(i,j,Link(0.000009,32))

optimizer=Optimize()
solution=optimizer.initialsize()
schedule=optimizer.optimize()

schedule = [0,1,2]
for task_id,cur_task in task_manager.tasks.items():
    cur_task.set_device(device_type="cur",device_num=task_id[1])
    if task_id[1]-1>=0:
        cur_task.set_device(device_type="from",device_num=task_id[1]-1)
    if task_id[1]+1<=2:
        cur_task.set_device(device_type="to",device_num=task_id[1]+1)

    cur_task.visualize()

for task_id,cur_task in task_manager.tasks.items():
    if rank==task_id[1]:
        cur_task.execute(batches[task_id[0]])


