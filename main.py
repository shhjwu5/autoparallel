import task
import profile
import torch.nn as nn
import torch

"""
当前任务：
1、调研硬件的建模（标准化测试？），进而修改GBDT的代码来做prediction
2、完善TaskManager，写出layer+batch生成manager.tasklist
3、完善Infrastructure，用邻接表写拓扑结构（整个写一个infrastructure类？
4、看一个parallel算法
5、torch.distributed, 研究能否通信
Inputs are temporally a Tasklist
"""
Layers = [task.LayerPacket(layer=nn.Linear(in_features=3, out_features=2), size=(1, 1, 3)),
          task.LayerPacket(layer=nn.Linear(in_features=2, out_features=1), size=(1, 1, 2))]

Batches = [torch.Tensor([1, 2, 3])]

Tasklist = {
    0: task.Task(layer=Layers[0], task_to=1, batch_num=0),
    1: task.Task(layer=Layers[1], task_from=0, batch_num=0)
}
"""
layerpacket = Tasklist[0].layer
print(layerpacket)
print(profile.generate_features(layerpacket))
print(profile.profile_time_consumption(layerpacket))
"""


Tasklist[0].get_data_from_batch(Batches)
Tasklist[0].pass_result(Tasklist)
Tasklist[1].pass_result(Tasklist)
