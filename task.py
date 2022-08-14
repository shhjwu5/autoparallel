import profile
from urllib.parse import _NetlocResultMixinBytes

import torch
import torch.nn as nn
import torch.distributed as dist

class LayerPacket:
    def __init__(self, layer, size):
        self.layer = layer
        self.size = size
        self.feature = []

    def get_feature(self):
        self.feature = profile.generate_features(self)

class Task:
    def __init__(self, task_id, layer_packets, batch_num,
                 task_from=[],task_to=[],
                 ):
        self.task_id = task_id
        self.layers = [layer_packet.layer for layer_packet in layer_packets]
        self.batch_num = batch_num
        self.input_size = layer_packets[0].size
        self.task_size = 0 #?所有layer_packet占用的总空间
        self.estimate_time = 0 #? 所有layer_packet耗时的和

        self.task_to = [task_to] if type(task_to)==int else task_to
        self.task_from = [task_from] if type(task_from)==int else task_from

        self.device = None
        self.device_to = []
        self.device_from = []

    def set_device(self,device_type,device_num):
        if device_type=="from":
            self.device_from.append(device_num)
        elif device_type=="to":
            self.device_to.append(device_num)
        else:
            self.device = device_num
        
    def execute(self,data):
        if len(self.device_from)==0:
            #self.input = self._get_batch_data()
            self.input = data.to(torch.device("cuda:%d"%(self.device)))
        else:
            self.input = torch.zeros(self.input_size).to(torch.device("cuda:%d"%(self.device)))
            recv_input = dist.irecv(self.input,src=self.device_from[0])
            recv_input.wait()
        
        self.output = self.input
        for layer in self.layers:
            self.output = layer(self.output)

        if len(self.device_to)==0:
            print(self.output)
        else:
            dist.isend(self.output,dst=self.device_to[0])

    def _get_batch_data(self):
        pass

    def visualize(self):
        print("Task id:",self.task_id,"\t Input size:",self.input_size,
              "\t Device from",self.device_from,"\t Device",self.device,"\t Device to:",self.device_to)

class TaskManager:
    def __init__(self):
        self.num_tasks = 0
        self.tasks = {}

    def add_task(self,task):
        self.tasks[task.task_id] = task