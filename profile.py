import csv
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchstat import stat

pattNum = r"[0-9]+(\.[0-9]+)?"
model_times = 10


class statBuffer(object):
    def __init__(self):
        self.buffer = []

    def write(self, *args):
        self.buffer.append(args)


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        # 扩充

    def forward(self, x):
        output = self.conv1(x)
        return output


class SimpleFCModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleFCModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x


def layer_to_model(layer):
    if isinstance(layer, nn.Linear):
        return SimpleFCModel(layer.in_features, layer.out_features)
    elif isinstance(layer, nn.Conv2d):
        return ConvModel()  # Here, this class hasn't been well init yet


def generate_features(layerpacket):
    model = layer_to_model(layerpacket.layer)
    feature = []
    stdout = sys.stdout
    sys.stdout = statBuffer()
    if isinstance(model, ConvModel):
        stat(model, layerpacket.size)
    elif isinstance(model, SimpleFCModel):
        stat(model, layerpacket.size)  # 历史遗留问题，不确定要不要分开处理
    sys.stdout, data = stdout, sys.stdout
    info = data.buffer[0][0].split('\n')
    for line in info:
        if "Total" in line:
            match = re.search(pattNum, line)
            if match:
                if "MAdd" in line:
                    if "KMAdd" in line:
                        feature.append(eval(match.group()) / 1000)
                    elif "GMAdd" in line:
                        feature.append(eval(match.group()) * 1000)
                    elif "MMAdd" in line:
                        feature.append(eval(match.group()))
                    else:
                        feature.append(eval(match.group()) / 1000000)
                elif "Flops" in line:
                    if "KFlops" in line:
                        feature.append(eval(match.group()) / 1000)
                    elif "GFlops" in line:
                        feature.append(eval(match.group()) * 1000)
                    elif "MFlops" in line:
                        feature.append(eval(match.group()))
                    else:
                        feature.append(eval(match.group()) / 1000000)
                elif "B" in line:
                    if "KB" in line:
                        feature.append(eval(match.group()) / 1000)
                    elif "GB" in line:
                        feature.append(eval(match.group()) * 1000)
                    elif "MB" in line:
                        feature.append(eval(match.group()))
                    else:
                        feature.append(eval(match.group()) / 1000000)
    return feature


def profile_time_consumption(layerpacket):
    model = layer_to_model(layerpacket.layer)
    time = np.array([])
    for i in range(model_times):
        if isinstance(model, ConvModel):
            inputs = torch.randn(layerpacket.size)  # 这里好像要升一维
        elif isinstance(model, SimpleFCModel):
            inputs = torch.randn(layerpacket.size)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(inputs)
        with open("./profiler.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #  maybe in gPU, there will be different?
        if i == 0:
            continue
        with open("./profiler.txt", "r") as profiler:
            data = profiler.readlines()
            for line in data:
                if "Self CPU time total" in line:
                    match = re.search(pattNum, line)
                    if match:
                        if 'ms' in line:
                            time = np.append(time, eval(match.group()))
                        elif 'us' in line:
                            time = np.append(time, eval(match.group()) / 1000)
                        else:
                            time = np.append(time, eval(match.group()) * 1000)
    return time.mean()


def store_to_csv(layer):
    feature = generate_features(layer)
    feature.append(profile_time_consumption(layer))
    with open("./result_fc.csv", "a", newline='') as result:
        writer = csv.writer(result)
        writer.writerow(feature)
