def profile_hardware():
    pass

class Node:
    def __init__(self, node_id, computation_power, memory_size):
        self.node_id = node_id
        self.computation_power = computation_power
        self.memory_size = memory_size
        self.task_queue = []
        self.channel_num = 0 #? 干啥的
        self.total_bandwidth = 0 #? 干啥的

    def execute(self):
        pass #? 后面再搞

class Link:
    def __init__(self, latency, bandwidth):
        self.latency = latency
        self.bandwidth = bandwidth

class Infrastructure: #? nodes和links设置为啥不统一
    def __init__(self,nodes):
        self.num_nodes = len(nodes)
        self.nodes = nodes
        self.links = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

    def add_link(self, src, dst, link):
        self.links[src][dst] = self.links[dst][src] = link
