def profile_hardware():
    """
    Params needed:
    1. Type of device
    2. if CPU:
    (1) computation power
        number of cores(psutils.cpu_count)
        basic frequency(psutils.cpu_frequency)
        psutil库可以获得以上全部内容
        说明：对于SIMD的计算，根据CPU支持的指令集不同，会具有不同的并行能力
        峰值算力 = cores * frequency * FMA(generally 2) * 2(指令集AVX-512) * 512 / (32:单精度、64:双精度)
        可以把第三项及后面的简记为32Flops/cycle
        （这里也许还和取指周期有关系？但是有不同的说法，可能要测试一下，有的说这个取指周期就是上面的FMA系数）
        测试当前极限算力：https://github.com/pigirons/cpufp
    (2) memory size
    psutils.virtual_memory
    (3) memory_bandwidth
    psutils.disk_io_counters?（还是不知道最大能力怎么算）
    3. if GPU:
        同上，算力 = cores * frequency * 2FLops/cycle
        maybe torch.cuda.get_device_capability()
        三个感兴趣的参数都可以通过这个得到
        https://github.com/ekondis/gpuroofperf-toolkit


    """


class Node:
    def __init__(self, cp, ms):
        self.computation_power = cp
        self.memory_size = ms
        self.task_queue = []
        self.channel_num = 0
        self.total_bandwidth = 0

    def execute(self):
        return self.task_queue[0]  # learn torch.distributed first

    """
    cuda.memory_usage?
    cuda.utilization?
    也许不用slurm那种东西，用这些信息就足够fine-tune了么
    """


class Link:
    def __init__(self, bw, lt):
        self.bandwidth = bw
        self.latency = lt


class Infrastructure:
    def __init__(self, nodes):
        self.nodes = nodes
        self.links = []

    def add_link(self, src, dst, link):
        self.links.append(link)
