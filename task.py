import math
import profile


class LayerPacket:
    def __init__(self, layer, size):
        self.layer = layer
        self.size = size
        self.feature = []

    def get_feature(self):
        self.feature = profile.generate_features(self)


class Task:
    def __init__(self, layer,
                 batch_num,
                 task_from=None,
                 task_to=None,
                 ):
        """
        Here, all tasks are treated as if they are linearly relied on each other.
        So task_from and task_to should be traversed as soon as the task_list grows to a graph.
        All tasks are organized in a dict:
        {
            "task_num": task
        }
        and task_from/task_to only store the key "task_num"
        """

        if task_to is None:
            self.task_to = []
        else:
            self.task_to = [task_to]
        if task_from is None:
            self.task_from = []
        else:
            self.task_from = [task_from]
        self.layer = layer
        self.batch_num = batch_num
        self.result_before = math.inf  # what default value should it have?
        self.time = []

    def get_data_from_batch(self, data):
        if not self.task_from:
            self.result_before = data[self.batch_num]

    def compute(self):  # 如何确保上一个任务已经做完？
        return self.layer.layer(self.result_before)

    def pass_result(self, tasklist):
        if not self.task_to:
            print(self.compute())
        else:
            next_task = tasklist[self.task_to[0]]
            next_task.result_before = self.compute()

    def set_time(self, hardware_name, times):
        assert len(hardware_name) == len(times)
        for i in range(len(times)):
            self.time.append((hardware_name[i], times[i]))


class TaskManager:
    def __init__(self):
        self.task_list = {}
        self.num_list = []

    """
    Here, there maybe a relying relationship when adding tasks,
    but I don't know how is it described, so I have to ignore it.
    """
    def add_task(self, task):
        if len(self.num_list) == 0:
            self.num_list.append(len(self.task_list))
        num = self.num_list.pop()
        self.task_list[num] = task

    def end_task(self, num):
        self.num_list.append(num)
        #  not finished yet

    def grid_generation(self, layers, batches):
        for i in range(len(batches)):
            for layer in layers:
                task = Task(layer, batch_num=i)
                self.add_task(task)
