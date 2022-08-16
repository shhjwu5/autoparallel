import math
from re import I
import numpy as np
from pickle import TRUE
from tkinter import FALSE
from numpy import random
from task import Task,TaskManager,LayerPacket
from infrastructure import Node,Infrastructure

class Merge:
    def __init__(self,TaskManager,Task,LayerPacket,Node,Infrastructure):
        self.task_to=Task.task_to
        self.task_from=Task.task_from
        self.layer=Task.layer
        self.batch_num=Task.batch_num
        self.device_list=Infrastructure.nodes
        self.tasks=TaskManager.tasks
        self.num_list=TaskManager.num_list
    
    '''
    there need a strong connection between tasks,
    to ensure that the tasks can be merged
    '''
    def merge(self,coe):
        t_num=len(self.device_list)*coe
        if len(self.tasks)>t_num:
            mer=[]
            new_size=[]
            new_time=[]
            t=len(self.device_list)/t_num
            for i in range (t):
                for j in range(i*coe,(i+1)*coe):
                    temp=[]
                    temp_layer=0
                    if self.tasks[j].layer==temp_layer:
                        new_size[i]+=self.tasks[j].input_size
                        new_time[i]+=self.tasks[j].time
                        temp.append(j)
                    else:
                        mer.append(temp)
                        temp=[]
                        temp_layer=self.tasks[j].layer
                        temp.append(j)
                        j-=1
                        pass
        return new_size,new_time,mer

    def dismerge(self,mer,solution,new_size):
        real_solution=[0 for i in range(len(self.task_size))]
        for item in range(len(new_size)):
            for j in range(len(mer[item])):
                real_solution[mer[item][j]]=solution[item]
        return real_solution

class Topology:
    def __init__(self, parallel,Infrastructure,Merge):
        self.parallel = parallel
        self.problem_size = self.parallel.getProblemSize()
        self.solution_range = self.parallel.getSolutionRange()
        self.links=Infrastructure.links
        self.dev_num=len(Merge.device_list)

    def depend(self):
        t=len(self.parallel.tasks)
        task_depend=np.zeros((t,t),dtype=int)
        for i in range(t):
            if (type(self.tasks[i].task_from) ==int):
                task_from=self.tasks[i].task_from
                task_depend[i][task_from]=1
            if (type(self.tasks[i].task_to) ==int):
                task_to=self.tasks[i].task_to
                task_depend[i][task_to]=-1
        return task_depend

    def dev_top(self):
        dev_topo=self.links
        for i in range(self.dev_num):
            for k in range(self.dev_num):
                if dev_topo[i][k]==0:
                    dev_topo[i][k]=dev_topo[k][i]=[math.inf,0]
                elif type(dev_topo[i][k])==int:
                    dev_topo[i][k]=dev_topo[k][i]=[self.links[i][k].latency,self.links[i][k].bandwidth]
                else:
                    continue
        return dev_topo

class Parallel:
    def __init__(self,Task,Infrastructure,TaskManager,Node):
        self.task_queue=Node.task_queue
        self.batch_num=Task.batch_num
        self.device_list=Infrastructure.nodes
        self.links=Infrastructure.links
        self.tasks=TaskManager.tasks
        self.time=self.getTaskTime()
        self.device_size=self.getDeviceSize()
    
    def getDeviceSize(self):
        device_size=[0 for i in range(len(self.device_list))]
        for item in range(len(self.device_list)):
            device_size[item]=self.device_list[item].memory_size
        return device_size

    def getProblemSize(self):
        return len(self.tasks)

    def getSolutionRange(self):
        return len(self.device_list)

    def checkSize(self, solution):
        device_capacity = [0 for i in range(len(self.device_list))]
        for k in range(len(solution)):
            device_capacity[solution[k]] += self.tasks[k]
        flag = True
        for k in range(len(device_capacity)):
            if device_capacity[k]>self.device_list[k].memory_size:
                flag = False
                break
        return flag

    def checkDevice(self,solution):
        for k in range(1,len(solution)):
            if solution[k]!= solution[k-1]:
                if self.links[solution[k]][solution[k-1]] is None:
                    return FALSE            
        return TRUE

    def sumTime(self, solution):
        dev_topo=Topology.dev_top
        soluted=[0 for i in range(len(self.tasks))]
        cur_device_size = 0
        t = 0
        for k in range(len (solution)):
            if solution[k]== solution[k-1]:
                if(type(self.tasks[k].task_from) ==int):
                    if t<soluted[self.tasks[k].task_from]:
                        t=soluted[self.tasks[k].task_from]+self.tasks[k].time[solution[k]]
                    else:
                        t+=self.tasks[k].time[solution[k]]
                cur_device_size += self.tasks[k].size
                
            else:
                ltbw=dev_topo[solution[k-1][solution[k]]]
                latency=ltbw[0]
                bandwidth=ltbw[1]
                if(type(self.tasks[k].task_from) ==int):
                    if t<soluted[self.tasks[k].task_from]:
                        t=soluted[self.tasks[k].task_from]+self.tasks[k].time[solution[k]]+cur_device_size/bandwidth+latency
                    else:
                        t+=self.tasks[k].time[solution[k]]+cur_device_size/bandwidth+latency
                cur_device_size=self.tasks[k].size
        return t

class Genetic:
    def __init__(self, parallel, max_iterations):
        self.parallel = parallel
        self.problem_size = self.parallel.getProblemSize()
        self.solution_range = self.parallel.getSolutionRange()
        self.MaxIterations = max_iterations
        self.init_solution = self.getInitSolution()
        self.group = []

    def getInitSolution(self):
        solution = [0 for i in range(self.problem_size)]
        start = 0
        device = 0
        for i in range(self.problem_size):
            if sum(self.parallel.task_size[start:i+1]) > self.parallel.device_size[device]:
                device += 1
                start = i
            solution[i] = device

        return solution

    def getParents(self):
        father = self.group[0][1]
        mother = self.group[1][1]
        return father,mother

    def combine(self,father,mother):
        solution = []
        for i in range(self.problem_size):
            if random.random()<0.5:
                solution.append(father[i])
            else:
                solution.append(mother[i])
        return solution

    def mutate(self,solution):
        solution[random.randint(0,self.problem_size)] = random.randint(0,self.solution_range)
        return solution

    def run(self, verbose=False):
        init_time = self.parallel.sumTime(self.init_solution)
        self.group.append([init_time,self.init_solution])
        self.group.append([init_time,self.init_solution])
        for iter in range(self.MaxIterations):
            father,mother = self.getParents()
            child = self.combine(father,mother)
            solution = self.mutate(child)
            neighbor_sum_value = self.parallel.sumTime(solution)
            if self.parallel.checkDevice(solution):
                if self.parallel.checkSize(solution):
                    self.group.append([neighbor_sum_value,solution])
                    self.group.sort()

                    if len(self.group)>100:
                        self.group = self.group[:100]
            '''
            if verbose:
                print(iter, self.parallel.checkSize(self.group[0][1]), self.parallel.sumTime(self.group[0][0]))
            '''
        return iter + 1, self.group[0][0],self.group[0][1]


class Optimize:
    def __init__(self,TaskManager,Infrastructure):
        self.task=TaskManager.tasks
        self.Infrastructure=Infrastructure

    def initialsize(self):
        coe=5
        new_size,new_time,mer=Merge.merge(coe)
        parallel=Parallel(task_size=new_size,device_size=self.parallel.device_list,task_time=new_time)
        iter_num, min_value, best_solution = Genetic(parallel, max_iterations=10000)
        solution=Merge.dismerge(mer,best_solution,new_size)
        time=Parallel.sumTime(solution)
        return solution

    def optimize(self):
        dev_topo=Topology.dev_top()
        solution=Optimize.initialsize()
        task_depend=Topology.depend()
        time=Parallel.sumTime(solution)
        schdule={solution,[task_depend,dev_topo],time}
        return schdule
