import math
import numpy as np
from pickle import TRUE
from tkinter import FALSE
from numpy import random
from task import LayerPacket
from task import Task,TaskManager
from infrastructure import Node,Infrastructure

class Merge:
    def __init__(self,TaskManager,Task,LayerPacket,Node,Infrastructure):
        self.task_to=Task.task_to
        self.task_from=Task.task_from
        self.layer=Task.layer
        self.task_queue=Node.task_queue
        self.batch_num=Task.batch_num
        self.device_list=Infrastructure.nodes
        self.task_list=TaskManager.task_list
        self.num_list=TaskManager.num_list
    
    '''
    there need a strong connection between tasks,
    to ensure that the tasks can be merged
    '''
    def merge(self,coe):
        t_num=len(self.device_list)*coe
        if len(self.task_list)>t_num:
            mer=[]
            new_size=[]
            new_time=[]
            t=len(self.device_list)/t_num
            for i in range (t):
                for j in range(i*coe,(i+1)*coe):
                    temp=[]
                    temp_layer=0
                    if self.task_list[j].layer==temp_layer:
                        new_size[i]+=self.task_list[j].size
                        new_time[i]+=self.task_list[j].time
                        temp.append(j)
                    else:
                        mer.append(temp)
                        temp=[]
                        temp_layer=self.task_list[j].layer
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
    def __init__(self, parallel):
        self.parallel = parallel
        self.problem_size = self.parallel.getProblemSize()
        self.solution_range = self.parallel.getSolutionRange()

    def depend(self):
        t=len(self.parallel.task_list)
        task_depend=np.zeros((t,t),dtype=int)
        for i in range(t):
            
            if (self.task_list[i].task_from is not None):
                task_from=self.task_list[i].task_from
                task_depend[i][task_from]=1
            if (self.task_list[i].task_to is not None):
                task_to=self.task_list[i].task_to
                task_depend[i][task_to]=-1
        return task_depend

class Parallel:
    def __init__(self,Task,Infrastructure,TaskManager,Node):
        self.task_queue=Node.task_queue
        self.batch_num=Task.batch_num
        self.device_list=Infrastructure.nodes
        self.links=Infrastructure.links
        self.task_list=TaskManager.task_list
        self.time=Task.time

    def getProblemSize(self):
        return len(self.task_list)

    def getSolutionRange(self):
        return len(self.device_list)

    def checkSize(self, solution):
        device_capacity = [0 for i in range(len(self.device_list))]
        for k in range(len(solution)):
            device_capacity[solution[k]] += self.task_list[k]
        flag = True
        for k in range(len(device_capacity)):
            if device_capacity[k]>self.device_list[k].memory_size:
                flag = False
                break
        return flag

    def checkDevice(self,solution):
        for k in range(1,len(solution)):
            if solution[k]!= solution[k-1]:
                for item in range(len(self.links)):
                    if item.src==solution[k-1] and item.dst==solution[k]:
                        return TRUE
                    if item.dst==solution[k-1] and item.src==solution[k]:
                        return TRUE
        return FALSE

    def sumTime(self, solution):
        soluted=[0 for i in range(len(self.task_list))]
        cur_device_size = 0
        t = 0
        for k in range(len (solution)):
            if solution[k]== solution[k-1]:
                if(self.task_list[k].task_from is not None):
                    if t<soluted[self.task_list[k].task_from]:
                        t=soluted[self.task_list[k].task_from]+self.task_list[k].time[solution[k]]
                    else:
                        t+=self.task_list[k].time[solution[k]]
                cur_device_size += self.task_list[k].size
                
            else:
                latency,bandwidth=self.find_ltbw(solution[k-1],solution[k])
                if(self.task_list[k].task_from is not None):
                    if t<soluted[self.task_list[k].task_from]:
                        t=soluted[self.task_list[k].task_from]+self.task_list[k].time[solution[k]]+cur_device_size/bandwidth+latency
                    else:
                        t+=self.task_list[k].time[solution[k]]+cur_device_size/bandwidth+latency
                cur_device_size=self.task_list[k].size
        return t

    def find_ltbw(self,dvc_1,dvc_2):
        for link in self.links:
            if (link.src==dvc_1 and link.dst==dvc_2) or (link.dst==dvc_1 and link.src==dvc_2):
                return link.latency,link.bandwidth
        return math.inf,0
            


class Genetic:
    def __init__(self, parallel, max_iterations, ):
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
        self.task=TaskManager.task
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
        solution=Optimize.initialsize()
        task_depend=Topology.depend()
        time=Parallel.sumTime(solution)
        schdule={solution,task_depend,time}
        return schdule
