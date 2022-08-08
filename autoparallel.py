import math
from collections import deque
from numpy import random
import itertools
import numpy as np
from time import *
import matplotlib.pyplot as plt

class Parallel:
    def __init__(self,task_size,device_size,task_time):
        self.task_size = task_size
        self.device_size = device_size
        self.task_time = task_time

    def getProblemSize(self):
        return len(self.task_size)

    def getSolutionRange(self):
        return len(self.device_size)

    def checkSize(self, solution):
        device_capacity = [0 for i in range(len(self.device_size))]
        for k in range(len(solution)):
            device_capacity[solution[k]] += self.task_size[k]
        flag = True
        for k in range(len(device_capacity)):
            if device_capacity[k]>self.device_size[k]:
                flag = False
                break
        return flag

    def sumTime(self, solution):
        t = 0
        latency=0.000009
        bandwidth =32
        for k in range(len(solution)):
            t += self.task_time[k][solution[k]]
        cur_device_size = self.task_size[0]
        for k in range(1,len(solution)):
            if solution[k]!= solution[k-1]:
                t+=latency + cur_device_size / bandwidth
                cur_device_size =self.task_size[k]
            else:
                cur_device_size+=self.task_size[k]
        return t


# 暴力搜索
class BruteSearch:
    def __init__(self, parallel):
        self.parallel = parallel
        self.problem_size = self.parallel.getProblemSize()
        self.solution_range = self.parallel.getSolutionRange()

    def run(self, verbose=False):
        solution = [0] * self.problem_size
        min_value = float("inf")
        iter_num = 0
        best_solution = None
        for i in range(self.solution_range**self.problem_size):
            iter_num += 1
            update = 1
            for index in range(len(solution)):
                if update==1:
                    solution[index] += 1
                    update = 0
                    if solution[index]>=self.solution_range:
                        solution[index] = 0
                        update = 1
                else:
                    break
            if self.parallel.checkSize(solution):
                value = self.parallel.sumTime(solution)
                if value < min_value:
                    min_value = value
                    best_solution = solution.copy()
            if verbose:
                print(iter_num, min_value)

        return iter_num, min_value, best_solution

# 模拟退火
class SimulatedAnnealing:
    def __init__(self, parallel, max_iterations, temp_max, temp_min, cold_ratio, neighbor_search_num):
        self.parallel = parallel
        self.problem_size = self.parallel.getProblemSize()
        self.solution_range = self.parallel.getSolutionRange()
        self.MaxIterations = max_iterations
        self.temp = temp_max
        self.temp_min = temp_min
        self.cold_ratio = cold_ratio
        self.neighbor_search_num = neighbor_search_num
        self.init_solution = self.getInitSolution()

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

    def getAllNeighbouringSolutions(self, cur_solution):
        all_neighbor_solutions = []

        for search_num in range(self.neighbor_search_num):
            neighbor_solution = cur_solution.copy()
            neighbor_solution[search_num] = random.randint(0,self.solution_range)
            if self.parallel.checkSize(neighbor_solution):
                all_neighbor_solutions.append(neighbor_solution)
        return all_neighbor_solutions

    def getMonteCarlo(self, abs_delta_E):
        return math.exp(-1 * abs_delta_E / self.temp)

    def coldDownTemperature(self):
        self.temp *= self.cold_ratio

    def run(self, verbose=False):
        cur_solution = self.init_solution
        for iter in range(self.MaxIterations):
            cur_sum_value = self.parallel.sumTime(cur_solution)
            all_neighbor_solutions = self.getAllNeighbouringSolutions(cur_solution)
            if len(all_neighbor_solutions)==0:
                continue
            neighbor_solution = all_neighbor_solutions[random.choice(len(all_neighbor_solutions))]
            neighbor_sum_value = self.parallel.sumTime(neighbor_solution)

            if neighbor_sum_value < cur_sum_value:
                cur_solution = neighbor_solution
            else:
                delta_E = cur_sum_value - neighbor_sum_value   # < 0
                accept_prob = self.getMonteCarlo(math.fabs(delta_E))
                if random.uniform(0, 1) < accept_prob:
                    cur_solution = neighbor_solution

            if verbose:
                print(iter, self.parallel.checkSize(cur_solution), self.parallel.sumTime(cur_solution))

            self.coldDownTemperature()
            if self.temp < self.temp_min:
                break

        return iter + 1, self.parallel.sumTime(cur_solution),cur_solution

# 遗传算法
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

            if self.parallel.checkSize(solution):
                self.group.append([neighbor_sum_value,solution])
                self.group.sort()

                if len(self.group)>100:
                    self.group = self.group[:100]

            if verbose:
                print(iter, self.parallel.checkSize(self.group[0][1]), self.parallel.sumTime(self.group[0][0]))

        return iter + 1, self.group[0][0],self.group[0][1]

#粒子群
    class PSO:
        def __init__(self, parallel, c1,c2,W_max,W_min,V_max,V_min):
            self.parallel = parallel
            self.problem_size = self.parallel.getProblemSize()
            self.solution_range = self.parallel.getSolutionRange()

            self.c1=c1
            self.c2=c2
            self.W_max=W_max
            self.W_min=W_min
            self.V_max=V_max
            self.V_min=V_min
            
            self.init_solution = self.getInitSolution()
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
        def init_x(self):
            n=self.problem_size
            population = []
            for i in range(n):
                gene = []
                a = np.random.randint(0, 2)
                gene.append(a)
                population.append(gene)
            return population
        def init_v(self):
            n=self.problem_size
            v=[]
            for i in range (n):
                vi = []
                a = random.random() * (self.V_max - self.V_min) + self.V_min
                vi.append(a)
                v.append(vi)
            return v
        def fitness(self,x):
            w=self.parallel.task_size
            v=self.parallel.task_time
            n=self.problem_size
            fitvalue=[]
            fitweight=[]
            for i in range (n):
                a = 0  # 每个粒子的重量
                b = 0  # 每个粒子的价值(适应度)
                if x[i] == 1:
                    a += w
                    b += v
                if a > self.parallel.device_size:
                    b = 0
                fitvalue.append(b)
                fitweight.append(a)
            return fitvalue, fitweight
        def update_pbest(self,x,fitvalue,pbest,px):
            n=self.problem_size
            pb = pbest
            for i in range (n):
                if fitvalue[i] > pbest[i]:
                    pbest[i] = fitvalue[i]
                    px[i] = x[i]
            return pb, px

        def update_gbest (self, pbest, gbest,g,x):
            n=self.problem_size
            gb=gbest
            for i in range (n):
                if pbest[i] > gb:
                    gb = pbest[i]
                    g = x[i]
            return gb, g

        def update_v(self,v,x,pbest,g):
            vmin=self.parallel.V_min
            vmax=self.parallel.V_max
            c1=self.parallel.c1
            c2=self.parallel.c2
            n=self.problem_size

            for i in range(n):
                a=random.random()
                b=random.random()
                v[i][j]=v[i][j]+c1*a*(pbest[i][j]-x[i][j])+c2*b*(g[j]-x[i][j])
                if v[i][j]<vmin:
                    v[i][j]=vmin
                if v[i][j]>vmax:
                    v[i][j]=vmax
            return v

        def update_x(self,x,v):
            n=self.problem_size
            for i in range(n):
                a=random.random()
                x[i][j]=1/(1+math.exp(-v[i][j]))
                if x[i][j]>a:
                    x[i][j]=1
                else:
                    x[i][j]=0
            return x

        def run(self, verbose=False):
            item=[]
            itemg=[]
            init_time = self.parallel.sumTime(self.init_solution)
            for i in range(100):
                fv, fw = self.fitness(x)
                pb, px = self.update_pbest(x, fv, pb, px)
                gb, g = self.update_gbest(x, pb, gb, g)
                item.append(gb)
                itemg.append(g)
                v = self.update_v(v, x, px, g)
                x = self.update_x(x, v)
            if verbose:
                print(iter, self.parallel.checkSize(itemg[0]), self.parallel.sumTime(itemg[0]))
            time_pso=self.parallel.sumTime(item)
            return item,time_pso
            
	



def main(task_size,device_size,task_time,seed=42):
    print(f'Task Size: {task_size}')
    print(f'Device Size: {device_size}')
    print(f'Task Time: {task_time}')
    print()

    parallel = Parallel(task_size=task_size,device_size=device_size,task_time=task_time)
    methods = ['SimulatedAnnealing','Genetic']#, 'TabuSearch']

    random.seed(seed)
    
    for method in methods:
        if method == 'brute':
            runner = BruteSearch(parallel)
        elif method == 'SimulatedAnnealing':
            runner = SimulatedAnnealing(parallel, max_iterations=10000, temp_max=500, temp_min=0.1 ** 10,
                                        cold_ratio=0.999, neighbor_search_num=4)
        elif method == 'Genetic':
            runner = Genetic(parallel, max_iterations=10000)
            
        iter_num, min_value, best_solution = runner.run(verbose=False)

        print(f"Mode: {method:20s}  run for {iter_num:6d} iterations, got the min value {min_value:4f}, best solution is {best_solution}")


if __name__ == "__main__":
    task_num = 10
    device_num = 5
    task_size = [random.randint(1,10) for i in range(task_num)]
    device_size = [random.randint(5,20) for i in range(device_num)]
    task_time = [
                    [random.randint(1,5) for i in range(device_num)] for j in range(task_num)
                ]
    main(task_size=task_size,device_size=device_size,task_time=task_time,seed=42)
