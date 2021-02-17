import fw_algorithms as fw
import ssc_algorithms as ssc

import numpy as np
from cvxopt import matrix, solvers
from scipy import sparse, linalg
from datetime import datetime
from time import process_time
import sys, re, glob
import concurrent.futures
import multiprocessing
import bottleneck

def threaded_multiple_arg_min(vector, s):
    count = multiprocessing.cpu_count()
    n = vector.size
    if (s == 0):
        return np.array([])
    if (s == 1):
        return np.argmin(vector)
    if (s > n):
        return np.argsort(vector)
    if (n < 1000):
        return np.argsort(vector)[:s]
    split_size = max(s * 100, n // 10 // count)# s*s*1000 ## ????
    while ((n % split_size) <= s*10):
        split_size += s * 10 // count  + 1
    l = list(range(0,vector.size,split_size))
    r_list = [ [] for x in l]
    t_count = min(count, (n - 1) // split_size + 1)
    #executor = concurrent.futures.ThreadPoolExecutor(count)
    #def _run_sort_thread(indexes, s):
    #    for i in indexes:
    #        x = split_size * i
    #        r_x = bottleneck.argpartition(vector[x:x+split_size], s)[:s] + x
    #        r_list[i] = list(r_x)
    #futures = {}
    for i in range(len(l)):
    #for i in range(1, t_count):
        #indexes = range(i,len(l),t_count)
        #args = (_run_sort_thread, indexes, s)
        #futures[executor.submit(*args)] = i
        x = split_size * i
        r_x = bottleneck.argpartition(vector[x:x+split_size], s)[:s] + x
        r_list[i] = list(r_x)
    #_run_sort_thread( range(0,len(l),t_count), s)
    #concurrent.futures.wait(futures)
    i_list = []
    for x in r_list:
        i_list += x
    indexes = np.array(i_list)
    #indexes = bottleneck.argpartition(vector, s)[:s]
    values = vector[indexes]
    new_index = np.argsort(values)[:s]
    return indexes[new_index]

class S_minimal_subset_descriptor():
    def __getitem__(self, key):
        if (type(key) == tuple):
            if (len(key) > 2):
                raise IndexError("Too many indexes: %d > 2"%len(key))
            k1, k2 = key
        else:
            k1, k2 = key, slice(None,None,None)
        if (type(k1) == int) or (type(k1) == np.int64):
            if (k1 >= self.vertex_counter):
                raise IndexError("Number of registered vertices %d is lower then index %d"%(
                    self.vertex_counter, k1
                ))
            k1 = [k1]
        if (type(k1) == slice):
            start, stop, step = k1.indices(self.vertex_counter)
            k1 = list(range(start, stop, step))
        r = np.zeros((len(k1), self.dim))
        i = 0
        for k in k1:
            #ones = np.array(self.y_vertices_list[k])
            ones = self.y_vertices_list[k]
            r[i, ones] = 1
            i += 1
        return r[:, k2].ravel()
    def __init__(self, dimension, S):
        self.dim = dimension
        self.S = S
        self.y_vertices_list = [[]]
        self.vertex_counter = 1
        self.be_verbose = False
    def __len__(self):
        return self.vertex_counter
    def __matmul__(self, gy):
        neg_y_direction, = np.where(gy <= 0)
        min_neg_dir = threaded_multiple_arg_min(gy[neg_y_direction], self.S)
        indexes_y = neg_y_direction[min_neg_dir]
        index_sorted = sorted(list(indexes_y))
        index_y = self.get_vertex(index_sorted)
        #index_y = 0
        #while (index_y < len(self.y_vertices_list)) and (self.y_vertices_list[index_y] != index_sorted):
        #    index_y += 1
        #if (index_y == len(self.y_vertices_list)):
        #    self.y_vertices_list.append(index_sorted)
        result = np.zeros(index_y + 1)
        value = sum(gy[indexes_y])
        result[index_y] = value
        return result
        #point = sparse.dok_matrix((index_y+1, self.dim))
        #point = np.zeros((self.dim, index_y + 1))
        #point[index_y, index_sorted] = 1
        #return point
    def get_vertex(self, v):
        #v = sorted(v)
        i = 0
        while (i < self.vertex_counter) and (self.y_vertices_list[i] != v):
            i += 1
        if (i == self.vertex_counter):
            self.vertex_counter += 1
            self.y_vertices_list.append(v)
            if self.be_verbose:
                print("Added vertex to list:", "egdes are ", v)
            return i
        return i

class XY_FW_Algorithm(fw.FW_Base_Algorithm):
    def __init__(self, y_class = None, y_option = None, *args, **kwargs):
        y_class = y_class or self.y_class
        if y_option:
            self.y = y_class(**y_option)
        else:
            ykwargs = kwargs.copy()
            ykwargs["verbosity"] = (kwargs.get("verbosity") or 1) - 1
            self.y = y_class(*args, **ykwargs)
        super().__init__(*args, **kwargs)
    def set_parameters(self, dim, vertices, ykwargs = {}, **kwargs):
        xdim, self.ydim = dim
        xvertices, self.yvertices = vertices
        self.y.set_parameters(self.ydim, self.yvertices, **ykwargs)
        super().set_parameters(xdim, xvertices, **kwargs)
    def initialize_y(self, startY, **kwargs):
        self.y.initialize(start_point_convcomb = startY)
        self.y.initialize_log()
        return
    def set_objective(self, x_generator, y_generator, start_point = None, **kwargs):
        self.x_generator = x_generator
        self.y_generator = y_generator
        if (type(start_point)==type(None)):
            start_point = np.zeros(self.dim + self.ydim)
        x = start_point[:self.dim]
        y = start_point[self.dim:]
        yf, ygrad = self.y_generator(x)
        xf, xgrad = self.x_generator(y)
        self.y.set_objective(yf, ygrad, start_point=y, **kwargs)
        super().set_objective(xf, xgrad, start_point=x, **kwargs)
    def run(self, max_iter = 1000, start_point_convcomb = (None,None), **kwargs):
        self.log_name = "X:"
        self.y.log_name = "Y:"
        startX, startY = start_point_convcomb 
        self.initialize(startX, **kwargs)
        self.initialize_log()
        self.initialize_y(startY, **kwargs)
        self.xy_t = np.hstack((self.x_t, self.y.x_t))
        self.f, self.dF = self.x_generator(self.y.x_t)
        #for t in range(max_iter):
        self.t = 0
        while (self.t <= max_iter):
            t = self.t
            self.path.append(self.xy_t)
            self.log("f(xy_t)", self.f_x)
            self.log("L", self.lipschitz)
            gradienteX = self.dF(self.x_t)
            gap = self.iteration(gradienteX)
            self.log("xtrace", gap)
            self.gap = gap
            if (gap > self.epsilon):
                self.shrink_active_set()
                old_x_t = self.x_t
                old_f_x = self.f_x
                self.x_t = self.active_set.value()
                self.f_x = self.f(self.x_t)
                self.update_lipschitz(gradienteX, old_x_t, old_f_x)
                self.y.f, self.y.dF = self.y_generator(self.x_t)
                self.y.f_x = self.y.f(self.y.x_t)
            gradienteY = self.y.dF(self.y.x_t)
            gap = self.y.iteration(gradienteY)
            self.log("ytrace", gap)
            self.gap = max(gap, self.gap)
            self.track_time(self.gap)
            if (self.gap <= self.epsilon):
                self.xy_t = np.hstack((self.x_t, self.y.x_t))
                self.print("Found a result at iteration number %d"%t)
                fx = self.y.f_x
                self.print(" gap = ", self.gap, "x_%d="%t, self.xy_t, "f(xy_%d)="%t, fx)
                self.result = (self.xy_t, (self.active_set, self.y.active_set), fx , True)
                return True
            if (gap > self.epsilon):
                self.y.shrink_active_set()
                old_y_t = self.y.x_t
                old_f_y = self.y.f_x
                self.y.x_t = self.y.active_set.value()
                self.y.f_x = self.y.f(self.y.x_t)
                self.y.update_lipschitz(gradienteY, old_y_t, old_f_y)
                self.f, self.dF = self.x_generator(self.y.x_t)
                self.f_x = self.f(self.x_t)
            self.xy_t = np.hstack((self.x_t, self.y.x_t))
            self.t += 1
        self.print("Exceeded max number of iteration")
        self.print(" gap = ", self.gap, "x_%d="%t, self.xy_t, "f(x_%d)="%t, self.f_x)
        self.result = (self.xy_t, (self.active_set, self.y.active_set), self.f_x , False)
        return False
