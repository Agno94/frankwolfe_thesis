import numpy as np
#from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import line_search
from scipy import sparse, linalg
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from datetime import datetime
from time import process_time
import sys, re, glob, gc, importlib, json, signal

import fw_algorithms as fw
import ssc_algorithms as ssc

dimacs_dir_path = "/home/agno/Uni/Thesis/Code/dimacs/"

test_results = {}
algo_list = []
options = {"verbosity" : 0, "epsilon": 1e-6}
algo_list.append(fw.FW_AwayStep_ClassicLipschizStep_Algorithm(**options))
algo_list.append(fw.FW_Pairwise_ClassicLipschizStep_Algorithm(**options))
for A in algo_list:
    A.set_line_search_parameters()
algo_list.append(ssc.FW_AwayStep_SSC_Algorithm(**options))
algo_list.append(ssc.FW_Pairwise_SSC_Algorithm(**options))

np.set_printoptions(threshold=100)

def output_to_json():
    currenttime = datetime.now().isoformat()[:19]
    f=open("clique_test_result_%s.json"%currenttime, "w")
    json.dump(test_results, f, indent = 1)
    f.close()

def sigint(sig, frame):
    print("SIGINT received")
    print("Saving Results")
    output_to_json()
    print("Exit")
    sys.exit(2)

signal.signal(signal.SIGINT, sigint)

def load_dimacs_graph(name):
    t0=process_time()
    f = open(dimacs_dir_path + name, 'r')
    edges = []
    N, M, m, n = 0,0,0,0
    t1=process_time()
    rows = f.readlines()
    t2=process_time()
    for row in rows:
        args = row.split()
        if (not N):
            #if (match):
            if (len(args)>=4) and (args[0]=="p"):
                #n, m = int(match.group(1)), int(match.group(2))
                n, m = int(args[2]), int(args[3])
                N = max(N,n)
                M = m
        if (len(args)>=3) and (args[0]=="e"):
            #v1, v2 = int(match.group(1)), int(match.group(2))
            v1, v2 = int(args[1]), int(args[2])
            edges.append((v1,v2))
            N = max(N, v1, v2)
    print("Caricamento %s: Controllo caricamento"%name, (n == N), (m == M), (M == len(edges)))
    t3=process_time()
    Q = -np.eye(N)
    for v1, v2 in edges:
        Q[v1-1, v2-1] = -2
        Q[v2-1, v1-1] = -2
    del row, edges
    c = np.zeros(N)
    t4=process_time()
    return Q, c, N, M, (t0,t1,t2,t3,t4)

def check_clique(Q, x):
    findx, = np.where(abs(x)>1e-3)
    print(findx)
    c = findx.size
    v = abs(sum(sum(Q[findx][:, findx])) + c*(2*(c-1)+1))
    print("Check clique:", c, v)
    return v<0.1, c

def create_functions(Q, c, N):
    dim = N
    def objective(x):
        return 1/2* x.T @ (Q @ x) - c @ x
    def gradient(x):
        return Q @ x - c
    return objective, gradient

test_name_list_file=open("dimacs/to_be_tested")
test_name_list = test_name_list_file.read().split()
test_name_list_file.close()
print("Test list: ", test_name_list)
test_results = {}
error_list = []
times = {ALG.label:[] for ALG in algo_list}
clique_nums = {ALG.label:[] for ALG in algo_list}
for test_name in test_name_list[:]:
    if (test_name[0]=="#"): 
        continue
    print("*"*30+"\nRunning test %s\n"%test_name+"*"*30)
    Q, c, N, M, times = load_dimacs_graph(test_name + ".clq")
    simplex = fw.SimplexDescriptor(N)
    objective, gradient = create_functions(Q, c, N)
    times = {ALG.label:[] for ALG in algo_list}
    clique_nums = {ALG.label:[] for ALG in algo_list}
    iteration_nums = {ALG.label:[] for ALG in algo_list}
    for i in range(0,10):
        print("===\nExecution #%02d\n==="%i)
        np.random.seed(i)
        w = np.random.rand(N)
        start_weights = w / sum(w)
        start_indexes = np.arange(0,N)
        for ALG in algo_list:
            ALG.set_parameters(N, simplex)
            ALG.set_objective(objective, gradient, 
                              start_point = simplex[N-1],
                              alpha = 1)#alpha = 2 * LipschitzDFLowerBound)
            x0 = fw.ConvexCombination(simplex[start_indexes], start_indexes, start_weights)
            t0 = process_time()
            ALG.run(max_iter = 10000, start_point_convcomb = x0)
            t1 = process_time()
            delta_t = t1 - t0
            x, cv, fx, status = ALG.result
            iterations = ALG.t
            check, cliq_num = check_clique(Q, x)
            if (check):
                print("Is correct")
            else:
                print("Is NOT correct")
                error_list.append((test_name, i, x, check, cliq_num))
            print(x, fx, "\n")
            times[ALG.label].append(delta_t)
            clique_nums[ALG.label].append(cliq_num)
            iteration_nums[ALG.label].append(iterations)
    for ALG in algo_list:
        print(ALG.label)
        print(clique_nums[ALG.label], sum(clique_nums[ALG.label])/10)
        print(times[ALG.label], sum(times[ALG.label])/10)
        print(iteration_nums[ALG.label], sum(iteration_nums[ALG.label])/10)
    test_results[test_name] = {"cpu_time": times, "clique_number": clique_nums, "iteration_number": iteration_nums}

output_to_json()

table_latex = ""
test_table = []
for test_name in test_results:
    result = test_results[test_name]
    data_to_show_str = [test_name.replace("_", "\\_")]
    data_to_show = []#test_name]
    for alg_label in result["clique_number"]:
        p=4
        d = []
        clique_number = np.array(result["clique_number"][alg_label])
        cpu_time = np.array(result["cpu_time"][alg_label])
        v = [
            min(clique_number),
            np.mean(clique_number),
            max(clique_number),
            np.std(clique_number),
            min(cpu_time),
            np.mean(cpu_time),
            max(cpu_time),
            np.std(cpu_time)
        ]
        d[0:2] = [ str(v[0]), "%.1f"%v[1], str(v[2]) ]
        d[3:8] = [ "%.2f"%l for l in v[3:8] ]
        data_to_show += v
        data_to_show_str += d
        #print(v, d)
    test_table.append(data_to_show)
    table_latex += " & ".join(data_to_show_str) + " \\\\\n"
print(table_latex)