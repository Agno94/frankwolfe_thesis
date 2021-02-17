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
import sys, re, glob, gc, importlib, os, signal, json

import fw_algorithms as fw
import ssc_algorithms as ssc
import xy_fw_algorithms as xy_fw

S = 5

dimacs_dir_path = "/home/agno/Uni/Thesis/Code/dimacs/"

class AwayStep_SSC_s_defective_Clique(xy_fw.XY_FW_Algorithm, ssc.FW_AwayStep_SSC_Algorithm):
    label = "SSC AwayStep FW"
    y_class = ssc.FW_AwayStep_SSC_Algorithm

class Pairwise_SSC_s_defective_Clique(xy_fw.XY_FW_Algorithm, ssc.FW_Pairwise_SSC_Algorithm):
    label = "SSC Pairwise FW"
    y_class = ssc.FW_Pairwise_SSC_Algorithm

class AwayStep_ClassicLipshitz_s_defective_Clique(xy_fw.XY_FW_Algorithm, fw.FW_AwayStep_ClassicLipschizStep_Algorithm):
    label = "Classic AwayStep FW w/ Lipschiz Step"
    y_class = fw.FW_AwayStep_ClassicLipschizStep_Algorithm

class Pairwise_ClassicLipshitz_s_defective_Clique(xy_fw.XY_FW_Algorithm, fw.FW_Pairwise_ClassicLipschizStep_Algorithm):
    label = "Classic Pairwise FW w/ Lipschiz Step"
    y_class = fw.FW_Pairwise_ClassicLipschizStep_Algorithm

algo_list = []
test_results = {}
error_list = []
options = {"verbosity" : 1, "epsilon": 1e-4, "lipschitz_cap": 1e2}
algo_list.append(AwayStep_ClassicLipshitz_s_defective_Clique(**options))
algo_list.append(Pairwise_ClassicLipshitz_s_defective_Clique(**options))
#for A in algs:
#    A.set_line_search_parameters()
algo_list.append(AwayStep_SSC_s_defective_Clique(**options))
algo_list.append(Pairwise_SSC_s_defective_Clique(**options))
algo_list = algo_list

np.set_printoptions(threshold=100)

def output_to_json():
    currenttime = datetime.now().isoformat()[:19]
    f=open("defclique_test_result_%s.json"%currenttime, "w")
    json.dump(test_results, f, indent = 1)
    f.close()

def sigint(sig, frame):
    print("SIGINT received")
    print("Saving Results")
    output_to_json()
    print("Exit")
    sys.exit(2)

signal.signal(signal.SIGINT, sigint)


def graph_from_edges(N, M, edges):
    A_G = np.zeros((N,N))
    for v1, v2 in edges:
        A_G[v1-1, v2-1] = 1
        A_G[v2-1, v1-1] = 1
    E_bar_list = []
    for v1 in range(0,N):
        for v2 in range(v1+1, N):
            if (not A_G[v1,v2]):
                E_bar_list.append((v1,v2))
    max_edges = N*(N-1)//2
    print("Check: %d == %d"%(max_edges, len(E_bar_list) + M), max_edges == (len(E_bar_list) + M))
    return A_G, E_bar_list

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
            if (len(args)>=4) and (args[0]=="p"):
                n, m = int(args[2]), int(args[3])
                N = max(N,n)
                M = m
        if (len(args)>=3) and (args[0]=="e"):
            v1, v2 = int(args[1]), int(args[2])
            edges.append((v1,v2))
            N = max(N, v1, v2)
    print("Caricamento %s: Controllo caricamento"%name, (n == N), (m == M), (M == len(edges)))
    t3=process_time()
    A_G, E_bar_list = graph_from_edges(N, M, edges)
    del row, edges
    t4=process_time()
    return A_G, E_bar_list, N, M, (t0,t1,t2,t3,t4)

dim, Mbar, E_bar_array, edge_second_node_dict, edge_two_node_dict = 0,0,0,0,0
def create_functions(A_G, E_bar_list, c, N, M, l_yparameter = 1):
    Mbar = len(E_bar_list)
    dim = N
    E_bar_array = np.array(E_bar_list)
    edge_second_node_dict = { v1: [] for v1 in range(0,N)}
    edge_two_node_dict = {}
    i = 0
    for v1,v2 in E_bar_list:
        edge_second_node_dict[v1].append(i)
        edge_two_node_dict[(v1,v2)] = i
        edge_two_node_dict[(v2,v1)] = i
        i += 1
    for v1 in range(0,N):
        edge_second_node_dict[v1] = np.array(edge_second_node_dict[v1])
    def A_Gbar(y):
        A_y = np.zeros((N,N))
        for v1 in range(0,N):
            e_list = edge_second_node_dict[v1]
            if not len(e_list):
                continue
            try:
                v2s = E_bar_array[e_list,1]
            except IndexError:
                print(v1, e_list)
            A_y[v1,v2s] = y[e_list]
            A_y[v2s,v1] = y[e_list]
        return A_y
    def generate_with_y(y):
        A_y = A_Gbar(y)
        Q = - (A_G + A_y + 0.5 * np.eye(N))
        y_sq = y.T @ y
        def obj_of_x(x):
            f = x.T @ Q @ x - l_yparameter / 2 * y_sq
            return f
        def grad_x_of_x(x):
            return 2 * Q @ x
        return obj_of_x, grad_x_of_x
    def generate_with_x(x):
        gx = np.zeros(Mbar)
        for v1 in range(0,N):
            e_list = edge_second_node_dict[v1]
            if not len(e_list):
                continue
            try:
                v2s = E_bar_array[e_list,1]
            except IndexError:
                print(v1, e_list)
            gx[e_list] = - 2 * x[v1] * x[v2s]
        x_Q_x = - x.T @ (A_G + 0.5 * np.eye(N)) @ x
        def obj_of_y(y):
            f = x_Q_x
            nonzero, = np.where(y >= 1e-9)
            for e in nonzero:
                v1, v2 = E_bar_list[e]
                f -= 2 * x[v1] * x[v2] * y[e]
            f = f - l_yparameter / 2 * y.T @ y
            return f
        def grad_y_of_y(y):
            return gx - l_yparameter * y
        return obj_of_y, grad_y_of_y
    #return objective, gradient
    def check_clique(xy, S, display = 0):
        x = xy[:N]
        y = xy[N:]
        if (abs(sum(x)-1) > 1e-6):
            print("ERROR sum(x) == %.2f"%sum(x))
        print("ERROR" if (sum(y) > S+1e-6) else "INFO", "sum(y) == %.2f"%sum(y))
        A_y = A_Gbar(y)
        A_y[np.where(abs(A_y) >= 1e-5)] = 1
        findx, = np.where(abs(x) > 1e-3/N)
        findy, = np.where(abs(y) > 1e-5)
        print("x, y, x.size, y.size, findx", x, y, x.size, y.size, findx, findy)
        c = findx.size
        vx = sum(sum(A_G[findx][:, findx]))
        vy = sum(sum(A_y[findx][:, findx]))
        missing_egdes = vy / 2
        if missing_egdes > S+1e-8:
                print("ERROR", S)
        delta = (vx + vy) - (c)*(c-1)
        result = abs(delta) < 0.1
        print("Check is", result, " (clique size:%d, vx:%.1f, vy:%.1f, delta:%.2f)"%(c, vx, vy, delta))
        #plt.show()
        if display:
            A_show = 0.8 * A_G + 0.5 * A_y
            A_show[findx][:, findx] += 1.1 * A_G[findx][:, findx]
            plt.matshow(A_show[findx][:, findx])
        if display > 1:
            y_vertices = listOfVertices(findy)
            print("y's edges' vertices:",y_vertices)
            indexes = list(findx) + sorted(list(set(y_vertices) - set(findx)))
            print("  indexes=",indexes)
            tot_vertices = np.array([int(x) for x in indexes])
            plt.matshow(A_show[tot_vertices][:, tot_vertices], cmap='plasma')
        plt.show()
        return (result, c, missing_egdes)
    return generate_with_y, generate_with_x, check_clique

def generate_random_start_01y(i = None):
    start_indexes = np.arange(0,N)
    if (i != None):
        np.random.seed(i)
    w = np.random.rand(N)
    xstart_weights = w / sum(w)
    xstart = (x_vertices[start_indexes], start_indexes, xstart_weights)
    num_y_tochoose = np.random.randint(1,S)
    chosen_edges = np.random.choice(np.arange(0,Mbar), size=num_y_tochoose)
    ystart_list = [sorted(list(chosen_edges))]
    ystart_matrix = np.zeros((1, Mbar))
    ystart_matrix[0, chosen_edges] = 1
    return xstart, ystart_matrix, ystart_list, False

test_name_list_file=open("dimacs/to_be_tested_defclique")
test_name_list = test_name_list_file.read().split()
test_name_list_file.close()
active_test_name_list = []
test_skip = False
for test_name in test_name_list:
    if (test_name == "!stop"):
        test_skip = True
        continue
    if (test_name == "!start"):
        test_skip = False
        continue
    if (test_name[0] == "#") or test_skip: 
        continue
    if (not os.path.isfile(dimacs_dir_path + test_name + ".clq")):
        print("Missing test", test_name)
        continue
    active_test_name_list.append(test_name)    
active_test_name_list = active_test_name_list[:]

print("Test list: ", active_test_name_list)
test_results = {}
error_list = []
error_count = 0
for test_name in active_test_name_list:
    if os.path.isfile("alt"):
        os.remove("alt")
        break
    print("*"*30+"\nRunning test %s\n"%test_name+"*"*30)
    A_G, E_bar_list, N, M, times = load_dimacs_graph(test_name + ".clq")
    Mbar = len(E_bar_list)
    print("N = ", N, " Mbar = ", Mbar, " S = ", S)
    y_vertices = xy_fw.S_minimal_subset_descriptor(Mbar, S)
    x_vertices = fw.SimplexDescriptor(N)
    generate_with_y, generate_with_x, check_clique = create_functions(
        A_G, E_bar_list, None, N, M, l_yparameter=1e-4)
    times = {ALG.label:[] for ALG in algo_list}
    clique_nums = {ALG.label:[] for ALG in algo_list}
    missing_edges_nums = {ALG.label:[] for ALG in algo_list}
    for i in range(10):
        print("===\nExecution #%02d\n==="%i)
        xstart, ymatrix, ylist, yweight = generate_random_start_01y(i)
        yindexes = []
        for yvl in ylist:
            yindexes.append(y_vertices.get_vertex(yvl))
        for ALG in algo_list:
            print("Execution %02d of %s for test %s"%(i, ALG.label, test_name))
            ALG.set_parameters((N, Mbar), (x_vertices, y_vertices))
            ALG.set_objective(generate_with_y, generate_with_x,
                              start_point = np.hstack((np.ones(N)/N, np.ones(Mbar)/Mbar)),
                              alpha = 1)
            xstartcc = fw.ConvexCombination(*xstart)
            ystartcc = fw.ConvexCombination(ymatrix, yindexes, yweight)
            t0 = process_time()
            ALG.run(max_iter = 50000, start_point_convcomb = (xstartcc, ystartcc))
            t1 = process_time()
            delta_t = t1 - t0
            x, cv, fx, status = ALG.result
            check, cliq_num, missing_edges = check_clique(x, S, display=0)
            if (not check):
                ALG.epsilon = 3.1e-5
                print("Searching again")
                t0 = process_time()
                ALG.run(max_iter = 20000, start_point_convcomb = cv)
                t1 = process_time()
                delta_t += t1 - t0
                x, cv, fx, status = ALG.result
                check, cliq_num, missing_edges = check_clique(x, S, display=0)
                ALG.epsilon = 1e-4
            if (check):
                print("Is correct")
            else:
                print("Is NOT correct")
                error_list.append((test_name, i, ALG.label, x, check, cliq_num))
            print("Y vertex counter:", y_vertices.vertex_counter)
            print(x, fx, "\n")
            times[ALG.label].append(delta_t)
            clique_nums[ALG.label].append(cliq_num)
            missing_edges_nums[ALG.label].append(missing_edges)
    for ALG in algo_list:
        print(ALG.label)
        print(clique_nums[ALG.label], sum(clique_nums[ALG.label])/10)
        print(times[ALG.label], sum(times[ALG.label])/10)
        print(missing_edges_nums[ALG.label], sum(missing_edges_nums[ALG.label])/10)
    test_results[test_name] = {
        "cpu_time": times,
        "clique_number": clique_nums,
        "missing_edges": missing_edges_nums
    }
    if (len(error_list) > error_count):
        print("!"*10)
        print("NEW ERRORS")
        print("!"*10)
        print(error_list[error_count:])
        error_count = len(error_list)
    elif error_list:
        print("No new error")

print("Error list:", error_list)

test_results["errors"] = error_list
output_to_json()
del test_results["errors"]

table_latex = ""
test_table = []
for test_name in test_results:
    result = test_results[test_name]
    data_to_show = []
    row_number = len(result["clique_number"].keys())
    first_row = True
    for alg_label in result["clique_number"]:
        table_latex += "\n"
        name_str = test_name.replace("_", "\\_")
        if first_row:
            data_to_show_str = [
                "\\multirow{%d}{*}{%s}"%(row_number, name_str), 
                "\\multirow{%d}{*}{$%d$}"%(row_number, S),]
            first_row = False
        else:
            data_to_show_str = ["", ""]
        alg_name = alg_label.split(" FW ")[0]
        data_to_show_str.append(alg_name)
        d = []
        clique_number = np.array(result["clique_number"][alg_label])
        cpu_time = np.array(result["cpu_time"][alg_label])
        missing_edges = np.array(result["missing_edges"][alg_label])
        v = [
            min(clique_number),
            np.mean(clique_number),
            max(clique_number),
            np.std(clique_number),
            min(cpu_time),
            np.mean(cpu_time),
            max(cpu_time),
            np.std(cpu_time),
            min(missing_edges),
            np.mean(missing_edges),
            max(missing_edges),
            np.std(missing_edges),
        ]
        d[0:2] = [ str(v[0]), "%.1f"%v[1], str(v[2]) ]
        d[3:8] = [ "%.2f"%l for l in v[3:8] ]
        d[9:12] = [ str(v[8]), "%.1f"%v[9], str(v[10]), "%.1f"%v[11] ]
        data_to_show += v
        data_to_show_str += d
        table_latex += " & ".join(data_to_show_str) + " \\\\"
    test_table.append(data_to_show)
    table_latex += " \hline"
print(r'''
{
	\footnotesize
\begin{tabular}{||l||l|l||c|c|c|c||c|c|c|c||c|c|c|c||}
	\hline
	%\hline \hline & \multicolumn{8}{c||}{Classic Away-step} & \multicolumn{8}{c||}{Classic Pairwise} & \multicolumn{8}{c||}{SSC Away-step} & \multicolumn{8}{c||}{SSC Pairwise} \\ \hline
	Instance & s & Algorigthm & \multicolumn{4}{c||}{clique number} & \multicolumn{4}{c||}{cpu time} & \multicolumn{4}{c||}{missing edges} \\ \hline
	& & & min & mean & max & std & min & mean & max & std & min & mean & max & std \\ \hline
''')
print(table_latex)
print(r'''
\end{tabular}
}
''')

if (error_list):
    print("Errors have occured")
