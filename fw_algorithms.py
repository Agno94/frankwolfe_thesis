import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import line_search
from scipy import sparse
from scipy import linalg
from datetime import datetime
from time import process_time

def LinearMinimizer(c,A_ub,b_ub) :
    c = matrix(c.T.astype('float'))
    G = matrix(A_ub.astype('float'))
    h = matrix(b_ub.T.astype('float'))
    sol = solvers.lp(c, G, h, solver = 'glpk')
    return np.matrix(sol['x'])

def LinearMinimizerVertices(c, vertices): 
    index = np.argmin(vertices @ c)
    sol = vertices[index]
    return index, sol

def argmin_nonzero(k):
    threshold = (1 / k.shape[0]) * 1e-12
    index, Min, i = np.NaN, np.Inf, 0
    for x in k:
        if (abs(x) <= threshold): continue
        if (x < Min):
            Min = x
            index = i
        i += 1
    return index

class ConvexCombination:
    def __init__(self, points_matrix, vertex_indexes, weights = False):
        self.matrix = np.array(points_matrix)
        self.indexes = np.array(vertex_indexes)
        if (type(weights) == bool):
            l = points_matrix.shape[0]
            self.weights = 1/l * np.ones(l)
        else:
            self.weights = np.array(weights)
    def __del__(self):
        #print(" - - deleted - - ")
        #del self.matrix
        #del self.indexes
        #del self.weights
        #super().__del__()
        pass
    def copy(self):
        return ConvexCombination(self.matrix, self.indexes, self.weights)
    def value(self):
        return self.matrix.T @ self.weights
    def drop(self, index):
        self.matrix = np.delete(self.matrix, index, 0)
        self.indexes = np.delete(self.indexes, index)
        self.weights = np.delete(self.weights, index)
    def size(self):
        return self.weights.shape[0]

class SimplexDescriptor():
    def _get_column(self, k):
        r = np.zeros(self.dim)
        r[k] = 1
        return r
    def _change_column(self, r, i, k):
        r[i,k] = 1
    def __getitem__(self, key):
        if (type(key) == tuple):
            if (len(key) > 2):
                raise IndexError("Too many indexes: %d > 2"%len(key))
            k1, k2 = key
        else:
            k1, k2 = key, slice(None,None,None)
        if (type(k1) == int) or (type(k1) == np.int64):
            r = self._get_column(k1)
            return r[k2]
        if (type(k1) == slice):
            start, stop, step = k1.indices(self.vertices_numbers)
            rlen = (stop - start - 1) // step + 1
            indexes = list(range(start, stop, step))
        else:
            rlen = len(k1)
            indexes = k1
        i, r = 0, np.zeros([rlen, self.dim])
        for x in indexes:
            self._change_column(r, i, x)
            i += 1
        return r[:, k2]
        #return r
    def __init__(self, dimension):
        self.dim = dimension
        self.vertices_numbers = self.dim
    def __matmul__(self, v):
        if (len(v) != self.dim):
            raise IndexError('Wrong dimension')
        r = v[np.arange(0,self.dim)]
        return r

class Simplex0Descriptor(SimplexDescriptor):
    def __init__(self, dimension):
        self.dim = dimension
        self.vertices_numbers = self.dim + 1
    def _get_column(self, k):
        r = np.zeros(self.dim)
        if (k != 0):
            r[k-1] = 1
        return r
    def _change_column(self, r, i, k):
        if k:
            r[i,k-1] = 1
    def __matmul__(self, v):
        if (len(v) != self.dim):
            raise IndexError('Wrong dimension')
        r = np.hstack((np.zeros(1), v[np.arange(0,self.dim)]))
        return r

def caratheodory(convexcomb):
    S = convexcomb.matrix.T
    number = convexcomb.weights.shape[0]
    S0 = S[:,1:] - np.matrix(S[:,0]).T @ np.matrix(np.ones(number -1))
    nullspace = linalg.null_space(S0)
    k=nullspace[:,0]
    for t in nullspace.T:
        if (t @ np.ones(number - 1)):
            k = t
            break
    k = np.hstack([-k @ np.ones(number -1), k]) / (k @ np.ones(number -1))
    index = argmin_nonzero(k)
    alpha = - convexcomb.weights[index] / k[index]
    #print(" -- -- k, index, Lambda, alpha", k, index, convexcomb.weights, alpha)
    new_weights = convexcomb.weights + alpha * k
    return ConvexCombination(S.T, convexcomb.indexes, new_weights)

class LineSearch_Base:
    def set_line_search_parameters(self, **kwargs):
        pass
    def do_line_search(self, x_t, d_t, grad, gamma_max):
        pass
class LineSearch_Scipy(LineSearch_Base):
    def do_line_search(self, x_t, d_t, grad, gamma_max):
        line_search_res = line_search(
            self.f, self.dF, x_t, d_t, grad,
            amax = gamma_max,
            maxiter = 100, c2 = 0.99,
        )
        if (line_search_res[0]): return line_search_res[0]
        line_search_res = line_search(
            self.f, self.dF, x_t, d_t, grad,
            amax = gamma_max / 100,
            c2 = 1
        )
        if (not line_search_res[0]):
            self.print("LinearSearch: Something off", line_search_res[0], gamma_max, line_search_res, level=1)
        return line_search_res[0] or 0
class LineSearch_Armijo(LineSearch_Base):
    def set_line_search_parameters(self, delta = 0.5, gamma = 1e-4, max_iter = 1000):
        self.LSO_delta = delta
        self.LSO_gamma = gamma
        self.LSO_max_iter = max_iter
    def do_line_search(self, x_t, d_t, grad, gamma_max):
        gamma = gamma_max
        fx = self.f(x_t)
        s = self.LSO_gamma * (grad.T @ d_t)
        for x in range(0,self.LSO_max_iter):
            if (self.f(x_t + gamma * d_t) <= (fx + gamma * s)):
                self.log("ArmijoSearchSteps", x)
                return gamma
            gamma = self.LSO_delta * gamma
        self.log("ArmijoSearchSteps", x+1)
        self.print("LinearSearch: Something off", gamma_max, level=1)
        return gamma
class LineSearch_LipscitzBased(LineSearch_Base):
    def do_line_search(self, x_t, d_t, grad, gamma_max):
        #-g'd/(L\|d\|^2) -g @ d / L d_t**2
        gamma = -(grad.T @ d_t) / self.lipschitz / (d_t.T @ d_t)
        self.print(" lso: ", "gamma is", gamma, "gamma_max:", gamma_max, level = 3)
        #self.log("LS: gamma>gamma_max", gamma>gamma_max)
        return min(gamma, gamma_max)
    
class FW_Base_Algorithm:
    label = "Base FW Active-Set Model"
    def __init__(self, epsilon = 1e-6, zero_threshold = 1e-12, start_point_offset = 0, verbosity = 1, lipschitz_cap = 0, **kwargs):
        self.epsilon = epsilon
        self.zero_threshold = zero_threshold
        self.result = None
        self.start_point_offset = start_point_offset
        self.verbosity = verbosity
        self.log_name = ""
        self.lipschitz_cap = lipschitz_cap
        self.t = 0
    def set_parameters(self, dim, vertices):
        self.dim = dim
        self.vertices = vertices
        #self.eqA = A
        #self.eqb = b
    def print(self, *args, level = 1):
        if (level <= self.verbosity):
            print(self.log_name, *args)
    def set_objective(self, func, grad, lipschitz = 0, start_point = None, **kwargs):
        self.f = func
        self.dF = grad
        self.lipschitz = lipschitz
        if (type(start_point)==type(None)):
            self.lipschitz_start_point = np.zeros(self.dim)
        else:
            self.print("x_-1 = ", start_point, level = 2)
            self.lipschitz_start_point = start_point
    def linear_minimizer(self, g):
        index = np.argmin(self.vertices @ g)
        point = self.vertices[index]
        return index, point
    def log(self, key, entry):
        if (not key in self.log_keys):
            self.print("Add key %s to log"%key, level=3)
            self.log_keys.add(key)
            if (not key in self.data):
                self.data[key] = []
        self.data[key].append(entry)
    def track_time(self, gap):
        delta = process_time() - self.start_time
        self.time_trace.append([delta, gap])
    def initialize_log(self,):
        self.print("Inizializing instance of ", self.__class__.__name__, level = 0)
        self.print(" target epsilon=", self.epsilon,
                "start lipschitz=", self.lipschitz, level = 0)
        self.print(" with x_0=",self.x_t,
                   "active-set-coefficients=",self.active_set.weights)
        self.path = []
        self.start_time = process_time()
        self.time_trace = []
        self.data = {"trace":[]}
        self.log_keys = set(["trace"])
        self.ready = True
    def initialize(self, start_point_convcomb = None, start_lipschitz = 0, **kwargs):
        self.result = None
        self.sameX = 0
        if start_point_convcomb:
            self.active_set = start_point_convcomb
        else:
            indexes = np.arange(self.start_point_offset,self.start_point_offset+1)
            self.active_set = ConvexCombination(self.vertices[indexes], indexes)
        self.x_t = self.active_set.value()
        self.f_x = self.f(self.x_t)
        if (start_lipschitz):
            self.lipschitz = start_lipschitz
        elif (not self.lipschitz):
            start_fx = self.f(self.lipschitz_start_point)
            self.update_lipschitz(self.dF(self.lipschitz_start_point), self.lipschitz_start_point, start_fx)
            delta_x = (self.x_t - self.lipschitz_start_point)
            delta_ddfx= (self.dF(self.lipschitz_start_point) - self.dF(self.x_t))
            ratio = linalg.norm(delta_ddfx) / linalg.norm(delta_x)
            self.print("alternative lipschitz bound = ", ratio, "is ", "better:" if (ratio > self.lipschitz) else "worst", level = 1)
        self.ready = True
    def update_lipschitz(self, gradiente, other_x_t, f_other_x):
        delta_x = self.x_t - other_x_t
        delta_x_normq = (delta_x.T @ delta_x)
        if (not delta_x_normq):
            self.print("Same x, Δx = 0", level=1)
            self.sameX += 1
            #self.print((delta_x == 0).all(), self.x_t, np.where(self.x_t > 0), level=1)
            #if (self.sameX == 3):
            #    self.t += 100
            return
        self.sameX = 0
        L_k = 2 * (self.f_x - f_other_x - gradiente.T @ delta_x) / delta_x_normq
        self.print("L_k bound = ", L_k, "old bound =", self.lipschitz, level = 2)
        if (self.lipschitz < abs(L_k)):
            self.print("new Lipschiz(∇f) = ", L_k, ">=", self.lipschitz, level = 1)
        self.lipschitz = max(self.lipschitz, abs(L_k))
        if self.lipschitz_cap and (self.lipschitz > self.lipschitz_cap):
            self.print("Lipschitz constant capped, L~ = ", self.lipschitz, level = 1)
            self.lipschitz = min(self.lipschitz, self.lipschitz_cap)
    def run(self, max_iter = 1000, start_point_convcomb = None, **kwargs):
        self.initialize(start_point_convcomb, **kwargs)
        self.initialize_log()
        #for t in range(max_iter):
        self.t = 0
        while (self.t <= max_iter):
            t = self.t
            self.path.append(self.x_t)
            self.log("f(x_t)", self.f_x)
            self.log("L", self.lipschitz)
            gradiente = self.dF(self.x_t)
            gap = self.iteration(gradiente)
            self.log("trace", gap)
            self.track_time(gap)
            self.gap = gap
            if (gap <= self.epsilon):
                self.print("Found a result at iteration number %d"%t)
                self.print(" gap = ", gap, "x_%d="%t, self.x_t, "f(x_%d)="%t, self.f_x)
                self.result = (self.x_t, self.active_set, self.f_x, True)
                return True
            self.shrink_active_set()
            #log gap, and other info
            old_x_t = self.x_t
            old_f_x = self.f_x
            self.x_t = self.active_set.value()
            self.f_x = self.f(self.x_t)
            self.update_lipschitz(gradiente, old_x_t, old_f_x)
            self.t += 1
        self.print("Exceeded max number of iteration")
        self.print(" gap = ", gap, "x_%d="%t, self.x_t, "f(x_%d)="%t, self.f_x)
        self.result = (self.x_t, self.active_set, self.f_x, False)
        return False
    #def iteration(self, gradiente):
    #    pass
    def shrink_active_set(self):
        w = self.active_set.weights
        l = w.size
        zero_weights = np.where(np.abs(w) <= self.zero_threshold/l)
        if zero_weights[0].size:
            self.active_set.drop(zero_weights)

#class FW_Base_with_Lipschitz_Algorithm(FW_Base_Algorithm):
#    label = "Base FW Active-Set and Lipscitz Model"   
            
class FW_Pairwise_Classic_Proto(FW_Base_Algorithm):
    def iteration(self, grad):
        # s_t: vertex given by linear minimizer
        s_t_index, s_t = self.linear_minimizer(grad)
        s_t = np.array(s_t).ravel()
        d_FW = s_t - self.x_t
        g_FW = -grad.T @ d_FW
        if g_FW <= self.epsilon:
            return g_FW
        # s_t: vertex given by active set
        v_t_setIndex = np.argmax(self.active_set.matrix @ grad)
        v_t_vertexIndex = self.active_set.indexes[v_t_setIndex]
        v_t = self.active_set.matrix[v_t_setIndex]
        # pairwise direction
        d_t = s_t - v_t
        step_type = "PFW"
        weight_v_t = self.active_set.weights[v_t_setIndex]
        gamma_max = weight_v_t
        gamma = self.do_line_search(self.x_t, d_t, grad, gamma_max)
        self.print("step_type:", step_type, "γ_max:", gamma_max, "γ=", gamma, "|d_t|=", linalg.norm(d_t), level = 2)
        self.active_set.weights[v_t_setIndex] = weight_v_t - gamma
        if (abs(self.active_set.weights[v_t_setIndex]) <= self.zero_threshold):
            self.active_set.drop(v_t_setIndex)
        if (s_t_index in self.active_set.indexes):
            i, = np.where(self.active_set.indexes == s_t_index)
            setIndex = i[0]
            self.active_set.weights[setIndex] += gamma
        else :
            self.active_set.indexes = np.hstack((self.active_set.indexes, s_t_index))
            self.active_set.matrix = np.vstack((self.active_set.matrix,s_t))
            self.active_set.weights = np.hstack((self.active_set.weights, gamma))
            if (self.active_set.size() > (self.dim+1)):
                self.print(" Performing caratheodory", level=3)
                self.active_set = caratheodory(self.active_set)
                self.print(" - λ_%d"%t, linalg.norm(self.x_t - self.active_set.value()), level=4)
        diff = gamma * linalg.norm(d_t)
        return g_FW

class FW_AwayStep_Classic_Proto(FW_Base_Algorithm):
    def iteration(self, grad):
        # s_t: vertex given by linear minimizer
        s_t_index, s_t = self.linear_minimizer(grad)
        s_t = np.array(s_t).ravel()
        d_FW = s_t - self.x_t
        # s_t: vertex given by active set
        v_t_setIndex = np.argmax(self.active_set.matrix @ grad)
        v_t_vertexIndex = self.active_set.indexes[v_t_setIndex]
        v_t = self.active_set.matrix[v_t_setIndex]
        d_A = self.x_t - v_t
        g_FW = -grad.T @ d_FW
        g_A = -grad.T @ d_A
        self.log("traceAS", g_A)
        if g_FW <= self.epsilon:
            return g_FW
        if (g_FW >= g_A):
            step_type = "FW"
            d_t = d_FW
            gamma_max = 1
        else:
            step_type = "A"
            d_t = d_A
            weight_v_t = self.active_set.weights[v_t_setIndex]
            gamma_max = weight_v_t / (1 - weight_v_t)
        gamma = self.do_line_search(self.x_t, d_t, grad, gamma_max)
        self.print("step_type:", step_type, "γ_max:", gamma_max, "γ=", gamma, "|d_t|=", linalg.norm(d_t), level = 2)
        if (step_type == "A"):
            self.active_set.weights = (1 + gamma) * self.active_set.weights
            self.active_set.weights[v_t_setIndex] = weight_v_t - gamma * (1 - weight_v_t)
        else:
            self.active_set.weights = (1-gamma) * self.active_set.weights
            if (s_t_index in self.active_set.indexes):
                i, = np.where(self.active_set.indexes == s_t_index)
                setIndex = i[0]
                self.active_set.weights[setIndex] += gamma
            else:
                if (gamma == 1):
                    self.active_set.indexes[0] = s_t_index
                    self.active_set.matrix[0] = s_t
                    self.active_set.weights[0] = 1
                else:
                    self.active_set.indexes = np.hstack((self.active_set.indexes, s_t_index))
                    self.active_set.matrix = np.vstack((self.active_set.matrix, s_t))
                    self.active_set.weights = np.hstack((self.active_set.weights, gamma))
                    if (self.active_set.size() > (self.dim + 1)):
                        self.active_set = caratheodory(self.active_set)
        diff = gamma * linalg.norm(d_t)
        # log diff
        return g_FW

class FW_AwayStep_ClassicScipy_Algorithm(FW_AwayStep_Classic_Proto, LineSearch_Scipy):
    label = "Classic AwayStep FW w/ Scipy"
    pass
class FW_Pairwise_ClassicScipy_Algorithm(FW_Pairwise_Classic_Proto, LineSearch_Scipy):
    label = "Classic Pairwise FW w/ Scipy"
    pass

class FW_AwayStep_ClassicArmijo_Algorithm(FW_AwayStep_Classic_Proto, LineSearch_Armijo):
    label = "Classic AwayStep FW w/ Armijo"
    pass
class FW_Pairwise_ClassicArmijo_Algorithm(FW_Pairwise_Classic_Proto, LineSearch_Armijo):
    label = "Classic Pairwise FW w/ Armijo"
    pass

class FW_AwayStep_ClassicLipschizStep_Algorithm(FW_AwayStep_Classic_Proto, LineSearch_LipscitzBased):
    label = "Classic AwayStep FW w/ Lipschiz Step"
    pass
class FW_Pairwise_ClassicLipschizStep_Algorithm(FW_Pairwise_Classic_Proto, LineSearch_LipscitzBased):
    label = "Classic Pairwise FW w/ Lipschiz Step"
    pass