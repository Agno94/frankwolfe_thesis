import fw_algorithms as fw
import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import line_search
from scipy import sparse, linalg
from datetime import datetime
from time import process_time
import sys, re, glob

class FW_Base_SSC_Algorithm(fw.FW_Base_Algorithm):
    def __init__(self, multi_step_max_iter = 1000, tweak_factor = 1, **kwargs):
        self.multi_step_max_iter = multi_step_max_iter
        self.tweak_factor = tweak_factor
        super().__init__(**kwargs)
    def get_auxiliary_setsize(self, g, d, n=0):
        n = n or linalg.norm(d)
        if (n != linalg.norm(d)):
            print(" !! Error")
        if (n < self.zero_threshold**2):
            return 0
        L = self.lipschitz
        y, x = self.y_j, self.x_t
        if linalg.norm(x-y):
            s = y - x
            norm_of_s = linalg.norm(s)
            gap = g.T @ d
            # stop if norm(s) >= (g @ (d / norm(d))) / L
            if (norm_of_s * n * L > gap):
                return 0
            #yc = y - x - 0.5 * g / L
            y2center = x + g / 2 / L - y
            r = linalg.norm(g / 2 / L)
            #p1 = n**2 * (r**2 - yc @ yc)
            #p2 = (d @ yc)**2
            #bound1 = ( - d @ yc + ( p2 + p1 )**0.5 ) / n**2
            #p2 = d @ yc
            #bound1 = ( -p2 + ( p2**2 + p1 )**0.5 ) / n**2
            p1 = y2center.T @ d
            p2 = n**2 * (r**2 - y2center.T @ y2center)
            bound1 = ( p1 + (p1**2 + p2)**0.5 ) / n**2
            # BOUND1: Center x + g/2L and radius |g|/2L
            if (bound1 != bound1):
                print("   bound1 calc: p1, p2 =", p1, p2, "n=", n, "- d @ yc =", - d @ yc , "len = ", len(self.path))
            #p1 = d.T @ s
            #p2 = (g.T @ d)**2 / L**2 - n**2 * s.T @ s
            p1 = - s.T @ d
            p2 = ( gap**2 / L**2 - n**2 * norm_of_s**2)
            #p2 = (g.T @ d)**2 / L**2 - n**2 * s.T @ s
            bound2 = ( p1 + (p1**2 + p2)**0.5 ) / n**2
            # BOUND2: Center x and radius g@d/2nL
        else:
            bound2 = (d.T @ g) / L / n**2
            bound1 = bound2
        self.log("bound ratio", bound2/bound1)
        self.log("bound1", bound1)
        self.log("bound2", bound2)
        beta = min(bound1, bound2)
        new_y = y + beta * d
        return min(bound1, bound2)
    def short_step_chain(self, g, e_FW_index, e_FW):
        self.e_j = self.active_set.copy()
        self.y_j = self.e_j.value()
        for j in range(0, self.multi_step_max_iter):
            r = self.small_step(g, e_FW_index, e_FW)
            if (r):
                self.print("Terminating multistep loop with code %d"%r, level=3)
                self.log("MultiStepTermination", r)
                break
        self.log("MultiStepsNumber", j+1)
        if (j == self.multi_step_max_iter - 1):
            self.print("SSCMultistepProcedure: max number of iterations reached", level=2)
    def iteration(self, grad):
        e_FW_index, e_FW = self.linear_minimizer(grad)
        gap = -grad.T @ (e_FW - self.x_t)
        if gap <= self.epsilon:
            #if (not gap):
            return gap
        self.short_step_chain(-grad, e_FW_index, e_FW)
        self.active_set = self.e_j
        del self.e_j
        del self.y_j
        return gap

class FW_AwayStep_SSC_Algorithm(FW_Base_SSC_Algorithm):
    label = "SSC AwayStep FW"
    def small_step(self, g, e_FW_index, e_FW):
        e_Away_setIndex = np.argmax(self.e_j.matrix @ -g)
        e_Away_vertexIndex = self.e_j.indexes[e_Away_setIndex]
        e_Away = self.e_j.matrix[e_Away_setIndex]
        d_FW = e_FW - self.y_j
        n_FW = linalg.norm(d_FW)
        d_Away = self.y_j - e_Away
        n_Away = linalg.norm(d_Away)
        if ((n_Away * g.T @ d_FW) >= (n_FW * g.T @ d_Away)): #FW vs Away criterion
            step_type = "FW"
            d_j, n_j = d_FW, n_FW
            gammamax = 1
        else:
            step_type = "Away"
            d_j, n_j = d_Away, n_Away
            gammamax = self.e_j.weights[e_Away_setIndex]/(1 - self.e_j.weights[e_Away_setIndex])
        gamma_bound = self.get_auxiliary_setsize(g, d_j, n_j)
        if (gamma_bound <= 0):
            return 1
        gamma = min(gammamax, gamma_bound)
        if (step_type == "Away"):
            alpha_t = self.e_j.weights[e_Away_setIndex]
            self.e_j.weights = (1 + gamma) * self.e_j.weights
            self.e_j.weights[e_Away_setIndex] = alpha_t - gamma * (1 - alpha_t)
        else:
            self.e_j.weights = (1-gamma) * self.e_j.weights
            if (e_FW_index in self.e_j.indexes):
                i, = np.where(self.e_j.indexes == e_FW_index)
                setIndex = i[0]
                self.e_j.weights[setIndex] += gamma
            else :
                if (gamma == 1):
                    self.e_j.indexes[0] = e_FW_index
                    self.e_j.matrix[0] = e_FW
                    self.e_j.weights[0] = 1
                else:
                    self.e_j.indexes = np.hstack((self.e_j.indexes, e_FW_index))
                    self.e_j.matrix = np.vstack((self.e_j.matrix, e_FW))
                    self.e_j.weights = np.hstack((self.e_j.weights, gamma))
                    if (self.e_j.size() > (self.dim + 1)):
                        self.e_j = caratheodory(self.e_j)
        zero_weights = np.where(
            np.abs(self.e_j.weights) <= self.zero_threshold/self.e_j.size())
        if zero_weights[0].size:
            self.e_j.drop(zero_weights)
        summ = self.e_j.weights @ np.ones(self.e_j.size())
        self.e_j.weights = self.e_j.weights / summ
        y_jplus1 = self.e_j.value()
        self.y_j = y_jplus1
        return (gamma < gammamax) and 2

class FW_Pairwise_SSC_Algorithm(FW_Base_SSC_Algorithm):
    label = "SSC Pairwise FW"
    def small_step(self, g, e_FW_index, e_FW):
        e_Away_setIndex = np.argmax(self.e_j.matrix @ -g)
        e_Away_vertexIndex = self.e_j.indexes[e_Away_setIndex]
        e_Away = self.e_j.matrix[e_Away_setIndex]
        step_type = "PFW"
        d_j = e_FW - e_Away
        n_j = linalg.norm(d_j)
        if (n_j == 0):
            return 3
        if (e_FW_index in self.e_j.indexes):
            i, = np.where(self.e_j.indexes == e_FW_index)
            setIndex = i[0]
            y_FW_j = self.e_j.weights[setIndex]
        else:
            y_FW_j = 0
        gammamax = min(1 - y_FW_j, self.e_j.weights[e_Away_setIndex])
        gamma_bound = self.get_auxiliary_setsize(g, d_j, n_j)
        if (gamma_bound <= 0):
            return 1
        gamma = min(gammamax, gamma_bound)
        self.e_j.weights[e_Away_setIndex] -= gamma
        if (y_FW_j):
            self.e_j.weights[setIndex] += gamma
        else:
            if (abs(self.e_j.weights[e_Away_setIndex]) <= 1e-12):
                self.e_j.drop(e_Away_setIndex)
            self.e_j.indexes = np.hstack((self.e_j.indexes, e_FW_index))
            self.e_j.matrix = np.vstack((self.e_j.matrix, e_FW))
            self.e_j.weights = np.hstack((self.e_j.weights, gamma))
            if (self.e_j.size() > (self.dim + 1)):
                self.e_j = caratheodory(self.e_j)
        zero_weights = np.where(
            np.abs(self.e_j.weights) <= self.zero_threshold/self.e_j.size())
        if zero_weights[0].size:
            self.e_j.drop(zero_weights)
        summ = self.e_j.weights @ np.ones(self.e_j.size())
        self.e_j.weights = self.e_j.weights / summ
        y_jplus1 = self.e_j.value()
        self.y_j = y_jplus1
        return (gamma < gammamax) and 2
