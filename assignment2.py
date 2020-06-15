# Assignment 2 part 1
# Coen Lenting

import pickle
import random
import time
from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

t1 = time.time()


class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, eps=10**-14, N=50, D=1, w=1.91, eta=1):
        self.N = N
        self.dx = 1/N
        self.D = D
        self.c = [[0 if (j != 0) else 1 for i in range(N)]for j in range(N)]
        self.w = w
        self.eta = eta
        self.eps = eps
        self.converged = False
        self.cluster = [[0 for i in range(N)] for j in range(N)]  # 0 is no cluster 1 is
        self.cluster[seed[1]][seed[0]] = 1
        self.clusters = {tuple([seed[1], seed[0]])}
        self.candidates = {tuple([seed[1], seed[0]])}  # (y, x) from top left

    def update(self):
        # update the matrix
        deltamax = 0
        for i in range(1, self.N - 1):  # y-axis
            for j in range(self.N):  # x-axis
                if self.cluster[i][j] == 0:
                    if (j - 1) < 0:
                        jm1 = self.N - 1
                        jp1 = j + 1
                    elif (j + 1) > self.N - 1:
                        jp1 = 0
                        jm1 = j - 1
                    else:
                        jm1 = j - 1
                        jp1 = j + 1

                    sparec = self.c[i][j]
                    self.c[i][j] = self.w/4 * (self.c[i + 1][j] + self.c[i-1][j] + self.c[i][jp1] + self.c[i][jm1]) + ((1-self.w) * self.c[i][j])
                    if self.c[i][j] < 0:
                        self.c[i][j] = 0
                    delta = abs(self.c[i][j] - sparec)
                    if delta > deltamax:
                        deltamax = delta
                else:
                    self.c[i][j] = 0

            # check for convergence
            if deltamax < self.eps:
                self.converged = True

    # compute the growth candidates
    def growth_candidates(self):
        g_candidates = set()
        self.candidates = self.clusters
        for i, j in self.candidates:
            if (j - 1 >= 0):
                g_candidates.add(tuple([i, j - 1]))
            if (j + 1 <= (self.N - 1)):
                g_candidates.add(tuple([i, j + 1]))
            if (i - 1 >= 0):
                g_candidates.add(tuple([i - 1, j]))
            if (i + 1 <= (self.N - 1)):
                g_candidates.add(tuple([i + 1, j]))

        self.candidates = self.candidates | g_candidates

    # take a growth step
    def growth(self):

        self.growth_candidates()

        c_sum = 0
        combined_c = 0
        rndm = random.random()

        # compute total concentration
        for i, j in self.candidates:
            c_sum += self.c[i][j] ** self.eta

        # compute the growth site
        for i, j in self.candidates:
            combined_c += (self.c[i][j] ** self.eta)
            if (combined_c / c_sum) > rndm:
                self.clusters.add(tuple([i, j]))
                self.cluster[i][j] = 1
                break
        self.converged = False


N = 100
etas = [0.5, 1, 1.5, 2]
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
cntr = 0

# compute DLA for all eta's
for eta in etas:
    c = C(seed=[N//2, N - 1], N=N, eta=eta)
    while (c.converged == False):
        c.update()
    c.eta = eta

    for i in range(250):
        if i % 10 == 0:
            print(i)
        c.growth()

        while (c.converged == False):
            c.update()

    for i in range(N):
        for j in range(N):
            if c.cluster[i][j] == 1:
                c.c[i][j] = float('nan')

    im = axs[cntr].imshow(c.c)
    axs[cntr].set_title("eta = " + str(eta))
    axs[cntr].set_xlabel("x position [-]")
    axs[cntr].set_ylabel("y position [-]")
    cntr += 1


fig.colorbar(im, ax=axs)
t2 = time.time()
print(t2-t1, "TIME")
plt.show()
