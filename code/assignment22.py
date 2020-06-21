# Assignment 2 part 2
# Coen Lenting

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
import pickle

t1 = time.time()


class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, N):
        self.N = N
        self.dx = 1/N
        self.c = [[0 for i in range(N)]for j in range(N)]
        self.cluster = [[0 for i in range(N)] for j in range(N)] # 0 is no cluster 1 is
        self.cluster[seed[1]][seed[0]] = 1
        self.clusters = {tuple([seed[1], seed[0]])}
        self.candidates = {tuple([seed[1], seed[0]])} # (y, x) from top left
        self.walking = True

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

    # check if the move doesn't go in the cluster
    def check_move(self, set):
        if set in self.cluster:
            return False
        else:
            return True

    # walk untill stuck to a candidate
    def walker(self, p_stick):
        rndm = random.randrange(self.N)

        # release at random x-position
        walker_p = [0, rndm]
        self.growth_candidates()
        self.walking = True
        no_match = True
        check_stick = True

        # while not sticking
        while no_match == True:
            print(walker_p)
            rndm_direc = random.randrange(4) # 0 up, 1 right, 2 down, 3 left

            # take step in direction
            if rndm_direc == 0:

                if walker_p[0] - 1 >= 0:
                    if self.check_move(tuple([walker_p[0] - 1, walker_p[1]])):
                        walker_p = [walker_p[0] - 1, walker_p[1]]
                    else:
                        check_stick = False
                else:
                    rndm = random.randrange(self.N)
                    walker_p = [0, rndm]
            elif rndm_direc == 1:
                if (walker_p[1] + 1) > (self.N - 1):
                    if self.check_move(tuple([walker_p[0], walker_p[0]])):
                        walker_p = [walker_p[0], 0]
                    else:
                        check_stick = False
                else:
                    if self.check_move(tuple([walker_p[0], walker_p[1] + 1])):
                        walker_p = [walker_p[0], walker_p[1] + 1]
            elif rndm_direc == 2:
                if walker_p[0] + 1 <= (self.N - 1):
                    if self.check_move(tuple([walker_p[0] + 1, walker_p[1]])):
                        walker_p = [walker_p[0] + 1, walker_p[1]]
                    else:
                        check_stick = False
                else:
                    rndm = random.randrange(self.N)
                    walker_p = [0, rndm]
            else:
                if (walker_p[1] - 1) < 0:
                    if self.check_move(tuple([walker_p[0], self.N - 1])):
                        walker_p = [walker_p[0], self.N - 1]
                    else:
                        check_stick = False
                else:
                    if self.check_move(tuple([walker_p[0], walker_p[1] - 1])):
                        walker_p = [walker_p[0], walker_p[1] - 1]
                    else:
                        check_stick = False

            if check_stick == True:
                for i, j in self.candidates:
                    if i == walker_p[0] and j == walker_p[1]:
                        x = random.random()
                        if x > p_stick:
                            self.clusters.add(tuple([i, j]))
                            self.cluster[i][j] = 1
                            no_match = False
                            self.walking = False
                            break
            check_stick = True


N = 100

# controls the chance of the random walker sticking to the cluster
# higher means lower chance
p_stick = 0.5
fig, axs = plt.subplots(1, 1)

c = C(seed = [N//2, N - 1], N = N)

# number of points
for i in range(25):
    while (c.walking == True):
        c.walker(p_stick)
    c.walking = True
    if i % 10 == 0:
        print(i)


for i in range(N):
    for j in range(N):
        if c.cluster[i][j] == 1:
            c.c[i][j] = float('nan')

axs.imshow(c.c, cmap = 'cubehelix')
axs.set_title("P stick : {}".format(p_stick))
axs.set_xlabel("x position [-]")
axs.set_ylabel("y position [-]")


t2 = time.time()
print(t2-t1, "TIME")
plt.show()
