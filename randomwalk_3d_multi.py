import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
import pickle
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

t1 = time.time()

class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, N):
        self.N = N
        self.dx = 1/N
        self.c = [[[0 for i in range(N)] for j in range(N)] for k in range(N)]
        self.cluster = [[[0 for i in range(N)] for j in range(N)] for k in range(N)] # 0 is no cluster 1 is
        self.cluster[seed[2]][seed[1]][seed[0]] = 1
        self.clusters = {tuple([seed[2], seed[1], seed[0]])}
        self.candidates = {tuple([seed[2], seed[1], seed[0]])} # (y, x) from top left
        self.walking = True

    # compute the growth candidates
    def growth_candidates(self):
        g_candidates = set()
        self.candidates = self.clusters
        for i, j, k in self.candidates:
            if (j - 1 >= 0):
                g_candidates.add(tuple([i, j - 1, k]))
            if (j + 1 <= (self.N - 1)):
                g_candidates.add(tuple([i, j + 1, k]))
            if (i - 1 >= 0):
                g_candidates.add(tuple([i - 1, j, k]))
            if (i + 1 <= (self.N - 1)):
                g_candidates.add(tuple([i + 1, j, k]))
            if (k - 1 >= 0):
                g_candidates.add(tuple([i, j, k - 1]))
            if (k + 1 <= (self.N - 1)):
                g_candidates.add(tuple([i, j, k + 1]))

        self.candidates = self.candidates | g_candidates

    # check if the move doesn't go in the cluster
    def check_move(self, set):
        if set in self.cluster:
            return False
        else:
            return True

    # walk untill stuck to a candidate
    def walker(self, p_stick):
        # N = 1500
        # R = (np.random.rand(N)*6).astype("int")
        # x = np.zeros(N)
        # y = np.zeros(N)
        # z = np.zeros(N)
        # x[ R==0 ] = -1; x[ R==1 ] = 1 #assigning the axis for each variable to use
        # y[ R==2 ] = -1; y[ R==3 ] = 1
        # z[ R==4 ] = -1; z[ R==5 ] = 1
        # x = np.cumsum(x) #The cumsum() function is used to get cumulative sum over a DataFrame or Series axis i.e. it sums the steps across for eachaxis of the plane.
        # y = np.cumsum(y)
        # z = np.cumsum(z)

        # release at random x-position
        # walker_p = [x[0], y[0], z[0]]


        # create list of walkers
        walker_p = []
        for w in range(100):
            rndm1 = random.randrange(self.N)
            rndm2 = random.randrange(self.N)
            walker = [rndm2, 0, rndm1]
            walker_p.append(walker)

        self.growth_candidates()
        self.walking = True
        no_match = True
        check_stick = True
        attempt = 1


        # while not sticking
        while no_match == True:

            for w in range(len(walker_p)):
                # print("attempt ", attempt)
                attempt += 1
                rndm_direc = random.randrange(6)

                # take step in direction UP
                if rndm_direc == 0:
                    if walker_p[w][0] - 1 >= 0:
                        if self.check_move(tuple([walker_p[w][0] - 1, walker_p[w][1], walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0] - 1, walker_p[w][1], walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [0, rndm, walker_p[w][2]] # walker 2 of 0?

                # take step in direction RIGHT
                elif rndm_direc == 1:
                    if (walker_p[w][1] + 1) > (self.N - 1):
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][0], walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0], 0, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1] + 1, walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] + 1, walker_p[w][2]]

                # take step in direction DOWN
                elif rndm_direc == 2:
                    if walker_p[w][0] + 1 <= (self.N - 1):
                        if self.check_move(tuple([walker_p[w][0] + 1, walker_p[w][1], walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0] + 1, walker_p[w][1], walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [rndm, 0, rndm]

                # take step in direction LEFT
                elif rndm_direc == 3:
                    if (walker_p[w][1] - 1) < 0:
                        if self.check_move(tuple([walker_p[w][0], self.N - 1, walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0], self.N - 1, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1] - 1, walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] - 1, walker_p[w][2]]
                        else:
                            check_stick = False

                # take step in direction FRONT
                elif rndm_direc == 5:
                    if walker_p[w][2] - 1 >= 0:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [0, rndm, walker_p[w][2]]
                    # else:
                    #     if self.check_move(tuple([walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1])):
                    #         walker_p = [walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1]
                    #     else:
                    #         check_stick = False

                # take step in direction BACK
                elif rndm_direc == 4:
                    if (walker_p[w][2] + 1) > (self.N - 1):
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1], walker_p[w][2]])):
                            walker_p[w] = [walker_p[w][0], 0, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1], walker_p[w][2] + 1])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1], walker_p[w][2] + 1]

                if check_stick == True:
                    attempt = 1
                    for i, j, k in self.candidates:
                        if i == walker_p[w][0] and j == walker_p[w][1] and k == walker_p[w][2]:
                            x = random.random()
                            if x > p_stick:
                                self.clusters.add(tuple([i, j, k]))
                                self.cluster[i][j][k] = 1
                                no_match = False
                                self.walking = False
                                break

                check_stick = True




N = 70

# controls the chance of the random walker sticking to the cluster
# higher means lower chance
p_stick = 0.2
c = C(seed = [N//2, N//2, N - 1], N = N)

# number of points
for i in tqdm(range(250)):
    while (c.walking == True):
        c.walker(p_stick)
    c.walking = True

for i in range(N):
    for j in range(N):
        for k in range(N):
            if c.cluster[i][j][k] == 1:
                c.c[i][j][k] = float('nan')

## 3D plotting
plt.figure()
ax = plt.gca(projection='3d')
for list_of_point in c.clusters:
    ax.scatter(list_of_point[0],list_of_point[1],list_of_point[2])
plt.show()

# 2D plotting
c_list =  [[0 for i in range(N)] for j in range(N)]

for i in range(N):
    for j in range(N):
        for k in range(N):
            if c.cluster[i][j][k] == 1:
                c_list[i][j] = float('nan')

fig, axs = plt.subplots(1, 1)
axs.imshow(c_list, cmap = 'cubehelix')
# axs.set_title("P stick : {}".format(p_stick))
axs.set_xlabel("x position [-]")
axs.set_ylabel("y position [-]")
plt.show()

t2 = time.time()
print(t2-t1, "TIME")
