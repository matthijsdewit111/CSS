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
from neuronal_tree import Tree

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
        self.tree = Tree(seed, bounds=[[0, N], [0, N], [0, N]])

    def neighbouring_cluster(self, x, y, z):
        neighbours = {
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1)
        }
        neigh_clust = []
        for neighbour in neighbours:
            # print("neigh", neighbour, self.cluster_test._coords_list)
            if list(neighbour) in self.tree:
                neigh_clust.append(neighbour)

        return neigh_clust

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
    def walker(self, p_stick, creation_time):

        # create walker
        rndm1 = random.randrange(self.N)
        rndm2 = random.randrange(self.N)
        walker_p = [rndm2, 0, rndm1]

        self.growth_candidates()
        self.walking = True
        no_match = True
        check_stick = True

        # while not sticking
        while no_match == True:
            rndm_direc = random.randrange(6)

            # take step in direction UP
            if rndm_direc == 0:
                if walker_p[0] - 1 >= 0:
                    if self.check_move(tuple([walker_p[0] - 1, walker_p[1], walker_p[2]])):
                        walker_p= [walker_p[0] - 1, walker_p[1], walker_p[2]]
                    else:
                        check_stick = False
                else:
                    rndm = random.randrange(self.N)
                    walker_p = [0, rndm, walker_p[2]] # walker 2 of 0?

            # take step in direction RIGHT
            elif rndm_direc == 1:
                if (walker_p[1] + 1) > (self.N - 1):
                    if self.check_move(tuple([walker_p[0], walker_p[0], walker_p[2]])):
                        walker_p = [walker_p[0], 0, walker_p[2]]
                    else:
                        check_stick = False
                else:
                    if self.check_move(tuple([walker_p[0], walker_p[1] + 1, walker_p[2]])):
                        walker_p = [walker_p[0], walker_p[1] + 1, walker_p[2]]

            # take step in direction DOWN
            elif rndm_direc == 2:
                if walker_p[0] + 1 <= (self.N - 1):
                    if self.check_move(tuple([walker_p[0] + 1, walker_p[1], walker_p[2]])):
                        walker_p = [walker_p[0] + 1, walker_p[1], walker_p[2]]
                    else:
                        check_stick = False
                else:
                    rndm = random.randrange(self.N)
                    walker_p = [rndm, 0, rndm]

            # take step in direction LEFT
            elif rndm_direc == 3:
                if (walker_p[1] - 1) < 0:
                    if self.check_move(tuple([walker_p[0], self.N - 1, walker_p[2]])):
                        walker_p= [walker_p[0], self.N - 1, walker_p[2]]
                    else:
                        check_stick = False
                else:
                    if self.check_move(tuple([walker_p[0], walker_p[1] - 1, walker_p[2]])):
                        walker_p = [walker_p[0], walker_p[1] - 1, walker_p[2]]
                    else:
                        check_stick = False

            # take step in direction FRONT
            elif rndm_direc == 5:
                if walker_p[2] - 1 >= 0:
                    if self.check_move(tuple([walker_p[0], walker_p[1], walker_p[2] - 1])):
                        walker_p= [walker_p[0], walker_p[1], walker_p[2] - 1]
                    else:
                        check_stick = False
                else:
                    rndm = random.randrange(self.N)
                    walker_p = [0, rndm, walker_p[2]]

            # take step in direction BACK
            elif rndm_direc == 4:
                if (walker_p[2] + 1) > (self.N - 1):
                    if self.check_move(tuple([walker_p[0], walker_p[1], walker_p[2]])):
                        walker_p = [walker_p[0], 0, walker_p[2]]
                    else:
                        check_stick = False
                else:
                    if self.check_move(tuple([walker_p[0], walker_p[1], walker_p[2] + 1])):
                        walker_p = [walker_p[0], walker_p[1], walker_p[2] + 1]

            if check_stick == True:
                for i, j, k in self.candidates:
                    if i == walker_p[0] and j == walker_p[1] and k == walker_p[2]:
                        x = random.random()
                        if x > p_stick:
                            self.clusters.add(tuple([i, j, k]))
                            self.cluster[i][j][k] = 1

                            self.tree.add([i, j, k], creation_time)

                            no_match = False
                            self.walking = False
                            break

            check_stick = True


N = 50

# controls the chance of the random walker sticking to the cluster
# higher means lower chance
p_stick = 0.5
# fig, axs = plt.subplots(1, 1)

c = C(seed = [N//2, N // 2, N//2], N = N)

# number of points
for i in tqdm(range(250)):
    while (c.walking == True):
        c.walker(p_stick, i + 1)
    c.walking = True

# cluster
for i in range(N):
    for j in range(N):
        for k in range(N):
            if c.cluster[i][j][k] == 1:
                c.c[i][j][k] = float('nan')

c.tree._plot3d()

## 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for list_of_point in c.clusters:
    ax.scatter(list_of_point[0], list_of_point[1], list_of_point[2])

ax.axes.set_xlim3d(left=0, right=N)
ax.axes.set_ylim3d(bottom=0, top=N)
ax.axes.set_zlim3d(bottom=0, top=N)
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
axs.set_title("P stick : {}".format(p_stick))
axs.set_xlabel("x position [-]")
axs.set_ylabel("y position [-]")
plt.show()

t2 = time.time()
print(t2-t1, "TIME")
