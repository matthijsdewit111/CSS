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

from neuronal_tree import Tree

t1 = time.time()


class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, N):
        self.N = N
        self.dx = 1 / N
        self.c = [[[0 for i in range(N)] for j in range(N)] for k in range(N)]
        self.cluster = [[[0 for i in range(N)] for j in range(N)] for k in range(N)] # 0 is no cluster 1 is
        self.cluster[seed[0]][seed[1]]] = 1
        self.clusters = {tuple([seed[0], seed[1]])}
        self.candidates = {tuple([seed[0], seed[1]])} # (y, x) from top left
        self.walking = True
        self.tree = Tree(seed, bounds=[[0, N], [0, N]]])

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


    def boundaries(self, coords):
        i, j, k = coords
        new_coords = coords
        if i < 0:
            new_coords =  [self.N - 1, new_coords[1], new_coords[2]]
        if i > self.N - 1:
            new_coords =  [0, new_coords[1], new_coords[2]]
        if k < 0:
            new_coords =  [new_coords[0], new_coords[1], self.N - 1]
        if k > self.N - 1:
            new_coords =  [new_coords[0], new_coords[1], 0]

        return new_coords

    # compute the growth candidates
    def growth_candidates(self):
        # create a set for all possible growth candidates
        # final_neighbours = set()
        #
        # for node in self.tree._node_list:
        #
        #     coords = node.coords
        #     d = 3
        #     offsets = np.indices((3,) * d) - 1
        #     reshaped_offsets = np.stack(offsets, axis=d).reshape(-1, d)
        #     offsets_without_middle_point = np.delete(reshaped_offsets, int(d**3 / 2), axis=0)
        #     neighbours = offsets_without_middle_point + coords
        #     neighbours = neighbours.tolist()
        final_neighbours = set()
        for node in self.tree._node_list:
            coords = node.coords
            neighbours = self.tree._get_neighbours(node.coords)

            for neighbour in neighbours:
                if neighbour not in self.tree._node_list:
                    if 0 < neighbour[1] < self.N - 1:
                        # final_neighbours.add(tuple(self.boundaries(neighbour)))
                        final_neighbours.add(tuple(self.tree.boundaries(neighbour)))

        self.candidates = final_neighbours

    # check if the move doesn't go in the cluster
    def check_move(self, set):
        if set in self.cluster:
            return False
        else:
            return True

    # walk untill stuck to a candidate
    def walker(self, p_stick, creation_time):

        # create list of walkers
        walker_p = []
        for w in range(1):
            rndm1 = random.randrange(self.N)
            walker = [rndm1, 0]
            walker_p.append(walker)
        #
        # # create walker
        # rndm1 = random.randrange(self.N)
        # rndm2 = random.randrange(self.N)
        # walker_p = [rndm2, 0, rndm1]

        self.growth_candidates()
        self.walking = True
        no_match = True
        check_stick = True

        # while not sticking
        while no_match == True:

            for w in range(len(walker_p)):

                rndm_direc = random.randrange(6)

                # take step in direction UP
                if rndm_direc == 0:
                    if walker_p[w][0] - 1 >= 0:
                        if self.check_move(tuple([walker_p[w][0] - 1, walker_p[w][1]])):
                            walker_p[w] = [walker_p[w][0] - 1, walker_p[w][1]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [0, rndm] # walker 2 of 0?

                # take step in direction RIGHT
                elif rndm_direc == 1:
                    if (walker_p[w][1] + 1) > (self.N - 1):
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][0]])):
                            walker_p[w] = [walker_p[w][0], 0]
                        else:
                            check_stick = False
                    else:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1] + 1])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] + 1]

                # take step in direction DOWN
                elif rndm_direc == 2:
                    if walker_p[w][0] + 1 <= (self.N - 1):
                        if self.check_move(tuple([walker_p[w][0] + 1, walker_p[w][1]])):
                            walker_p[w] = [walker_p[w][0] + 1, walker_p[w][1]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [rndm, 0]

                # take step in direction LEFT
                elif rndm_direc == 3:
                    if (walker_p[w][1] - 1) < 0:
                        if self.check_move(tuple([walker_p[w][0], self.N - 1])):
                            walker_p[w] = [walker_p[w][0], self.N - 1]
                        else:
                            check_stick = False
                    else:
                        if self.check_move(tuple([walker_p[w][0], walker_p[w][1] - 1])):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] - 1]
                        else:
                            check_stick = False


                if check_stick == True:
                    for i, j in self.candidates:
                        if i == walker_p[w][0] and j == walker_p[w][1]:
                            x = random.random()
                            if x > p_stick:
                                self.clusters.add(tuple([i, j]))
                                self.cluster[i][j] = 1

                                self.tree.add([i, j], creation_time)

                                no_match = False
                                self.walking = False
                                break

                check_stick = True


N = 50

# controls the chance of the random walker sticking to the cluster
# higher means lower chance
p_stick = 0
# fig, axs = plt.subplots(1, 1)

c = C(seed = [N//2, N - 1], N = N)

# number of points
for i in tqdm(range(600)):
    while (c.walking == True):
        c.walker(p_stick, i + 1)
    c.walking = True

# cluster
for i in range(N):
    for j in range(N):
        for k in range(N):
            if c.cluster[i][j] == 1:
                c.c[i][j] = float('nan')

c.tree._plot3d()

## 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for list_of_point in c.clusters:
    ax.scatter(list_of_point[0], list_of_point[1])

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
