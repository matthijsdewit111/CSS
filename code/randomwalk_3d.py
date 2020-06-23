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
        self.walking = True
        self.tree = Tree(seed, bounds=[[0, N], [0, N], [0, N]])

    # check if the move doesn't go in the cluster
    def check_move(self, coord):
        if coord in self.tree:
            return False
        else:
            return True

    # walk untill stuck to a candidate
    def walker(self, p_stick, creation_time):

        # create list of walkers
        walker_p = []
        for w in range(1):
            rndm1 = random.randrange(self.N)
            rndm2 = random.randrange(self.N)
            rndm3 = random.randrange(self.N)
            walker = [rndm2, 0, rndm1]
            walker_p.append(walker)

        candidates = self.tree.growth_candidates()

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
                        if self.check_move([walker_p[w][0] - 1, walker_p[w][1], walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0] - 1, walker_p[w][1], walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [0, rndm, walker_p[w][2]] # walker 2 of 0?

                # take step in direction RIGHT
                elif rndm_direc == 1:
                    if (walker_p[w][1] + 1) > (self.N - 1):
                        if self.check_move([walker_p[w][0], walker_p[w][0], walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0], 0, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        if self.check_move([walker_p[w][0], walker_p[w][1] + 1, walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] + 1, walker_p[w][2]]

                # take step in direction DOWN
                elif rndm_direc == 2:
                    if walker_p[w][0] + 1 <= (self.N - 1):
                        if self.check_move([walker_p[w][0] + 1, walker_p[w][1], walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0] + 1, walker_p[w][1], walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        rndm = random.randrange(self.N)
                        walker_p[w] = [rndm, 0, rndm]

                # take step in direction LEFT
                elif rndm_direc == 3:
                    if (walker_p[w][1] - 1) < 0:
                        if self.check_move([walker_p[w][0], self.N - 1, walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0], self.N - 1, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        if self.check_move([walker_p[w][0], walker_p[w][1] - 1, walker_p[w][2]]):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1] - 1, walker_p[w][2]]
                        else:
                            check_stick = False

                # take step in direction BACK (Z DOWN)
                elif rndm_direc == 5:
                    if walker_p[w][2] - 1 >= 0:
                        if self.check_move([walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1]):
                            walker_p[w] = [walker_p[w][0], walker_p[w][1], walker_p[w][2] - 1]
                        else:
                            check_stick = False
                    else:
                        walker_p[w] = [walker_p[w][0], walker_p[w][1], self.N - 1]

                # take step in direction FRONT (Z UP)
                elif rndm_direc == 4:
                    if (walker_p[w][2] + 1) > (self.N - 1):
                        if self.check_move([walker_p[w][0], walker_p[w][1], walker_p[w][2] + 1]):
                            walker_p[w] = [walker_p[w][0], 0, walker_p[w][2]]
                        else:
                            check_stick = False
                    else:
                        walker_p[w] = [walker_p[w][0], walker_p[w][1], 0]

                if check_stick == True:
                    for i, j, k in candidates:
                        if [i, j, k] == walker_p[w]:
                            x = random.random()
                            if x > p_stick:
                                self.tree.add([i, j, k], creation_time)
                                no_match = False
                                self.walking = False
                                break

                check_stick = True


N = 50

# controls the chance of the random walker sticking to the cluster
# higher means lower chance
p_stick = 0
# fig, axs = plt.subplots(1, 1)

c = C(seed = [N//2, N - 1, N//2], N = N)

# number of points
for i in tqdm(range(600)):
    while (c.walking == True):
        c.walker(p_stick, i + 1)
    c.walking = True

# cluster
for node in c.tree:
    i, j, k = node.coords
    c.c[i][j][k] = float('nan')

c.tree._plot3d()

## 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for node in c.tree:
    list_of_point = node.coords
    ax.scatter(list_of_point[0], list_of_point[1], list_of_point[2])

ax.axes.set_xlim3d(left=0, right=N)
ax.axes.set_ylim3d(bottom=0, top=N)
ax.axes.set_zlim3d(bottom=0, top=N)
plt.show()

# 2D plotting
c_list =  [[0 for i in range(N)] for j in range(N)]

for node in c.tree:
    i, j, k = node.coords
    c_list[i][j] = float('nan')

fig, axs = plt.subplots(1, 1)
axs.imshow(c_list, cmap = 'cubehelix')
axs.set_title("P stick : {}".format(p_stick))
axs.set_xlabel("x position [-]")
axs.set_ylabel("y position [-]")
plt.show()

t2 = time.time()
print(t2-t1, "TIME")
