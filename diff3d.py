# Coen Lenting

import random
import time
from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from mpl_toolkits.mplot3d import Axes3D

t1 = time.time()


class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, eps=10**-7, x = 20, y = 20, z = 20, w=1, eta=1):
        self.x = x
        self.y = y
        self.z = z
        self.dx = 1/x

        # initialize the space to a gradient from 1 to 0
        self.c = np.zeros((x, y, z))
        for i in range(x):
            for j in range(y):
                self.c[i][j] = [1 - j/(y - 1) for k in range(z)]
        self.w = w
        self.eta = eta
        self.eps = eps
        self.converged = False
        self.cluster = np.zeros((x, y, z))  # 0 is no cluster 1 is
        self.cluster[x//2][y - 1][z//2] = 1
        self.clusters = {tuple([x//2, y-1, z//2])}
        self.candidates = {tuple([x//2, y-1, z//2])}  # (y, x) from top left




    def neighbours(self, x, y, z):
        if x + 1 < self.x - 1:
            xp = x + 1
        else:
            xp = 0

        if z + 1 < self.z - 1:
            zp = z + 1
        else:
            zp = 0

        if x - 1 < 0:
            xm = self.x - 1 
        else:
            xm = x - 1

        if z - 1 < 0:
            zm = self.z - 1
        else:
            zm = z - 1

        return xp, xm, zp, zm

    # compute the new value for the point
    def SOR(self, x, y, z):
        xp, xm, zp, zm = self.neighbours(x, y, z)
        return self.w/6 * (self.c[x][y + 1][z] + self.c[x][y - 1][z] + self.c[x][y][zm] + self.c[x][y][zp] + self.c[xp][y][z] + self.c[xm][y][z] + ((1 - self.w) * self.c[x][y][z]))

    def update(self):
        # update the matrix
        deltamax = 0
        for j in range(1, self.y - 1):
            for i in range(self.x):
                for k in range(self.z):
                    if self.cluster[i][j][k] == 0:
                        original_val = self.c[i][j][k]
                        new_val = self.SOR(i, j, k)
      
                        if new_val > 0:
                            self.c[i][j][k] = new_val
                        else:
                            self.c[i][j][k] = 0

                        delta = abs(new_val - original_val)
                        if delta > deltamax:
                            deltamax = delta
                    else:
                        self.c[i][j][k] = 0

                if deltamax < self.eps:
                    self.converged = True

    # compute the growth candidates
    def growth_candidates(self):
        g_candidates = set()
        self.candidates = self.clusters
        for i, j, k in self.candidates:
            if (j - 1 >= 0):
                g_candidates.add(tuple([i, j - 1, k]))
            if (j + 1 <= (self.y - 1)):
                g_candidates.add(tuple([i, j + 1, k]))
            if (i - 1 >= 0):
                g_candidates.add(tuple([i - 1, j, k]))
            if (i + 1 <= (self.x - 1)):
                g_candidates.add(tuple([i + 1, j, k]))
            if (k - 1 >= 0):
                g_candidates.add(tuple([i, j, k - 1]))
            if (k + 1 <= (self.z - 1)):
                g_candidates.add(tuple([i, j, k + 1]))
        self.candidates = self.candidates | g_candidates  

    # take a growth step
    def growth(self):
        self.growth_candidates()

        c_sum = 0
        combined_c = 0
        rndm = random.random()

        # compute total concentration
        for i, j, k in self.candidates:
            c_sum += self.c[i][j][k] ** self.eta

        # compute the growth site
        for i, j, k in self.candidates:
            combined_c += (self.c[i][j][k] ** self.eta)
            if (combined_c / c_sum) > rndm:
                self.clusters.add(tuple([i, j, k]))
                self.cluster[i][j][k] = 1
                break
        self.converged = False

# parameter that controls the shape of the cluster. Higher -> more stretched out
eta = 3
x, y, z = [40, 70, 40]


c = C(seed=[x//2, x - 1], x = x, y = y, z = z, eta=eta, w = 1)
while c.converged == False:
    c.update()

for i in range(100):
    if i % 10 == 0:
        print(i)
    c.growth()

    while (c.converged == False):
        c.update()

# for i in range(x):
#     for j in range(y):
#         for k in range(z):
#             if c.cluster[i][j][k] == 1:
#                 c.c[i][j][k] = float('nan')

# fig, ax = plt.subplots(2, 2)
# axs = ax.flatten()
# axs[0].imshow(c.c[x//2])
# axs[1].imshow(c.c[x//2 + 1])
# axs[2].imshow(c.c[x//2 + 2])
# axs[3].imshow(c.c[x//2 - 1])
# plt.show()

fig, ax = plt.subplots(1, 1)

side = np.zeros((y, z))
for i in range(x):
    for j in range(y):
        for k in range(z):
            if c.cluster[i][j][k] == 1:
                print(1 - (abs(i - x//2)/x), i, x//2)
                side[j][k] += 1 * (1 - (abs(i - x//2)) / x)
ax.imshow(side)
# plt.show()



# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, x)
ax.set_ylim(0, y)
ax.set_zlim(0, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
clusters = list(c.clusters)
for x, y, z in clusters:
    ax.scatter(x, y, z)
ax.view_init(0, 180)
plt.show()
# fig.colorbar(im, ax=axs)
# t2 = time.time()
# print(t2-t1, "TIME")
# plt.show()
