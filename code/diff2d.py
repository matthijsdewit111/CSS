# Coen Lenting

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from neuronal_tree import Tree

t1 = time.time()


class DLA_diff2d():
    # the diffusion class. Owns a DLA_diff3d.c which is the matrix with all the information
    def __init__(self, seed, eps=10**-8, x = 20, y = 20, w=1, eta=1):
        self.x = x
        self.y = y
        self.dx = 1/x

        # initialize the space to a gradient from 1 to 0
        self.c = np.zeros((x, y))
        for i in range(x):
            self.c[i] = [1 - j/(y - 1) for j in range(y)]
        self.w = w
        self.eta = eta
        self.eps = eps
        self.converged = False 

        self.tree = Tree(seed, bounds = [[0, x], [0, y]])


    # compute the neighbours in x and z direction, accounting for periodic boundaries
    def neighbour2d(self, x, y):
        if x + 1 < self.x - 1:
            xp = x + 1
        else:
            xp = 0

        if x - 1 < 0:
            xm = self.x - 1 
        else:
            xm = x - 1

        return xp, xm

    # compute the new value for the point with Successive Over Relaxation
    def SOR(self, x, y):
        xp, xm= self.neighbour2d(x, y)
        return self.w/4 * (self.c[x][y + 1] + self.c[x][y - 1] + self.c[xp][y] + self.c[xm][y] + ((1 - self.w) * self.c[x][y]))

    def update(self):
        # update the matrix
        deltamax = 0
        for j in range(1, self.y - 1):
            for i in range(self.x):

                if [i, j] not in self.tree:

                    # save values to compute the change in value for one iteration
                    original_val = self.c[i][j]
                    new_val = self.SOR(i, j)
  
                    if new_val > 0:
                        self.c[i][j] = new_val
                    else:
                        self.c[i][j] = 0

                    delta = abs(new_val - original_val)
                    if delta > deltamax:
                        deltamax = delta
                else:
                    self.c[i][j] = 0

            if deltamax < self.eps:
                    self.converged = True


    # take a growth step
    def growth(self, creation_time):
        # variables to determine to which point is being grown
        c_sum = 0
        combined_c = 0
        rndm = random.random()
        candidates = self.tree.growth_candidates()

        # compute total concentration
        for i, j in candidates:
            c_sum += self.c[i][j] ** self.eta

        # compute the growth site concentrations
        for i, j in candidates:
            combined_c += (self.c[i][j] ** self.eta)
            if (combined_c / c_sum) > rndm:

                self.tree.add([i, j], creation_time)

                break
        self.converged = False

# parameter that controls the shape of the cluster. Higher -> more stretched out
eta = 1
x, y = [50, 80]


dla_diffusion = DLA_diff2d(seed=[x//2, y - 1], x = x, y = y, eta=eta, w = 1)
while dla_diffusion.converged == False:
    dla_diffusion.update()

for t in range(200):
    if t % 10 == 0:
        print(t)
    dla_diffusion.growth(t + 1)

    while (dla_diffusion.converged == False):
        dla_diffusion.update()


t2 = time.time()
print(t2-t1, "TIME")
for i in range(x):
    for j in range(y):
        if [i, j] in dla_diffusion.tree:
            dla_diffusion.c[i][j] = float('nan')

plt.imshow(np.transpose(dla_diffusion.c)[::-1])
dla_diffusion.tree.plot()



plt.show()
