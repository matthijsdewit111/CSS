# Coen Lenting

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from neuronal_tree import Tree


class DLA_diff3d():
    # the diffusion class. Owns a DLA_diff3d.c which is the matrix with all the information
    def __init__(self, seed, eps=10**-5, x=20, y=20, z=20, w=1, eta=1, PS = 40):
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

        self.tree = Tree(seed, bounds=[[0, x], [0, y], [0, z]], PS = PS)

    # compute the neighbours in x and z direction, accounting for periodic boundaries

    def neighbour2d(self, x, y, z):
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

    # compute the new value for the point with Successive Over Relaxation
    def SOR(self, x, y, z):
        xp, xm, zp, zm = self.neighbour2d(x, y, z)
        return self.w/6 * (self.c[x][y + 1][z] + self.c[x][y - 1][z] + self.c[x][y][zm] + self.c[x][y][zp] + self.c[xp][y][z] + self.c[xm][y][z] + ((1 - self.w) * self.c[x][y][z]))

    def update(self):
        # update the matrix
        deltamax = 0
        for j in range(1, self.y - 1):
            for i in range(self.x):
                for k in range(self.z):
                    if [i, j, k] not in self.tree:

                        # save values to compute the change in value for one iteration
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

    # take a growth step

    def growth(self, creation_time):
        # variables to determine to which point is being grown
        c_sum = 0
        combined_c = 0
        rndm = random.random()
        candidates = self.tree.growth_candidates()

        # compute total concentration
        for i, j, k in candidates:
            c_sum += self.c[i][j][k] ** self.eta

        # compute the growth site concentrations
        for i, j, k in candidates:
            combined_c += (self.c[i][j][k] ** self.eta)
            if (combined_c / c_sum) > rndm:

                self.tree.add([i, j, k], creation_time)

                break
        self.converged = False


# This if statement (you will see it alot in python files)
# prevents any code within it from running when importing the file
# otherwise whenever someone imports this, all this code would run.
# it will now only run this code when you explicitly run this file with 'python diff3d.py'
if __name__ == "__main__":
    t1 = time.time()
    # parameter that controls the shape of the cluster. Higher -> more stretched out
    eta = 4
    x, y, z = [40, 60, 40]

    dla_diffusion = DLA_diff3d(seed=[x//2, y - 1, z//2], x=x, y=y, z=z, eta=eta, w=1)
    while dla_diffusion.converged == False:
        dla_diffusion.update()

    for t in range(150):
        if t % 10 == 0:
            print(t)
        dla_diffusion.growth(t + 1)

        while (dla_diffusion.converged == False):
            dla_diffusion.update()

    # fig, ax = plt.subplots(1, 1)

    # side = np.zeros((y, z))
    # for i in range(x):
    #     for j in range(y):
    #         for k in range(z):
    #             if dla_diffusion.cluster[i][j][k] == 1:
    #                 print(1 - (abs(i - x//2)/x), i, x//2)
    #                 side[j][k] += 1 * (1 - (abs(i - x//2)) / x)
    # ax.imshow(side)

    # plot slices

    t2 = time.time()
    print(t2-t1, "TIME")
    print(dla_diffusion.tree.get_asymmetry_index())

    dla_diffusion.tree.plot()

    # for i in range(x):
    #     for j in range(y):
    #         for k in range(z):
    #             if dla_diffusion.cluster[i][j][k] == 1:
    #                 dla_diffusion.c[i][j][k] = float('nan')

    # fig, ax = plt.subplots(3, 2, figsize = (8, 8), constrained_layout = True)
    # axs = ax.flatten()
    # axs[0].imshow(dla_diffusion.c[x//2])
    # axs[1].imshow(dla_diffusion.c[x//2 + 1])
    # axs[2].imshow(dla_diffusion.c[x//2 + 2])
    # axs[3].imshow(dla_diffusion.c[x//2 + 3])
    # axs[4].imshow(dla_diffusion.c[x//2 - 1])
    # axs[5].imshow(dla_diffusion.c[x//2 - 2])

    plt.show()
