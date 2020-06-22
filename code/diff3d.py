# Coen Lenting

import random
import time
import matplotlib.pyplot as plt
import numpy as np

from neuronal_tree import Node, Tree

from mpl_toolkits.mplot3d import Axes3D

t1 = time.time()


class DLA_diff3d():
    # the diffusion class. Owns a DLA_diff3d.c which is the matrix with all the information
    def __init__(self, seed, eps=10**-5, x = 20, y = 20, z = 20, w=1, eta=1):
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

        self.candidates = {}  

        self.cluster_test = Tree([x//2, y - 1, z//2], bounds = [[0, x], [0, y], [0, z]])


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
            if list(neighbour) in self.cluster_test:
                neigh_clust.append(neighbour)

        return neigh_clust

    def boundaries(self, coords):
        i, j, k = coords
        new_coords = coords
        if i < 0:
            new_coords =  [self.x - 1, new_coords[1], new_coords[2]]
        if i > self.x - 1:
            new_coords =  [0, new_coords[1], new_coords[2]]
        if k < 0:
            new_coords =  [new_coords[0], new_coords[1], self.z - 1]
        if k > self.z - 1:
            new_coords =  [new_coords[0], new_coords[1], 0]
        
        return new_coords

    def update(self):
        # update the matrix
        deltamax = 0
        for j in range(1, self.y - 1):
            for i in range(self.x):
                for k in range(self.z):
                    if [i, j, k] not in self.cluster_test:

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

    # compute the growth candidates
    def growth_candidates(self):
        # create a set for all possible growth candidates
        final_neighbours = set()

        for node in self.cluster_test._node_list:

            coords = node.coords
            d = 3
            offsets = np.indices((3,) * d) - 1
            reshaped_offsets = np.stack(offsets, axis=d).reshape(-1, d)
            offsets_without_middle_point = np.delete(reshaped_offsets, int(d**3 / 2), axis=0)
            neighbours = offsets_without_middle_point + coords
            neighbours = neighbours.tolist()
            
            for neighbour in neighbours:
                if neighbour not in self.cluster_test._node_list:
                    if 0 < neighbour[1] < self.y - 1:
                        final_neighbours.add(tuple(self.boundaries(neighbour)))

        self.candidates = final_neighbours



    # take a growth step
    def growth(self, creation_time):
        self.growth_candidates()

        # variables to determine to which point is being grown
        c_sum = 0
        combined_c = 0
        rndm = random.random()

        # compute total concentration
        for i, j, k in self.candidates:
            c_sum += self.c[i][j][k] ** self.eta

        # compute the growth site concentrations
        for i, j, k in self.candidates:
            combined_c += (self.c[i][j][k] ** self.eta)
            if (combined_c / c_sum) > rndm:

                self.cluster_test.add([i, j, k], creation_time)

                break
        self.converged = False

# parameter that controls the shape of the cluster. Higher -> more stretched out
eta = 4
x, y, z = [40, 60, 40]


dla_diffusion = DLA_diff3d(seed=[x//2, x - 1], x = x, y = y, z = z, eta=eta, w = 1)
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

dla_diffusion.cluster_test.plot()

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
