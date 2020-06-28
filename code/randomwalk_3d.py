import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neuronal_tree import Tree


class randomwalk_3D():
    """
    The random walker DLA model. Owns an attribute c which is the matrix with all the information.
    """
    def __init__(self, seed, x = 50, y = 50, z = 50, PS = 40):
        self.x = x
        self.y = y
        self.z = z
        self.dx = 1 / x
        self.walking = True
        self.tree = Tree(seed, bounds = [[0, x], [0, y], [0, z]], PS = PS)

        # initialize the space to a gradient from 1 to 0
        self.c = np.zeros((x, y, z))
        for i in range(x):
            for j in range(y):
                self.c[i][j] = [1 - j / (y - 1) for k in range(z)]

        self.transformations = {
            0: self.up,
            1: self.down,
            2: self.left,
            3: self.right,
            4: self.forward,
            5: self.backward
        }

    # check if the move doesn't go in the cluster
    def check_move(self, coord):
        return coord not in self.tree

    # assume tree is growing downwards
    def up(self, walker_p):
        # Y + 1
        new_walker_p = walker_p.copy()
        new_walker_p[1] += 1

        if new_walker_p[1] > self.y - 1:
            # respawn at Y=0 when moving past Y_max
            new_walker_p[1] = 0

        return new_walker_p

    def down(self, walker_p):
        # Y - 1
        new_walker_p = walker_p.copy()
        new_walker_p[1] -= 1

        if new_walker_p[1] < 0:
            # dont go lower than Y=0
            new_walker_p[1] = 0

        return new_walker_p

    def left(self, walker_p):
        # X - 1
        new_walker_p = walker_p.copy()
        new_walker_p[0] -= 1

        new_walker_p = self.tree.boundaries(new_walker_p)

        return new_walker_p

    def right(self, walker_p):
        # X + 1
        new_walker_p = walker_p.copy()
        new_walker_p[0] += 1

        new_walker_p = self.tree.boundaries(new_walker_p)

        return new_walker_p

    def forward(self, walker_p):
        # Z + 1
        new_walker_p = walker_p.copy()
        new_walker_p[2] += 1

        new_walker_p = self.tree.boundaries(new_walker_p)

        return new_walker_p

    def backward(self, walker_p):
        # Z - 1
        new_walker_p = walker_p.copy()
        new_walker_p[2] -= 1

        new_walker_p = self.tree.boundaries(new_walker_p)

        return new_walker_p

    # walk untill stuck to a candidate

    def walker(self, p_stick, creation_time):

        # create list of walkers
        walker_p = []
        for w in range(150):
            rndm1 = random.randrange(self.x)
            rndm2 = random.randrange(self.z)
            walker = [rndm2, 0, rndm1]
            walker_p.append(walker)

        candidates = self.tree.growth_candidates()

        self.walking = True
        no_match = True
        check_stick = True

        # while not sticking
        while no_match == True:

            for w in range(len(walker_p)):
                # print(walker_p[w])
                rndm_direc = random.randrange(6)

                new_walker_p = self.transformations[rndm_direc](walker_p[w])

                if self.check_move(new_walker_p):
                    walker_p[w] = new_walker_p
                else:
                    check_stick = False

                if check_stick == True:
                    for i, j, k in candidates:
                        if [i, j, k] == walker_p[w]:
                            x = random.random()
                            if x > p_stick:
                                self.tree.add([i, j, k], creation_time)
                                candidates = self.tree.growth_candidates()
                                no_match = False
                                self.walking = False
                                break

                check_stick = True


if __name__ == "__main__":

    t1 = time.time()

    x, y, z = [40, 60, 40]

    # controls the chance of the random walker sticking to the cluster
    # higher means lower chance
    p_stick = 0

    c = randomwalk_3D(seed=[x // 2, y - 1, z // 2], x = x, y = y, z = z)

    # number of points
    for i in tqdm(range(40)):
        while (c.walking == True):
            c.walker(p_stick, i + 1)
        c.walking = True

    # cluster
    for node in c.tree:
        i, j, k = node.coords
        c.c[i][j][k] = float('nan')

    t2 = time.time()
    print(t2 - t1, "TIME")

    c.tree._plot3d()
