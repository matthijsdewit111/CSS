import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neuronal_tree import Tree


class randomwalk_2D():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, x = 50, y = 50, PS = 40):
        self.x = x
        self.y = y
        self.dx = 1 / x
        self.walking = True
        self.tree = Tree(seed, bounds=[[0, x], [0, y]], PS = PS)

        # initialize the space to a gradient from 1 to 0
        self.c = np.zeros((x, y))
        for i in range(x):
            self.c[i] = [1 - j/(y - 1) for j in range(y)]

        self.transformations = {
            0: self.up,
            1: self.down,
            2: self.left,
            3: self.right,
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

    # walk untill stuck to a candidate
    def walker(self, p_stick, creation_time):

        # create list of walkers
        walker_p = []
        for w in range(1):
            rndm = random.randrange(self.x)
            walker = [rndm, 0]
            walker_p.append(walker)

        candidates = self.tree.growth_candidates()

        self.walking = True
        no_match = True
        check_stick = True

        # while not sticking
        while no_match == True:

            for w in range(len(walker_p)):

                rndm_direc = random.randrange(4)

                new_walker_p = self.transformations[rndm_direc](walker_p[w])

                if self.check_move(new_walker_p):
                    walker_p[w] = new_walker_p
                else:
                    check_stick = False

                if check_stick == True:
                    for i, j in candidates:
                        if [i, j] == walker_p[w]:
                            x = random.random()
                            if x > p_stick:
                                self.tree.add([i, j], creation_time)
                                no_match = False
                                self.walking = False
                                break

                check_stick = True


if __name__ == "__main__":
    t1 = time.time()

    x, y = [40, 80]

    # controls the chance of the random walker sticking to the cluster
    # higher means lower chance
    p_stick = 0

    PS = 30

    DLA = randomwalk_2D(seed=[x // 2, y - 1], x = x, y = y, PS = PS)

    # number of points
    for i in tqdm(range(50)):
        while (DLA.walking == True):
            DLA.walker(p_stick, i + 1)
        DLA.walking = True

    t2 = time.time()
    print(t2-t1, "TIME")

    # convert data for plotting
    for i in range(x):
        for j in range(y):
            if [i, j] in DLA.tree:
                DLA.c[i][j] = float('nan')

    # 2D plotting
    DLA.tree.plot()
    plt.show()
