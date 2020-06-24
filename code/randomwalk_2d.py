import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neuronal_tree import Tree


class C():
    # the diffusion class. Owns a C.c which is the matrix with all the information
    def __init__(self, seed, N):
        self.N = N
        self.dx = 1/N
        self.c = [[[0 for i in range(N)] for j in range(N)]]
        self.walking = True
        self.tree = Tree(seed, bounds=[[0, N], [0, N]])

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

        if new_walker_p[1] > self.N - 1:
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
            rndm = random.randrange(self.N)
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
    
    N = 100

    # controls the chance of the random walker sticking to the cluster
    # higher means lower chance
    p_stick = 0
    # fig, axs = plt.subplots(1, 1)

    c = C(seed=[N//2, N - 1], N=N)

    # number of points
    for i in tqdm(range(500)):
        while (c.walking == True):
            c.walker(p_stick, i + 1)
        c.walking = True

    t2 = time.time()
    print(t2-t1, "TIME")

    # 2D plotting
    c_list = [[0 for i in range(N)] for j in range(N)]

    for node in c.tree:
        i, j = node.coords
        c_list[i][j] = float(1)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(c_list, cmap='cubehelix')
    axs.set_title("P stick : {}".format(p_stick))
    axs.set_xlabel("x position [-]")
    axs.set_ylabel("y position [-]")
    plt.show()
