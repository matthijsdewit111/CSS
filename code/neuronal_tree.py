import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Node:
    def __init__(self, coords, creation_time, parent_node):
        self.coords = coords
        self.creation_time = creation_time
        self.parent_node = parent_node
        self.child_nodes = []
        self.is_leaf = True
        self.number_of_leafs = None

        if parent_node is None:
            self.depth = 0
        else:
            self.depth = parent_node.depth + 1

    def add_child(self, coords, creation_time):
        new_node = Node(coords, creation_time, self)
        self.child_nodes.append(new_node)
        self.is_leaf = False
        return new_node

    def remove_child(self, node):
        self.child_nodes.remove(node)
        if len(self.child_nodes) == 0:
            self.is_leaf = True

    def get_nodes_in_terminal_branch(self):
        if self.is_leaf:
            return [self]

        # recurse to get all 'above'
        nodes_in_terminal_branch = []
        for child in self.child_nodes:
            for node in child.get_nodes_in_terminal_branch():
                nodes_in_terminal_branch.append(node)

        if len(self.child_nodes) == 1 and self.child_nodes[0] in nodes_in_terminal_branch:
            return nodes_in_terminal_branch + [self]
        else:
            return nodes_in_terminal_branch


    def get_branch_orders(self, order, orders_list):
        if self.is_leaf:
            return []

        n_children = len(self.child_nodes)

        if n_children == 1:
            if self.parent_node == None:  # root case
                return [0] + self.child_nodes[0].get_branch_orders(order, orders_list)
            return self.child_nodes[0].get_branch_orders(order, orders_list)

        if n_children > 1:
            new_order = order + 1
            new_orders = []
            for child in self.child_nodes:
                new_orders += [new_order] + child.get_branch_orders(new_order, orders_list)

            return orders_list + new_orders

    def get_A_ps(self):
        if self.is_leaf:
            return []

        # recurse to get all 'above'
        A_ps = []
        for child in self.child_nodes:
            for A_p in child.get_A_ps():
                A_ps.append(A_p)

        n_children = len(self.child_nodes)

        if n_children == 1:
            return A_ps

        if n_children == 2:
            A_p = self.calculate_A_p(
                self.child_nodes[0].get_number_of_leafs(),
                self.child_nodes[1].get_number_of_leafs()
            )
            return A_ps + [A_p]

        if n_children > 2:
            # we can only compare pairs
            # so we split them up and assign pairs randomly

            random.shuffle(self.child_nodes)  # hopefully this doesn't mess things up

            new_A_ps = []
            total_leafs_sub_tree1 = self.child_nodes[0].get_number_of_leafs()
            for i in range(1, n_children):
                total_leafs_sub_tree2 = self.child_nodes[i].get_number_of_leafs()
                A_p = self.calculate_A_p(total_leafs_sub_tree1, total_leafs_sub_tree2)
                new_A_ps.append(A_p)
                total_leafs_sub_tree1 += total_leafs_sub_tree2

            return A_ps + new_A_ps

    @staticmethod
    def calculate_A_p(r, s):
        if r == s == 1:
            return 0
        return abs(r - s) / (r + s - 2)

    def get_number_of_leafs(self):
        if self.number_of_leafs == None:
            # store number of leafs to make the code faster
            self.number_of_leafs = sum([child.get_number_of_leafs() for child in self.child_nodes]) + int(self.is_leaf)

        return self.number_of_leafs

    def __iter__(self):
        yield self
        for child in self.child_nodes:
            for grand_child in child:
                yield grand_child

    def __repr__(self):
        return "Node({})".format(str(self.coords))

    def __str__(self):
        return " " * self.depth + repr(self)


class Tree:
    def __init__(self, root_coords, bounds=[[0, 10], [0, 10], [0, 10]], p=0.4, PS=40):
        self._root = Node(root_coords, 0, None)
        self._node_list = [self._root]
        self._coords_list = [root_coords]
        self._dimensionality = len(root_coords)
        self._matrix_form = np.zeros(list(bound[1] - bound[0] for bound in bounds), dtype=int)
        self._matrix_form[tuple(root_coords)] = 1
        self.bounds = bounds
        self.system_time = 0
        self.p = p
        self.PS = PS

    def plot(self):
        if self._dimensionality == 2:
            self._plot2d()
        elif self._dimensionality == 3:
            self._plot3d()
        else:
            raise NotImplementedError

    def prune(self):
        # remove the nodes, created between 5 and PS timesteps ago
        # with a probability p

        leafs = []
        # leafs that meet the requirements
        for node in self:
            if node.is_leaf:
                if self.system_time - 5 > node.creation_time > self.system_time - self.PS:
                    leafs.append(node)

        # remove the nodes with chance p
        for node in leafs:
            rndm = random.random()
            if rndm < self.p:
                # print("node will be removed:", node)
                parent = node.parent_node

                # remove the node
                parent.remove_child(node)
                self._node_list.remove(node)
                self._coords_list.remove(node.coords)
                self._matrix_form[tuple(node.coords)] = 0

    def add(self, coords, creation_time):
        # adds a new node and prunes

        assert len(coords) == self._dimensionality

        # assign a random parent in range to the new node
        parent = np.random.choice(self._get_neighbour_nodes(coords))
        new_node = parent.add_child(coords, creation_time)

        self.system_time = creation_time

        self._node_list.append(new_node)
        self._coords_list.append(coords)
        self._matrix_form[tuple(coords)] = 1

        self.prune()

        return new_node

    def growth_candidates(self):
        # create a set for all possible growth candidates
        candidates = set()

        for node in self:
            neighbours = self._get_neighbour_coords(node.coords)

            for neighbour in neighbours:
                if neighbour not in self:
                    if self.bounds[1][0] < neighbour[1] < self.bounds[1][1] - 1:
                        candidates.add(tuple(self.boundaries(neighbour)))

        return candidates

    def boundaries(self, coords):
        peridoic_directions = [0, 2]  # X and Z, Y not
        new_coords = coords.copy()
        for i, coord in enumerate(coords):
            if i not in peridoic_directions:
                continue

            if coord < self.bounds[i][0]:
                new_coords[i] = self.bounds[i][1] - 1
            elif coord > self.bounds[i][1] - 1:
                new_coords[i] = self.bounds[i][0]

        return new_coords

    def get_root(self):
        return self._root

    def get_matrix(self):
        return self._matrix_form

    def get_asymmetry_index(self):
        A_ps = self._root.get_A_ps()
        return (1/len(A_ps)) * sum(A_ps)

    def get_branch_orders(self):
        return self._root.get_branch_orders(0, [])

    def get_lenghts_ratio(self):
        total_nodes = len(self)
        nodes_in_terminal_branch = len(self._root.get_nodes_in_terminal_branch())
        intermidiate_nodes = total_nodes - nodes_in_terminal_branch
        return nodes_in_terminal_branch / intermidiate_nodes

    def get_number_of_terminal_segments(self):
        # just count number of leafs
        number_of_terminal_segments = 0
        for node in self:
            if node.is_leaf:
                number_of_terminal_segments += 1

        return number_of_terminal_segments


    def get_fractal_dimension(self):
        Lx = self.bounds[0][1] - self.bounds[0][0]
        Ly = self.bounds[1][1] - self.bounds[1][0]

        # computing the fractal dimension
        # considering only scales in a logarithmic list
        scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
        Ns = []
        # looping over several scales
        for scale in scales:
            # computing the histogram
            H, edges = np.histogramdd(np.array(self._coords_list), bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
            Ns.append(np.sum(H > 0))

        # linear fit, polynomial of degree 1
        coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

        return -coeffs[0]

    def _get_neighbour_coords(self, coords):
        # calculate coords of all neighbours (Moore neighborhood) around a center coord
        d = self._dimensionality
        offsets = np.indices((3,) * d) - 1
        reshaped_offsets = np.stack(offsets, axis=d).reshape(-1, d)
        offsets_without_middle_point = np.delete(reshaped_offsets, int(d**3 / 2), axis=0)
        neighbours = offsets_without_middle_point + coords
        neighbours = neighbours.tolist()
        neighbours = [self.boundaries(neighbour) for neighbour in neighbours]

        return neighbours

    def _get_neighbour_nodes(self, coords):
        # get all nodes that are neighbours of a center coord
        neighbours = self._get_neighbour_coords(coords)
        return [node for node in self if node.coords in neighbours]

    def _plot2d(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(*self.bounds[0])
        ax.set_ylim(*self.bounds[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2D plot")

        for node in self:
            if node.parent_node:
                parent = node.parent_node
                xp, yp = parent.coords
                xc, yc = node.coords

                if xp == self.bounds[0][0] and xc == self.bounds[0][1] - 1:
                    xp = self.bounds[0][1]
                elif xp == self.bounds[0][1] - 1 and xc == self.bounds[0][0]:
                    xp = self.bounds[0][0] - 1

                ax.plot([xp, xc], [yp, yc], c='black')
        plt.show()

    def _plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(*self.bounds[0])
        ax.set_ylim(*self.bounds[1])
        ax.set_zlim(*self.bounds[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(0, 0)

        for node in self:
            if node.parent_node:
                parent = node.parent_node
                xp, yp, zp = parent.coords
                xc, yc, zc = node.coords

                if xp == self.bounds[0][0] and xc == self.bounds[0][1] - 1:
                    xp = self.bounds[0][1]
                elif xp == self.bounds[0][1] - 1 and xc == self.bounds[0][0]:
                    xp = self.bounds[0][0] - 1

                if zp == self.bounds[2][0] and zc == self.bounds[2][1] - 1:
                    zp = self.bounds[2][1]
                elif zp == self.bounds[2][1] - 1 and zc == self.bounds[2][0]:
                    zp = self.bounds[2][0] - 1

                ax.plot3D([xp, xc], [yp, yc], [zp, zc], c='black')

        # For plotting the leafs, not neccessary just keeping it here for now

        # leafs = []
        # for node in self._node_list:
        #     if node.is_leaf:
        #         leafs.append(node)
        # for node in leafs:
        #     if node.parent_node:
        #         parent = node.parent_node
        #         ax.plot3D([parent.coords[0], node.coords[0]], [parent.coords[1], node.coords[1]], [parent.coords[2], node.coords[2]], c='red', alpha = 0.4)

        plt.show()

        return ax

    def __len__(self):
        return len(self._node_list)

    def __iter__(self):
        return self._root.__iter__()

    def __contains__(self, coords):
        return coords in self._coords_list

    def __str__(self):
        return str(self._node_list)


if __name__ == "__main__":
    # plot a 2d test tree
    tree2d = Tree([5, 0], bounds=[[0, 10], [0, 10]])
    tree2d.add([5, 1], 0)
    tree2d.add([6, 1], 0)
    tree2d.add([7, 2], 0)
    tree2d.add([4, 1], 0)
    tree2d.add([3, 2], 0)
    tree2d.add([2, 3], 0)
    tree2d.add([4, 3], 0)
    tree2d.add([6, 0], 0)
    print(tree2d.get_number_of_terminal_segments())
    tree2d.plot()

    # # plot a 3d test tree
    # tree3d = Tree([0, 0, 0], bounds=[[0, 10], [0, 10], [0, 10]])
    # tree3d.add([0, 1, 0], 1)
    # tree3d.add([1, 1, 0], 2)
    # tree3d.add([9, 1, 0], 3)
    # tree3d.add([9, 2, 0], 4)
    # tree3d.add([9, 2, 9], 5)
    # tree3d.plot()
