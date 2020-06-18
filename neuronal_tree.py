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

        if parent_node is None:
            self.depth = 0
        else:
            self.depth = parent_node.depth + 1

    def add_child(self, coords, creation_time):
        new_node = Node(coords, creation_time, self)
        self.child_nodes.append(new_node)
        self.is_leaf = False
        return new_node

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
    def __init__(self, root_coords, bounds=[[0, 10], [0, 10], [0, 10]]):
        self._root = Node(root_coords, 0, None)
        self._node_list = [self._root]
        self._coords_list = [root_coords]
        self._dimensionality = len(root_coords)
        self._matrix_form = np.zeros(list(bound[1] - bound[0] for bound in bounds), dtype=int)
        self._matrix_form[tuple(root_coords)] = 1
        self.bounds = bounds

    def plot(self):
        if self._dimensionality == 2:
            self._plot2d()
        elif self._dimensionality == 3:
            self._plot3d()
        else:
            raise NotImplementedError

    def add(self, coords, creation_time, parent):
        assert len(coords) == self._dimensionality

        # replace with search an stuff to find a parent node instead of given as argument
        new_node = parent.add_child(coords, creation_time)
        self._node_list.append(new_node)
        self._coords_list.append(coords)

        self._matrix_form[tuple(coords)] = 1
        return new_node

    def get_root(self):
        return self._root

    def _plot2d(self):
        N = len(self._matrix_form)

        # convert all 1s in matrix to nans
        matrix_conv = [[0 for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(N):
                if self._matrix_form[i][j] == 1:
                    matrix_conv[i][j] = float('nan')

        # plot in 2D
        fig, axs = plt.subplots(1, 1)
        axs.imshow(matrix_conv, cmap = 'cubehelix')
        axs.set_title("2D plot")
        axs.set_xlabel("x position [-]")
        axs.set_ylabel("y position [-]")
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
                ax.plot3D([parent.coords[0], node.coords[0]], [parent.coords[1], node.coords[1]], [parent.coords[2], node.coords[2]], c='black')
        plt.show()

    def __iter__(self):
        return self._root.__iter__()

    def __contains__(self, coords):
        return coords in self._coords_list

    def __str__(self):
        return str(self._node_list)


if __name__ == "__main__":
    # plot a 2d test tree
    tree2d = Tree([5, 0], bounds=[[0, 10], [0, 10]])
    some_node = tree2d.add([5, 1], 1, tree2d.get_root())
    tree2d.add([6, 1], 2, tree2d.get_root())
    tree2d.add([4, 1], 3, tree2d.get_root())
    another_node = tree2d.add([5, 2], 4, some_node)
    tree2d.add([5, 3], 5, another_node)
    tree2d.plot()

    # plot a 3d test tree
    tree3d = Tree([5, 0, 5], bounds=[[0, 10], [0, 10], [0, 10]])
    some_node = tree3d.add([5, 1, 5], 1, tree3d.get_root())
    tree3d.add([6, 1, 5], 2, tree3d.get_root())
    tree3d.add([4, 1, 5], 3, tree3d.get_root())
    another_node = tree3d.add([5, 2, 5], 4, some_node)
    tree3d.add([5, 3, 4], 5, another_node)
    tree3d.plot()
