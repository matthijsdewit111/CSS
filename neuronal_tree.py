import matplotlib.pyplot as plt
import numpy as np


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
        # self._coords_list = [root_coords]
        self._dimensionality = len(root_coords)
        self._matrix_form = np.zeros(list(bound[1] - bound[0] for bound in bounds), dtype=int)
        self._matrix_form[tuple(root_coords)] = 1

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
        # self._coords_list.append(coords)

        self._matrix_form[tuple(coords)] = 1
        return new_node

    def get_root(self):
        return self._root

    def _plot2d(self):
        print(self._matrix_form)

    def _plot3d(self):
        raise NotImplementedError

    def __iter__(self):
        return self._root.__iter__()

    def __contains__(self, coords):
        return coords in self._coords_list

    def __str__(self):
        return str(self._node_list)


if __name__ == "__main__":
    tree = Tree([5, 0], bounds=[[0, 10], [0, 10]])

    rn = tree.add([5, 1], 1, tree.get_root())
    tree.add([6, 1], 2, tree.get_root())
    tree.add([4, 1], 3, tree.get_root())

    rn2 = tree.add([5, 2], 4, rn)
    tree.add([5, 3], 5, rn2)

    # print(tree)

    for node in tree:
        print(node)

    tree.plot()