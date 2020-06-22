import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
rng = np.random.default_rng()


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
        self.system_time = 0
        self.remove_nodes = []

    def plot(self):
        if self._dimensionality == 2:
            self._plot2d()
        elif self._dimensionality == 3:
            self._plot3d()
        else:
            raise NotImplementedError

    def prune(self, p, PS):
        leafs = []

        for i in range(1, len(self._node_list)):
            if self._node_list[i].is_leaf:
                if self.system_time - 5 > self._node_list[i].creation_time > self.system_time - PS:
                    leafs.append([self._node_list[i], i])

        leafs = leafs[::-1] 
        node_types = [[node.is_leaf, node] for node in self._node_list] 

        all_leafs = []

        for node in self._node_list:
            if node.is_leaf == True:
                all_leafs.append([node, node.creation_time])

        for node, index in leafs:
            rndm = random.random()
            if rndm < p:
                print("node will be removed:", node)
                parent = self._node_list[index].parent_node

                if len(parent.child_nodes) == 1:
                    parent.is_leaf = True

                removed1 = self._node_list.pop(index)
                removed2 = self._coords_list.pop(index)


    def add(self, coords, creation_time):
        assert len(coords) == self._dimensionality
        

        parent = rng.choice(self._get_neighbours(coords))
        new_node = parent.add_child(coords, creation_time)

        self.system_time = creation_time

        self._node_list.append(new_node)
        self._coords_list.append(coords)
        self._matrix_form[tuple(coords)] = 1    

        self.prune(0.4, 40)

        return new_node

    def boundaries(self, coords):
        i, j, k = coords
        new_coords = coords
        if i < 0:
            new_coords =  [self.bounds[0][1] - 1, new_coords[1], new_coords[2]]
        if i > self.bounds[0][1] - 1:
            new_coords =  [0, new_coords[1], new_coords[2]]
        if k < 0:
            new_coords =  [new_coords[0], new_coords[1], self.bounds[2][1] - 1]
        if k > self.bounds[2][1] - 1:
            new_coords =  [new_coords[0], new_coords[1], 0]
        
        return new_coords
        

    def get_root(self):
        return self._root

    def _get_neighbours(self, coords):
        # calculate coords of all neighbours (Moore neighborhood) around a center coord

        d = self._dimensionality
        offsets = np.indices((3,) * d) - 1
        reshaped_offsets = np.stack(offsets, axis=d).reshape(-1, d)
        offsets_without_middle_point = np.delete(reshaped_offsets, int(d**3 / 2), axis=0)
        neighbours = offsets_without_middle_point + coords
        neighbours = neighbours.tolist()
        neighbours = [self.boundaries(neighbour) for neighbour in neighbours]

        # return nodes that are neighbours
        return [node for node in self._node_list if node.coords in neighbours]

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
                ax.plot([parent.coords[0], node.coords[0]], [parent.coords[1], node.coords[1]], c='black')
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

        length = 0
        for x in self:
            length += 1

        for node in self._node_list:
            if node.parent_node:
                parent = node.parent_node
                ax.plot3D([parent.coords[0], node.coords[0]], [parent.coords[1], node.coords[1]], [parent.coords[2], node.coords[2]], c='black')
        leafs = []
        for node in self._node_list:
            if node.is_leaf:
                leafs.append(node)
        for node in leafs:
            if node.parent_node:
                parent = node.parent_node
                ax.plot3D([parent.coords[0], node.coords[0]], [parent.coords[1], node.coords[1]], [parent.coords[2], node.coords[2]], c='red', alpha = 0.4)

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
    tree2d.add([5, 1], 1)
    tree2d.add([6, 1], 2)
    tree2d.add([4, 1], 3)
    tree2d.add([5, 2], 4)
    tree2d.add([5, 3], 5)
    tree2d.plot()

    print(tree2d._get_neighbours([6, 2]))

    # plot a 3d test tree
    tree3d = Tree([5, 0, 5], bounds=[[0, 10], [0, 10], [0, 10]])
    tree3d.add([5, 1, 5], 1)
    tree3d.add([6, 1, 5], 2)
    tree3d.add([4, 1, 5], 3)
    tree3d.add([5, 2, 5], 4)
    tree3d.add([5, 2, 4], 5)
    tree3d.plot()
