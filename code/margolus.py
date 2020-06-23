from itertools import product
from random import sample

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# def sort_by_row_sum(array):
#     order = np.argsort(np.sum(array, axis=1))
#     return array[order]


class State:
    def __init__(self, state, dimensions=[2, 2]):
        self._state = np.array(state).flatten()
        self._reshaped_state = self._state.reshape(*dimensions)
        self.dimensions = dimensions

    def from_matrix(matrix):
        return State(matrix, matrix.shape)

    def get_state(self):
        return self._reshaped_state

    def __lt__(self, other):
        return sum(self._state) < sum(other._state)

    def __eq__(self, other):
        return np.all(self._state == other._state)

    def __str__(self):
        return str(self._reshaped_state)

    def __repr__(self):
        return str(self)


class States(list):
    def __init__(self, state_dimensions=[2, 2]):
        self._state_dimensions = state_dimensions
        self._cells_per_state = np.prod(self._state_dimensions)
        self._total_states = 2 ** self._cells_per_state
        self._init_states()

    def get_dimensionality(self):
        return len(self._state_dimensions)

    def index(self, state):
        return self._states.index(state)

    def _init_states(self):
        state_combinations = product([0, 1], repeat=self._cells_per_state)
        self._states = [State(state, self._state_dimensions) for state in state_combinations]
        self._states.sort()

    def __getitem__(self, index):
        return self._states[index]

    def __iter__(self):
        for state in self._states:
            yield state

    def __len__(self):
        return self._total_states

    def __str__(self):
        r = ""
        for i, state in enumerate(self._states):
            r += str(i) + ":\n" + str(state) + "\n"
        return r


class RuleTable(list):
    def __init__(self, states, diffusion_param=None):
        self.states = states
        if diffusion_param is None:
            self.diffusion_param = 1 / states.get_dimensionality()
        else:
            self.diffusion_param = diffusion_param

        self._rule_table = self._generate_rule_table()

    def transform(self, state):
        return states[self[states.index(state)]]

    def _generate_rule_table(self):
        # # generate shuffled version
        # states_shuffled = sample(self.states, len(self.states))

        # # sort shuffled, so corresponding rows between original
        # # and shuffled have same number of 1s and 0s
        # states_shuffled = np.sort(states_shuffled)

        # # generate rule by looking up indexes
        # rule_table = np.empty(len(self.states), dtype=int)
        # for i, state in enumerate(states_shuffled):
        #     rule_table[i] = states.index(state)

        # rotations version
        n = self.states.get_dimensionality()
        p = self.diffusion_param
        d = n * (n - 1) // 2  # degrees of freedom for rotations, 1 for 2D, 3 for 3D
        rotation_options = np.arange(d + 1)
        rotation_probabilities = [1 - d*p] + [p] * d
        states_rotated = []
        for state in self.states:
            rotation_axes = rng.choice(rotation_options, p=rotation_probabilities)  # 0 is no rotation
            rotation_direction = rng.choice([1, -1])

            original_matrix = state.get_state()
            rotated_matrix = original_matrix
            if rotation_axes != 0:
                if n == 2:  # 2D case
                    rotated_matrix = np.rot90(original_matrix, rotation_direction)
                elif n == 3:  # 3D case
                    axes = None
                    if rotation_axes == 1:
                        axes = (0, 1)
                    elif rotation_axes == 2:
                        axes = (0, 2)
                    elif rotation_axes == 3:
                        axes = (1, 2)
                    else:
                        print("you're not supposed to see this")

                    rotated_matrix = np.rot90(original_matrix, rotation_direction, axes=axes)
                else:
                    raise NotImplementedError

            rotated_state = State.from_matrix(rotated_matrix)
            states_rotated.append(rotated_state)

        # generate rule by looking up indexes
        rule_table = np.empty(len(self.states), dtype=int)
        for i, state in enumerate(states_rotated):
            rule_table[i] = states.index(state)

        return rule_table

    def __getitem__(self, index):
        return self._rule_table[index]

    def __str__(self):
        return str(self._rule_table)


def generate_init_setup(grid_dimensions=[10, 10]):
    # grid_dimensions[1] //= 2
    # no_particles_half = np.zeros(grid_dimensions)
    # particles_half = rng.integers(0, 2, grid_dimensions)
    # grid = np.hstack((no_particles_half, particles_half))
    grid = np.zeros(grid_dimensions)
    x = rng.integers(dimensions[0])
    y = rng.integers(dimensions[1])
    grid[x][y] = 1
    return grid


def step(grid, rule, t):

    offset = t % 2
    x, y = grid.shape

    for i in range((x - offset) // 2):
        for j in range((y - offset) // 2):
            x1 = i*2 + offset
            x2 = i*2 + offset + 2
            y1 = j*2 + offset
            y2 = j*2 + offset + 2
            sub_section = grid[x1:x2, y1:y2]
            state = State.from_matrix(sub_section)
            new_state = rule.transform(state)
            grid[x1:x2, y1:y2] = new_state.get_state()

    return grid


if __name__ == "__main__":
    dimensions = [2, 2]
    states = States(dimensions)
    # print(states)

    rule = RuleTable(states, diffusion_param=0.5)
    print("rule:", rule)

    # for i, state in enumerate(states):
    #     print("current:")
    #     print(state)
    #     print("rule:", rule[i])
    #     print("next:")
    #     print(rule.transform(state), '\n')

    grid = generate_init_setup(grid_dimensions=[10, 10])

    for t in range(100):
        grid = step(grid, rule, t)
        plt.imshow(grid)
        plt.show()
