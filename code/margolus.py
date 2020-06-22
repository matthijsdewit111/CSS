from itertools import product
from random import sample

import numpy as np

# def sort_by_row_sum(array):
#     order = np.argsort(np.sum(array, axis=1))
#     return array[order]


class State:
    def __init__(self, state, dimensions=[2, 2]):
        self._state = np.array(state)
        self.dimensions = dimensions

    def __lt__(self, other):
        return sum(self._state) < sum(other._state)

    def __eq__(self, other):
        return np.all(self._state == other._state)

    def __str__(self):
        return str(self._state.reshape(*dimensions))


class States(list):
    def __init__(self, state_dimensions=[2, 2]):
        self._state_dimensions = state_dimensions
        self._cells_per_state = np.prod(self._state_dimensions)
        self._total_states = 2 ** self._cells_per_state
        self._init_states()

    def index(self, state):
        print("index")
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
    def __init__(self, states):
        self.states = states

        self._rule_table = self._generate_rule_table()

    def transform(self, state):
        return states[self[states.index(state)]]

    def _generate_rule_table(self):
        # generate shuffled version
        states_shuffled = sample(self.states, len(self.states))

        # sort shuffled, so corresponding rows between original
        # and shuffled have same number of 1s and 0s
        states_shuffled = np.sort(states_shuffled)

        # generate rule by looking up indexes
        rule_table = np.empty(len(self.states), dtype=int)
        for i, state in enumerate(states_shuffled):
            rule_table[i] = states.index(state)

        return rule_table

    def __getitem__(self, index):
        return self._rule_table[index]
    
    def __str__(self):
        return str(self._rule_table)


def generate_init_setup(grid_dimensions=[10, 10]):
    pass


if __name__ == "__main__":
    dimensions = [2, 2]
    states = States(dimensions)
    print(states)

    rule = RuleTable(states)
    print("rule:", rule)

    current_state = states[5]
    print("current state / states[5]:\n", current_state)
    print("rule[5]:", rule[5])
    print("next state / states[rule[5]]:\n", rule.transform(current_state))
