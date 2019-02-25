import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from enum import Enum



class Move(Enum):

    UP = [0, 0, 1, 0]
    DOWN = [0, 0, 0, 1]
    LEFT = [1, 0, 0, 0]
    RIGHT = [0, 1, 0, 0]
    STAY = [0, 0, 0, 0]

class Grid():


    def __init__(self):

        self.possible_moves = np.array([
                [[0,1,0,1],[1,1,0,0],[0,1,0,1],[0,0,0,0]],
                [[0,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                [[0,1,1,0],[1,1,0,0],[1,1,1,0],[1,0,1,0]]
                ])

        self.terminal_states_rewards = np.array([
                    [0.0,0.0,0.0,1.0],
                    [0.0,0.0,0.0,-1.0],
                    [0.0,0.0,0.0,0.0]
                    ])

        self.cost_by_move = -0.04
        self.gamma = 1.0


        self.grid_i_size = 3
        self.grid_j_size = 4

        self.optimal_policy = np.zeros((self.grid_i_size, self.grid_j_size, 4))

    def evaluate_policy_by_iteration(self,policy):

        """
        Evaluate policy by iteration. Dynamic programming approach.

        Section 4.4 Value iteration in Reinforment Learning: An introduction by Sutton
        """

        # TODO use tolerance as possible break

        max_number_of_iteration = 100

        values = np.zeros((self.grid_i_size, self.grid_j_size))

        tolerance = 0.01

        for iteration in range(max_number_of_iteration):

            for i in range(self.grid_i_size):
                for j in range(self.grid_j_size):

                    inital_index = [i, j]
                    new_index_for_move = self.change_index_according_to_move([i, j],
                                                                            policy[i,j])
                    values[i,j] = 0.0

                    if self._isTerminalState(inital_index):
                        values[i, j] = self.terminal_states_rewards[tuple(inital_index)]

                    elif self.stateIsNotAccessible(inital_index):
                        values[i, j] = 0.0

                    else:
                        values[i, j] += self.cost_by_move
                        values[i, j] += self.gamma*values[tuple(new_index_for_move)]

        return values

    def improve_policy(self,policy,values):
        """

        Policy Improvement p.98 section 4.2

        :param policy:
        :param values:
        :return:
        """

        possible_moves = np.array([
                            Move.LEFT.value,
                            Move.RIGHT.value,
                            Move.UP.value,
                            Move.DOWN.value
                            ])



        policy_stable = False

        while not policy_stable:

            policy_stable = True

            for i in range(self.grid_i_size):
                for j in range(self.grid_j_size):

                        temp_move, initalIndex = policy[i, j], [i, j]

                        # evaluate possible actions

                        if self._isTerminalState(initalIndex):
                            value_of_action = self.terminal_states_rewards[tuple(initalIndex)]
                            action_to_take = Move.STAY.value

                        elif self.stateIsNotAccessible(initalIndex):
                            value_of_action = 0.0
                            action_to_take = Move.STAY.value

                        else:

                            values_per_move = self._evaluate_moves(initalIndex, possible_moves, values)

                            action_to_take_index = np.argmax(values_per_move)
                            action_to_take = possible_moves[action_to_take_index]
                            value_of_action = values_per_move[action_to_take_index]

                        if not np.array_equal(action_to_take, temp_move):
                            policy_stable = False

                        policy[i, j] = action_to_take
                        values[i, j] = value_of_action


        return policy,values

    def change_index_according_to_move(self,index,move_array):

        # boolean nomenclature [left, right, up, down]
        # if going up, the index is smaller

        newIndex = index.copy()

        newIndex[0] += np.dot(move_array[2:4], [-1, 1])
        newIndex[1] +=  np.dot(move_array[0:2],[-1,1])

        assert self._isIndexValid(newIndex), "new index not valid"

        return newIndex

    def _isIndexValid(self, index):

        if index[0] < 0: return False
        if index[0] >= self.grid_i_size : return False

        if index[1] < 0: return False
        if index[1] >= self.grid_j_size: return False

        return True

    def _isMoveValid(self, index, move_array):

        tempIndex = index.copy()

        tempIndex[0] += np.dot(move_array[2:4], [-1, 1])
        tempIndex[1] += np.dot(move_array[0:2], [-1, 1])

        return self._isIndexValid(tempIndex)

    def _isTerminalState(self, index):

        if self.terminal_states_rewards[tuple(index)] == 0.0 :

            return False

        return True

    def stateIsNotAccessible(self,index):

        if index == [1,1]:

            return True

        return False

    def evaluate_policy_by_MonteCarlo(self,policy):
        """
         Small modification of section 5.1 p.113. Start the evaluation now from random but from the less
         visited stateas it improved the convergence in this case convergence.

        :param policy:
        :return:
        """

        values = self.terminal_states_rewards.copy()
        visits = self._return_initial_visits()

        max_number_of_iteration = 100

        for iteration in range(max_number_of_iteration):

            i,j = self._return_random_state()

            if iteration > max_number_of_iteration/4:

                # Greedy evaluation
                i,j = self._return_less_visited_state(visits)

            while not self._isTerminalState([i, j]) and not self.stateIsNotAccessible([i, j]):


                new_index_for_move = self.change_index_according_to_move([i, j],
                                                                         policy[i, j])

                self._update_value(values, [i,j], new_index_for_move)
                self._update_visits(visits, [i,j])

                i = new_index_for_move[0]
                j = new_index_for_move[1]


        return values

    def show_values(self,values):

        conf_arr = values

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = ax.imshow(array(conf_arr), cmap='gray', interpolation='nearest')

        for i, cas in enumerate(conf_arr):
            for j, c in enumerate(cas):
                if c != 0.0:
                    plt.text(j - 0.2, i + .2, '{:1.2f}'.format(c), fontsize=14, color='red')

        cb = fig.colorbar(res)

        gca().tick_params(axis='x', labelbottom='off')
        gca().tick_params(axis='y', labelleft='off')

        plt.show()

    def _evaluate_moves(self,state_index,possible_moves, values):

        #number_of_possible_moves = 4
        values_per_move = np.zeros(len(possible_moves))

        for index, move in enumerate(possible_moves):

            if not self._isMoveValid(state_index, move):
                values_per_move[index] = float('-inf')
                continue

            new_index_for_move = self.change_index_according_to_move(state_index, move)

            values_per_move[index] += self.cost_by_move
            values_per_move[index] += self.gamma * values[tuple(new_index_for_move)]

        return values_per_move

    def _return_initial_visits(self):

        visits = np.zeros((self.grid_i_size, self.grid_j_size))
        visits[0, 3] = float('inf')
        visits[1, 3] = float('inf')
        visits[1, 1] = float('inf')

        return visits

    def _return_random_state(self):

        return np.random.randint(low=0, high=self.grid_i_size),np.random.randint(low=0, high=self.grid_j_size)

    def _return_less_visited_state(self,visits):

        index_less_evaluated = np.argwhere(visits == np.min(visits))
        i = index_less_evaluated[0][0]
        j = index_less_evaluated[0][1]

        return i,j

    def _update_value(self,values, current_index, new_index_for_move):

        previous_value = values[tuple(current_index)]
        current_reward = self.cost_by_move + values[tuple(new_index_for_move)]
        correction_value = self.gamma * (current_reward - previous_value)  # here gamma is the learning rate

        values[tuple(current_index)] += correction_value

    def _update_visits(self,visits,state):

        visits[tuple(state)] += 1

    def show_policy(self,policy):

        """
        Reference for matplotlib artists

        This example displays several of matplotlib's graphics primitives (artists)
        drawn using matplotlib API. A full list of artists and the documentation is
        available at http://matplotlib.org/api/artist_api.html.

        Copyright (c) 2010, Bartosz Telenczuk
        BSD License
        """
        import matplotlib.pyplot as plt
        plt.rcdefaults()

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        number_i = policy.shape[0]
        number_j = policy.shape[1]

        fig, ax = plt.subplots()
        # create 3x3 grid to plot the artists
        grid = np.mgrid[0.1:0.7:1j * number_j, 0.1:0.5:1j * number_i].reshape(2, -1).T

        patches = []

        index_grid = 0
        for j in range(number_j):
            for i in range(number_i):
                i = number_i - i - 1

                action_to_take = policy[i, j]
                shifts = action_to_take * [-1, 1, 1, -1]

                shift_x = sum(shifts[0:2])
                shift_y = sum(shifts[2:4])

                arrow = mpatches.Arrow(grid[index_grid, 0] - 0.05 * shift_x,
                                       grid[index_grid, 1] - 0.05 * shift_y,
                                       0.1 * shift_x,
                                       0.1 * shift_y,
                                       width=0.1)

                patches.append(arrow)

                index_grid += 1

        collection = PatchCollection(patches)
        ax.add_collection(collection)
        ax.set_xticks(np.arange(0, 0.8, 0.2))
        ax.set_yticks(np.arange(0, 0.6, 0.2))

        plt.axis('equal')
        gca().tick_params(axis='x', labelbottom='off')
        gca().tick_params(axis='y', labelleft='off')
        plt.grid()

        plt.show()













