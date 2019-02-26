import unittest
from grid import Grid
import numpy as np

class test_object_grid(unittest.TestCase):



    def test_change_move_1(self):

        gridworld = Grid()

        expected_index = [0,2]
        initial_index = [0,1]
        move = [0,1,0,0]

        new_index = gridworld.change_index_according_to_move(initial_index,move)

        self.assertEqual(expected_index, new_index)

    def test_change_move_2(self):

        gridworld = Grid()

        expected_index = [0,0]
        initial_index = [1,0]
        move = [0,0,1,0]

        new_index = gridworld.change_index_according_to_move(initial_index,move)

        self.assertEqual(expected_index, new_index)

    def test_is_terminal_state_1(self):

        gridworld = Grid()
        index = [0,3]

        self.assertTrue(gridworld._isTerminalState(index))

    def test_is_terminal_state_2(self):

        gridworld = Grid()
        index = [1,3]

        self.assertTrue(gridworld._isTerminalState(index))

    def test_is_terminal_state_3(self):

        gridworld = Grid()
        index = [1,2]

        self.assertFalse(gridworld._isTerminalState(index))

    def test_policy_evaluation_by_iteration(self):
        """
        succes_of_move: 100%
        cost_by_move: -0.04
        gamma: 0.8

        :return:
        """

        gridworld = Grid()

        # boolean nomenclature [left, right, up, down]
        policy = np.array([[[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
                           [[0,0,1,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]],
                           [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0]]
                          ])

        values = gridworld.evaluate_policy_by_iteration(policy)


        expected_values = np.array([[0.6206,0.734,0.86,1],
                                    [0.51854,0,0.734,-1 ],
                                    [-0.79366,-0.8374,-0.886,-0.94 ],
                                    ])

        np.testing.assert_allclose(expected_values, values)


    def test_policy_improvement(self):
        """
        succes_of_move: 100%
        cost_by_move: -0.04
        gamma: 0.9

        :return:
        """

        gridworld = Grid()

        # boolean nomenclature [left, right, up, down]
        policy = np.array([[[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
                           [[0,0,1,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]],
                           [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0]]
                          ])
        expected_policy = np.array([[[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
                           [[0,0,1,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]],
                           [[0,1,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0]]
                          ])

        values = gridworld.evaluate_policy_by_iteration(policy)



        newPolicy,newValues = gridworld.improve_policy(policy,values)



        np.testing.assert_equal(expected_policy, newPolicy)

    def test_policy_evaluation_MonteCarlo(self):
        """
                succes_of_move: 100%
                cost_by_move: -0.04
                gamma: 0.8

                :return:
                """

        gridworld = Grid()

        # boolean nomenclature [left, right, up, down]
        policy = np.array([[[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                           [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                           [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
                           ])

        values = gridworld.evaluate_policy_by_MonteCarlo(policy)

        expected_values = np.array([[0.88, 0.92, 0.96, 1],
                                    [0.84, 0, 0.92, -1],
                                    [-1.16, -1.12, -1.08, -1.04],
                                    ])

        np.testing.assert_allclose(expected_values, values)


suite = unittest.TestLoader().loadTestsFromTestCase(test_object_grid)
unittest.TextTestRunner(verbosity=2).run(suite)


