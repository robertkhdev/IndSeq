import unittest
import pandas as pd
import time


from utils import *


def run_tests(diagnostic=False):
    np.random.seed(1234)
    K = 3  # indications
    r = 1 / (1 + 0.1)  # discount factor
    test_results = tuple([None] * K)
    launched = tuple([None] * K)
    launched_first = None
    t_per = 0
    # t_per = 1
    launch_period = None

    test_cases = pd.read_csv('TestCases.csv', index_col=0)

    failures = 0
    times = []
    total_tic = time.perf_counter()
    for index, row in test_cases.iterrows():
        if index >= 1:
            excel_value = row['Value']
            joint_probs = np.array([[[row['abc'], row['abC']],
                                     [row['aBc'], row['aBC']]],
                                    [[row['Abc'], row['AbC']],
                                     [row['ABc'], row['ABC']]]])
            ind_values = np.array([row['vA_A'], row['vB_A'], row['vC_A']])
            pricing_mults = np.array([1, row['B_mult'], row['C_mult']])
            test_costs = np.array([0.1] * K)

            x = State(Tests=test_results,
                      Launched=launched,
                      Period=t_per,
                      PeriodValue=0,
                      EV=0)

            tic = time.perf_counter()
            tree = tree_start(x, joint_prob=to_tuple(joint_probs),
                              test_costs=tuple(test_costs),
                              ind_demands=tuple(ind_values),
                              prices=tuple(pricing_mults),
                              discount_factor=r)
            toc = time.perf_counter()
            times.append(toc - tic)
            excel_value = round(excel_value, 5)
            val = round(tree.State.EV, 5)
            if excel_value != val:
                failures += 1
                print(index, excel_value == val, excel_value, val)
            if diagnostic:
                pass
                # print(json.dumps(tree))

    total_toc = time.perf_counter()

    print('\n', failures, 'failures out of', str(len(test_cases)))
    print('Avg. Time = ', np.mean(times))
    print('Total Time = ', total_toc - total_tic)


def make_root_state(n_indications: int) -> State:
    x = State(Tests=tuple([None] * n_indications),
              Launched=tuple([None] * n_indications),
              Period=0,
              PeriodValue=0,
              EV=0)
    return x


def make_uniform_problem(n_indications: int) -> dict:
    K = n_indications
    specs = {'joint_prob': to_tuple(np.ones(2 ** K).reshape(*([2] * K)) / (2 ** K)),
             'test_costs': tuple(np.array([0.1] * K)),
             'ind_demands': tuple(np.array([1] * K)),
             'prices': tuple(np.array([1] * K)),
             'discount_factor': 1 / (1 + 0.0),
             'patent_window': 10,
             'allow_simult_tests': False}
    return specs


class TestTreeK1(unittest.TestCase):
    def test_simple_tree(self):
        # set up tree
        K = 1
        x = make_root_state(K)
        specs = make_uniform_problem(K)
        specs['patent_window'] = 2
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 0.4)

    def test_no_demand(self):
        # set up tree
        K = 1
        x = make_root_state(K)
        specs = make_uniform_problem(K)
        specs['ind_demands'] = (0,)
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 0.0)

    def test_10_periods(self):
        # set up tree
        K = 1
        x = make_root_state(K)
        specs = make_uniform_problem(K)
        specs['patent_window'] = 10
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 4.4)


class TestTreeK2_example1(unittest.TestCase):
    # parameters for example 1
    K = 2
    joint_prob = np.array([[0.25, 0.45], [0.25, 0.05]])
    test_costs = np.array([0.1, 0.1])
    ind_demands = np.array([1, 1])
    prices = np.array([1, 0.5])
    discount_factor = 1 / (1 + 0.1)
    patent_window = 3

    # testing dimensions
    # - patent_window
    # - discount_factor
    # - allow_simult_tests
    # - price mechanism?

    def get_default_specs(self):
        return {'joint_prob': to_tuple(self.joint_prob),
                'test_costs': tuple(self.test_costs),
                'ind_demands': tuple(self.ind_demands),
                'prices': tuple(self.prices),
                'discount_factor': self.discount_factor,
                'patent_window': self.patent_window,
                'allow_simult_tests': False}


    def test_0(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.542975207, 6), 0.0)

    def test_1(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['discount_factor'] = 1 / (1 + 0.0)
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.655, 6), 0.0)

    def test_2(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['patent_window'] = 2
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.172727273, 6), 0.0)

    def test_3(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['patent_window'] = 2
        specs['allow_simult_tests'] = True
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.277272727, 6), 0.0)

    def test_4(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['allow_simult_tests'] = True
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.711157025, 6), 0.0)

    def test_5(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['discount_factor'] = 1 / (1 + 0.0)
        specs['allow_simult_tests'] = True
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.85, 6), 0.0)

    def test_6(self):
        x = make_root_state(self.K)
        specs = self.get_default_specs()
        specs['discount_factor'] = 1 / (1 + 0.0)
        specs['allow_simult_tests'] = True
        specs['patent_window'] = 2
        tree = tree_start(x, **specs)
        self.assertEqual(round(tree.State.EV - 0.325, 6), 0.0)



class TestTreeK2(unittest.TestCase):
    K = 2

    def test_2periods(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 2
        tree = tree_start(x, **specs)

        self.assertEqual(round(tree.State.EV - 0.4, 6), 0.0)

    def test_3periods(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 3
        tree = tree_start(x, **specs)

        self.assertEqual(round(tree.State.EV - 1.3, 6), 0.0)

    def test_2periods2(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 2
        specs['joint_prob'] = to_tuple(np.array([[0.5625, 0.1875], [0.1875, 0.0625]]))
        tree = tree_start(x, **specs)

        self.assertEqual(round(tree.State.EV - 0.15, 3), 0.0)

    def test_2periods2parallel(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 2
        specs['joint_prob'] = to_tuple(np.array([[0.5625, 0.1875], [0.1875, 0.0625]]))
        specs['allow_simult_tests'] = True
        specs['discount_factor'] = 1 / (1 + 0.0)
        tree = tree_start(x, **specs)

        self.assertEqual(round(tree.State.EV - 0.30, 3), 0.0)

    def test_2periods3parallel(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 3
        specs['joint_prob'] = to_tuple(np.array([[0.5625, 0.1875], [0.1875, 0.0625]]))
        specs['allow_simult_tests'] = True
        specs['discount_factor'] = 1 / (1 + 0.0)
        tree = tree_start(x, **specs)

        self.assertEqual(round(tree.State.EV - 0.80, 3), 0.0)

    def test_10periods(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['patent_window'] = 10
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 8.3)

    def test_no_demand(self):
        # set up tree
        x = make_root_state(self.K)
        specs = make_uniform_problem(self.K)
        specs['ind_demands'] = (0, 0)
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 0)


# @unittest.skip("work in progress")
class TestTreeK4(unittest.TestCase):

    def test_tree_k4(self):
        P1 = 0.5  # Probability of success for the first indication
        P2 = 0.5  # Probability of success for the second indication
        P3 = 0.5  # Probability of success for the third indication
        P4 = 0.5  # Probability of success for the fourth indication

        x = make_root_state(4)

        # Create a 4-dimensional array for joint probability distribution
        joint_prob = np.zeros((2, 2, 2, 2))

        # Populate the array with joint probabilities
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        joint_prob[i][j][k][l] = ((P1 if i == 1 else 1 - P1) *
                                                       (P2 if j == 1 else 1 - P2) *
                                                       (P3 if k == 1 else 1 - P3) *
                                                       (P4 if l == 1 else 1 - P4))

        test_costs = np.array([0.1, 0.1, 0.1, 0.1])
        ind_demands = np.array([1, 1, 1, 1])
        prices = np.array([1, 1, 1, 1])
        discount_factor = 1 / (1 + 0.0)
        patent_window = 2
        tree = tree_start(x, joint_prob=to_tuple(joint_prob),
                          test_costs=tuple(test_costs),
                          ind_demands=tuple(ind_demands),
                          prices=tuple(prices),
                          discount_factor=discount_factor,
                          patent_window=patent_window)
        self.assertEqual(round(tree.State.EV - 0.4, 6), 0.0)

    def test_tree_k4_simtesting(self):
        P1 = 0.9  # Probability of success for the first indication
        P2 = 0.9  # Probability of success for the second indication
        P3 = 0.9  # Probability of success for the third indication
        P4 = 0.9  # Probability of success for the fourth indication

        x = make_root_state(4)

        # Create a 4-dimensional array for joint probability distribution
        joint_prob = np.zeros((2, 2, 2, 2))

        # Populate the array with joint probabilities
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        joint_prob[i][j][k][l] = ((P1 if i == 1 else 1 - P1) *
                                                       (P2 if j == 1 else 1 - P2) *
                                                       (P3 if k == 1 else 1 - P3) *
                                                       (P4 if l == 1 else 1 - P4))

        test_costs = np.array([0.1, 0.1, 0.1, 0.1])
        ind_demands = np.array([1, 1, 1, 1])
        prices = np.array([1, 1, 1, 1])
        discount_factor = 1 / (1 + 0.0)
        patent_window = 2
        tree = tree_start(x, joint_prob=to_tuple(joint_prob),
                          test_costs=tuple(test_costs),
                          ind_demands=tuple(ind_demands),
                          prices=tuple(prices),
                          discount_factor=discount_factor,
                          patent_window=patent_window,
                          allow_simult_tests=True)
        self.assertEqual(round(tree.State.EV - 3.2, 6), 0.0)

    def test_tree_k4_simtesting_2periods(self):
        P1 = 0.9  # Probability of success for the first indication
        P2 = 0.9  # Probability of success for the second indication
        P3 = 0.9  # Probability of success for the third indication
        P4 = 0.9  # Probability of success for the fourth indication

        x = make_root_state(4)

        # Create a 4-dimensional array for joint probability distribution
        joint_prob = np.zeros((2, 2, 2, 2))

        # Populate the array with joint probabilities
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        joint_prob[i][j][k][l] = ((P1 if i == 1 else 1 - P1) *
                                                       (P2 if j == 1 else 1 - P2) *
                                                       (P3 if k == 1 else 1 - P3) *
                                                       (P4 if l == 1 else 1 - P4))

        test_costs = np.array([0.1, 0.1, 0.1, 0.1])
        ind_demands = np.array([1, 1, 1, 1])
        prices = np.array([1, 1, 1, 1])
        discount_factor = 1 / (1 + 0.0)
        patent_window = 2
        tree = tree_start(x, joint_prob=to_tuple(joint_prob),
                          test_costs=tuple(test_costs),
                          ind_demands=tuple(ind_demands),
                          prices=tuple(prices),
                          discount_factor=discount_factor,
                          patent_window=patent_window,
                          allow_simult_tests=True)
        self.assertEqual(round(tree.State.EV - 3.2, 6), 0.0)

    def test_tree_k4_simtesting_disc(self):
        P1 = 0.5  # Probability of success for the first indication
        P2 = 0.5  # Probability of success for the second indication
        P3 = 0.5  # Probability of success for the third indication
        P4 = 0.5  # Probability of success for the fourth indication

        x = make_root_state(4)

        # Create a 4-dimensional array for joint probability distribution
        joint_prob = np.zeros((2, 2, 2, 2))

        # Populate the array with joint probabilities
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        joint_prob[i][j][k][l] = ((P1 if i == 1 else 1 - P1) *
                                                       (P2 if j == 1 else 1 - P2) *
                                                       (P3 if k == 1 else 1 - P3) *
                                                       (P4 if l == 1 else 1 - P4))

        test_costs = np.array([0.1, 0.1, 0.1, 0.1])
        ind_demands = np.array([1, 1, 1, 1])
        prices = np.array([1, 1, 1, 1])
        discount_factor = 1 / (1 + 0.1)
        patent_window = 2
        tree = tree_start(x, joint_prob=to_tuple(joint_prob),
                          test_costs=tuple(test_costs),
                          ind_demands=tuple(ind_demands),
                          prices=tuple(prices),
                          discount_factor=discount_factor,
                          patent_window=patent_window,
                          allow_simult_tests=True)
        self.assertEqual(round(tree.State.EV - 1.4182, 4), 0.0)

    def test_no_demand(self):
        # set up tree
        K = 4
        x = make_root_state(K)
        specs = make_uniform_problem(K)
        specs['ind_demands'] = (0, 0, 0, 0)
        tree = tree_start(x, **specs)

        self.assertEqual(tree.State.EV, 0)


# class TestCalculateOutcomes(unittest.TestCase):
#     def setUp(self):
#         self.state = State(
#             Tests=(None, None, None, None),
#             Launched=(None, None, None, None),
#             Period=0,
#             PeriodValue=0.0,
#             EV=0.0
#         )
#         self.action = Action(Test=(0, 1))
#         self.joint_prob = np.array([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
#                                     [[[0.9, 0.8], [0.7, 0.6]], [[0.5, 0.4], [0.3, 0.2]]]])
#
#     def test_calculate_outcomes(self):
#         outcome_nodes, outcome_probs = calculate_outcomes(self.state, self.action)
#
#         # Check the number of outcomes
#         self.assertEqual(len(outcome_nodes), 4)  # 4 outcomes for 2 tests (2^2)
#         self.assertEqual(len(outcome_probs), 4)
#
#         # Check if probabilities are calculated correctly
#         expected_probs = [self.joint_prob[0][0][0][0], self.joint_prob[0][0][1][1],
#                           self.joint_prob[1][1][0][0], self.joint_prob[1][1][1][1]]
#         self.assertEqual(outcome_probs, expected_probs)


if __name__ == '__main__':
    unittest.main()
