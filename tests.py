import numpy as np
import pandas as pd
import time

import randmodels
from indseqtree import *
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
                      First=launched_first,
                      Period=t_per,
                      PeriodValue=0,
                      EV=0,
                      LaunchPer=launch_period)

            tic = time.perf_counter()
            tree = tree_start(x, joint_prob=to_tuple(joint_probs),
                              test_costs=tuple(test_costs),
                              ind_values=tuple(ind_values),
                              pricing_mults=tuple(pricing_mults),
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