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
    test_results = [None] * K
    # test_results = [1, None, None]
    launched = [None] * K
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
            x = (test_results,
                 launched,
                 launched_first,
                 t_per,
                 joint_probs,
                 test_costs,
                 ind_values,
                 pricing_mults,
                 r,
                 launch_period)

            x = State(Tests=tuple(test_results),
                      Launched=tuple(launched),
                      First=launched_first,
                      Period=t_per,
                      PeriodValue=0,
                      EV=0,
                      JointProb=to_tuple(joint_probs),
                      Action=None,
                      LaunchPer=launch_period,
                      TestCost=tuple(test_costs),
                      IndicationValues=tuple(ind_values),
                      PricingMults=tuple(pricing_mults),
                      DiscountRate=r)

            tic = time.perf_counter()
            # tree = vf(pickle.dumps(x))
            tree = vf(x)
            toc = time.perf_counter()
            times.append(toc - tic)
            excel_value = round(excel_value, 5)
            val = round(tree['value'].EV, 5)
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
