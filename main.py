import numpy as np
import time
from collections import namedtuple

"""
TODO

"""


# state update
def f(x, a, u):
    """
    x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None, time period)
    a: action = tuple (test index or None to stop, launch index or None)
    u: test outcome = 1 for success, 0 for failure
    """
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    probs = x[4]
    test, launch = a

    if launched_first is None and launch is not None:
        launched_first = launch
        launched[launch] = 1
    if launched_first is not None:
        can_launch = [x if x is not None else 0 for x in test_results]
        launched = np.array([x if x is not None else 0 for x in launched])
        unlaunched = (can_launch - launched) * np.arange(1, K+1)
        launched += unlaunched

    if test is not None:
        test_results[test] = u[test]

    t_per += 1

    return (test_results, launched, launched_first, t_per, probs)


# payoff function
def g(x, a, test_cost):
    """
    x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
    a: action = tuple (test index or None to stop, launch index or None)
    test_cost: test cost data
    """

    payoff = 0
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    test, launch = a

    # test cost
    if test is not None:
        payoff -= test_cost[a[0]]
    # launch value for first indication launched
    if launched_first is None and launch is not None:
        launched_first = launch
        launched[launch] = 1
        payoff += ind_values[launched_first] * pricing_mults[launched_first]
    # launch value for subsequent indications
    if launched_first is not None:
        can_launch = np.array([x if x is not None else 0 for x in test_results])
        launched = np.array([x if x is not None else 0 for x in launched])
        unlaunched = (can_launch - launched) * np.arange(1, K+1)
        unlaunched_index = [x - 1 for x in unlaunched if x > 0]
        for i in unlaunched_index:
            payoff += ind_values[i] * pricing_mults[launched_first]

    return payoff


# action space
def actions(x):
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    no_tests = False

    # what can still be tested
    tests = [i for i, v in enumerate(test_results) if v is None]
    if len(tests) == 0:
        no_tests = True
    tests.append(None)

    # what can be launched first, if none launched yet
    if launched_first is None:
        launches = [(i, v) for i, v in enumerate(test_results)]
        launches = [i for i, v in launches if v == 1]
        if not no_tests:
            launches.append(None)
    else:
        launches = [None]

    action_list = [(t, l) for t in tests for l in launches]
    return action_list


def format_state(x, str_end=''):
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    ret_str = ('T=' + ''.join([str(i) if i is not None else '-' for i in test_results]) +
               ' | L=' + ''.join('1' if i == 1 else '0' for i in launched) +
               ' | F=' + str(launched_first if launched_first is not None else '-') +
               ' | Per=' + str(t_per) + ' |' +
               str_end)

    return ret_str


# value function
def vf(x):

    t_per = x[3]
    probs = x[4]
    Node = namedtuple('Node', ['Tests', 'Launched', 'First', 'Period'])

    action_set = actions(x)
    action_values = []
    for a in action_set:
        if a[0] is not None:
            k = a[0]
            success = [0] * K
            success[k] = 1
            failure = [0] * K
            failure[k] = 0
            value_in_period = g(x, a, test_costs)

            action_string = 'Test ' + str(k)
            if a[1] is not None:
                action_string += 'Launch ' + str(a[1])

            x_s = f(x, a, success)
            x_f = f(x, a, failure)
            ps_numerator = calc_marginal(x_s)
            ps_denominator = ps_numerator + calc_marginal(x_f)
            ps = ps_numerator / ps_denominator
            success_value = value_in_period + r * vf(x_s)
            str_end = 'CF=' + str(value_in_period) + '|' + ' P_S=' + str(round(ps, 3)) + ' | ' + 'EV=' + str(round(success_value, 3))
            print('\t' * (t_per * 2 + 1), format_state(x, str_end))

            failure_value = value_in_period + r * vf(x_f)
            str_end = 'CF=' + str(value_in_period) + '|' + ' P_F=' + str(round(1 - ps, 3)) + ' | ' + 'EV=' + str(round(failure_value, 3))
            print('\t' * (t_per * 2 + 1), format_state(x, str_end))

            value = ps * success_value + (1 - ps) * failure_value
            print('\t' * (t_per * 2), format_state(x), round(value, 3), '{' + action_string + '}')
        else:
            # case of no test but launch something
            value = g(x, a, test_costs)
            print('\t' * (t_per * 2), format_state(x), round(value, 3))
        action_values.append(value)
        # else:
        #     value = g(x, a=(None, None), test_cost=test_costs)
        #     action_values.append(value)
        #     print('\t' * (t_per * 2), round(value, 6))
        #     print('\t' * (t_per * 2), format_state(x), round(value, 6))
        #     print('XXXXXXXXXXXXXXXXX')

    if len(action_values) == 0:
        ret_val = 0
        action_string = 'Do Nothing'
    else:
        ret_val = max(action_values)
        action_index = action_values.index(ret_val)
        action_string = 'Test ' + str(action_set[action_index][0]) + ' Launch ' + str(action_set[action_index][1])

    tree[Node(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per)] = action_string

    # print('\t' * (t_per * 2), round(ret_val, 6))
    # print('\t' * (t_per * 2), format_state(x), round(ret_val, 6))
    return ret_val


def calc_marginal(x):
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    probs = x[4]
    dims = len(test_results.shape)

    for v in test_results:
        if v is not None:
            probs = probs[v]
        else:
            probs = np.sum(probs, axis=0)

    new_probs = np.sum(probs)
    return new_probs


if __name__ == '__main__':
    np.random.seed(1234)
    K = 3  # indications
    r = 1 / (1 + 0.0)  # discount factor

    # test_costs = np.random.random(K) / 10
    # ind_values = np.random.random(K)
    # pricing_mults = np.append(np.array([1]), np.random.random(K - 1))
    joint_probs = np.random.dirichlet(np.ones(2**K)).reshape(*([2]*K))
    # np.split(np.random.dirichlet(np.ones(2 ** 2)), 2)

    test_costs = np.array([0.1] * K)
    ind_values = np.array([1.0] * K)
    pricing_mults = np.array([1.0] + [0.9] * K)
    pricing_mults = np.array([1.0] * K)

    test_results = [None] * K
    # test_results = [1, None, None]
    launched = [None] * K
    launched_first = None
    t_per = 0
    # t_per = 1
    x = (test_results, launched, launched_first, t_per, joint_probs)
    tree = dict()

    State = namedtuple('State', ['Tests', 'Launched', 'First', 'Period', 'PeriodValue', 'EV', 'JointProb', 'Parent', 'Children'])

    tic = time.perf_counter()
    val = vf(x)
    toc = time.perf_counter()
    for t in tree.items():
        print(t)

    print('Joint Probabilities of Test Success')
    print(joint_probs)
    print('Value: ', val, '   Seconds:', str(toc - tic))

