
import numpy as np
import pandas as pd
import json
import time
from collections import Counter, namedtuple
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional
import functools
import pickle
import dataclasses

# TODO convert lists to tuples wherever possible

PATENT_WINDOW = 20


# TODO add type hints to State class
@dataclasses.dataclass(frozen=True)
class State:
    Tests: any
    Launched: any
    First: any
    Period: any
    PeriodValue: any
    EV: any
    JointProb: any
    Action: any
    LaunchPer: any


# State = namedtuple('State',
#                    ['Tests',
#                     'Launched',
#                     'First',
#                     'Period',
#                     'PeriodValue',
#                     'EV',
#                     'JointProb',
#                     'Action',
#                     'LaunchPer'])


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
    test_costs = x[5]
    ind_values = x[6]
    pricing_mults = x[7]
    r = x[8]
    launch_period = x[9]
    test, launch = a
    K = len(test_results)

    if launched_first is None and launch is not None:
        launched_first = launch
        launched[launch] = 1
        launch_period = t_per
    if launched_first is not None:
        can_launch = [x if x is not None else 0 for x in test_results]
        launched = np.array([x if x is not None else 0 for x in launched])
        launched_clean = np.array([x if x is not None else 0 for x in launched])
        launched_01 = np.array([min(x, 1) for x in launched_clean])
        unlaunched = (can_launch - launched_01) * np.arange(1, K + 1)
        launched = launched_01 + unlaunched
        launched = [int(launch) for launch in launched]

    if test is not None:
        test_results[test] = u[test]

    t_per += 1

    return (test_results, launched, launched_first, t_per, probs, test_costs,
            ind_values, pricing_mults, r, launch_period)


def patent_window_mod(t_per, launch_per, window_len, r):
    time_left = max(0, launch_per+window_len-t_per)
    if time_left > 0:
        full_value = np.sum((1 + r) ** -np.array(range(window_len)))
        partial_value = np.sum((1 + r) ** -np.array(range(time_left)))
        return partial_value / full_value
    else:
        return 0


# payoff function
@functools.cache
def g(x, a):
    """
    x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
    a: action = tuple (test index or None to stop, launch index or None)
    test_cost: test cost data
    """

    x = pickle.loads(x)
    payoff = 0
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    test_costs = x[5]
    ind_values = x[6]
    pricing_mults = x[7]
    r = x[8]
    launch_period = x[9]
    test, launch = a
    K = len(test_results)

    # test cost
    if test is not None:
        payoff -= test_costs[a[0]]
    # launch value for first indication launched
    if launched_first is None and launch is not None:
        launched_first = launch
        launched[launch] = 1
        payoff += ind_values[launched_first] * pricing_mults[launched_first]
        launch_period = t_per
    # launch value for subsequent indications
    if launched_first is not None:
        can_launch = np.array([x if x is not None else 0 for x in test_results])
        launched_clean = np.array([x if x is not None else 0 for x in launched])
        launched_01 = np.array([min(x, 1) for x in launched_clean])

        unlaunched = (can_launch - launched_01) * np.arange(1, K + 1)
        unlaunched_index = [x - 1 for x in unlaunched if x > 0]
        for i in unlaunched_index:
            payoff += ind_values[i] * pricing_mults[launched_first] * patent_window_mod(t_per, launch_period, PATENT_WINDOW, r)

    return payoff


# action space
@functools.cache
def actions(x):
    x = pickle.loads(x)
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


def success_prob(x_s, x_f):
    ps_numerator = calc_marginal(x_s)
    ps_denominator = ps_numerator + calc_marginal(x_f)
    ps = ps_numerator / ps_denominator
    return ps


@functools.cache
def vf_test(x, a) -> Dict:
    x = pickle.loads(x)
    k = a[0]
    t_per = x[3]
    r = x[8]
    n_tests = len(x[0])
    success = [0] * n_tests
    success[k] = 1
    failure = [0] * n_tests
    failure[k] = 0

    value_in_period = g(pickle.dumps(x), a)

    x_s = f(x, a, success)
    x_f = f(x, a, failure)
    ps = success_prob(x_s, x_f)

    success_dict = vf(pickle.dumps(x_s))
    success_value = success_dict['value'].EV

    failure_dict = vf(pickle.dumps(x_f))
    failure_value = failure_dict['value'].EV

    value = value_in_period + r * (ps * success_value + (1 - ps) * failure_value)
    current_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                          PeriodValue=value_in_period, EV=value, JointProb=1,
                          Action={'Test': a[0], 'Launch': a[1], 'Index': None}, LaunchPer=x[9])

    node_dict = {'value': current_state, 'children': [success_dict, failure_dict]}

    return node_dict


@functools.cache
def vf_launch_without_test(x, a):
    value_in_period = g(x, a)
    value = value_in_period
    x = pickle.loads(x)
    t_per = x[3]
    if x[9] is None:
        launch_period = t_per
    else:
        launch_period = x[9]
    launch_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                         PeriodValue=g(pickle.dumps(x), a), EV=value, JointProb=1,
                         Action={'Test': a[0], 'Launch': a[1], 'Index': None}, LaunchPer=launch_period)
    node_dict = {'value': launch_state, 'children': []}
    return node_dict


@functools.cache
def vf_stop(x, a) -> Dict:
    x = pickle.loads(x)
    t_per = x[3]
    value_in_period = g(pickle.dumps(x), a)
    value = value_in_period
    stop_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                       PeriodValue=g(pickle.dumps(x), a), EV=value, JointProb=0,
                       Action={'Test': None, 'Launch': None, 'Index': None},
                       LaunchPer=None)
    node_dict = {'value': stop_state, 'children': []}
    return node_dict


# value function
@functools.cache
def vf(x) -> Dict:

    action_set = actions(x)
    action_values = []
    action_list = []
    node_dict = dict()

    if len(action_set) > 0:
        # there are one or more actions
        for a in action_set:
            if a[0] is not None:
                # do a test
                node_dict = vf_test(x, a)
            elif a[1] is not None:
                # case of no test but launch something -> at end node in tree
                node_dict = vf_launch_without_test(x, a)
            else:
                # no test or launch, so this is an endpoint where we stop
                node_dict = vf_stop(x, a)
            action_values.append(node_dict['value'].EV)
            action_list.append(node_dict)

        ret_val = max(action_values)
        choice_idx = action_values.index(ret_val)
        value_in_period = action_list[choice_idx]['value'].EV
        launch_period = action_list[choice_idx]['value'].LaunchPer
        action = {'Test': None, 'Launch': None, 'Index': choice_idx}
        x = pickle.loads(x)
        t_per = x[3]
        node_dict = {'value': State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                    PeriodValue=value_in_period, EV=ret_val, JointProb=0, Action=action,
                                    LaunchPer=launch_period),
                     'children': action_list}

    if len(action_values) == 0:
        # uncertainty node
        action = {'Test': None, 'Launch': None, 'Index': None}
        x = pickle.loads(x)
        node_dict = {'value': State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=x[3],
                                    PeriodValue=0, EV=0, JointProb=0, Action=action, LaunchPer=x[9]),
                     'children': []}

    return node_dict


def calc_marginal(x):
    test_results = np.array(x[0])
    probs = x[4]
    for v in test_results:
        if v is not None:
            probs = probs[v]
        else:
            probs = np.sum(probs, axis=0)

    new_probs = np.sum(probs)
    return new_probs

