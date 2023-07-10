import numpy as np
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional
import functools
import dataclasses

# TODO convert lists to tuples wherever possible

PATENT_WINDOW = 2000
CACHE_MAXSIZE = None


# TODO update all uses of "state" to use the State dataclass
# TODO add type hints to State class
@dataclasses.dataclass(frozen=True)
class State:
    Tests: tuple = None
    Launched: tuple = None
    First: int | None = None
    Period: int | None = None
    PeriodValue: float | None = None
    EV: float | None = None
    JointProb: tuple = None
    Action: Action = None
    LaunchPer: tuple = None
    TestCost: tuple = None
    IndicationValues: tuple = None
    PricingMults: tuple = None
    DiscountRate: float = None


@dataclasses.dataclass(frozen=True)
class Action:
    Test: any
    Launch: any
    Index: any


# state update
def f(x: State, a, u):
    """
    x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None, time period)
    a: action = tuple (test index or None to stop, launch index or None)
    u: test outcome = 1 for success, 0 for failure
    """
    test_results = x.Tests
    launched = x.Launched
    launched_first = x.First
    t_per = x.Period
    launch_period = x.LaunchPer
    test = a.Test
    launch = a.Launch
    if launched_first is None and launch is not None:
        launched_first = launch
        launched = list(launched)
        launched[launch] = t_per
        launched = tuple(launched)
        launch_period = t_per
    if launched_first is not None:
        successes = np.array([tr if tr is not None else 0 for tr in test_results])
        launched_prev = np.array([launch if launch is not None else 0 for launch in launched])
        launched_01 = np.array([1 if launch is not None else 0 for launch in launched])
        can_launch = successes - launched_01
        launched_now = can_launch * t_per
        launched = launched_prev + launched_now
        launched = tuple([launch if launch != 0 else None for launch in launched])

    if test is not None:
        test_results = list(test_results)
        test_results[test] = u[test]
        test_results = tuple(test_results)

    t_per += 1

    value_in_period = g(x, a)
    value = value_in_period + x.DiscountRate * x.EV
    action = Action(Test=test, Launch=launch, Index=None)
    new_x = dataclasses.replace(x,
                                Tests=test_results,
                                Launched=launched,
                                First=launched_first,
                                Period=t_per,
                                PeriodValue=value_in_period,
                                EV=value,
                                Action=action,
                                LaunchPer=launch_period)

    return new_x


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def patent_window_mod(t_per, launch_per, window_len, r):
    time_left = max(0, launch_per + window_len - t_per)
    if time_left > 0:
        full_value = np.sum((1 + r) ** -np.array(range(window_len)))
        partial_value = np.sum((1 + r) ** -np.array(range(time_left)))
        return partial_value / full_value
    else:
        return 0


# payoff function
@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def g(x: State, a) -> float:
    """
    x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
    a: action = tuple (test index or None to stop, launch index or None)
    test_cost: test cost data
    """

    payoff = 0

    test_results = x.Tests
    launched = x.Launched
    launched_first = x.First
    t_per = x.Period
    test_costs = x.TestCost
    ind_values = x.IndicationValues
    pricing_mults = x.PricingMults
    r = x.DiscountRate
    launch_period = x.LaunchPer

    test = a.Test
    launch = a.Launch
    K = len(test_results)

    # TODO a lot of this is similar to code in f() - refactor?
    # test cost
    if test is not None:
        payoff -= test_costs[a.Test]
    # launch value for first indication launched
    if launched_first is None and launch is not None:
        launched_first = launch
        launched = list(launched)
        launched[launch] = t_per
        launched = tuple(launched)
        payoff += ind_values[launched_first] * pricing_mults[launched_first]
        launch_period = t_per
    # launch value for subsequent indications
    if launched_first is not None:
        can_launch = np.array([x if x is not None else 0 for x in test_results])
        launched_01 = np.array([1 if x is not None else 0 for x in launched])
        # launched_01 = np.array([min(x, 1) for x in launched_clean])

        unlaunched = (can_launch - launched_01) * np.arange(1, K + 1)
        unlaunched_index = [x - 1 for x in unlaunched if x > 0]
        for i in unlaunched_index:
            payoff += ind_values[i] * pricing_mults[launched_first] * patent_window_mod(t_per, launch_period,
                                                                                        PATENT_WINDOW, r)

    return payoff


# action space
@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def actions(test_results, launched_first):
    # test_results = x.Tests
    # launched_first = x.First

    no_tests = False

    # what can still be tested
    tests: List[Optional[int]] = [i for i, v in enumerate(test_results) if v is None]
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
            launches.append(None)
    else:
        launches = [None]

    action_list = [Action(Test=t, Launch=l, Index=None) for t in tests for l in launches]
    return action_list


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def success_prob(s_test_results, s_probs, f_test_results, f_probs) -> float:
    ps_numerator = calc_marginal(s_test_results, s_probs)
    ps_denominator = ps_numerator + calc_marginal(f_test_results, f_probs)
    ps = ps_numerator / ps_denominator
    return ps


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def calc_marginal(test_results, probs):
    # test_results = x.Tests
    # probs = x.JointProb

    for v in test_results:
        if v is not None:
            probs = probs[v]
        else:
            probs = np.sum(probs, axis=0)

    new_probs = np.sum(probs)
    return new_probs


# @functools.lru_cache(maxsize=CACHE_MAXSIZE)
def vf_test(x: State, a) -> Dict:
    k = a.Test
    r = x.DiscountRate
    n_tests = len(x.Tests)
    success = [0] * n_tests
    success[k] = 1
    failure = [0] * n_tests
    failure[k] = 0

    value_in_period = g(x, a)

    x_s = f(x, a, success)
    x_f = f(x, a, failure)
    ps = success_prob(x_s.Tests, x_s.JointProb, x_f.Tests, x_f.JointProb)

    success_dict = vf(x_s)
    success_value = success_dict['value'].EV

    failure_dict = vf(x_f)
    failure_value = failure_dict['value'].EV

    value = value_in_period + r * (ps * success_value + (1 - ps) * failure_value)
    new_x = dataclasses.replace(x,
                                PeriodValue=value_in_period,
                                EV=value,
                                Action=a)

    node_dict = {'value': new_x, 'children': [success_dict, failure_dict], 'choice': None}

    return node_dict


# @functools.lru_cache(maxsize=CACHE_MAXSIZE)
def vf_endpoint(x: State, a) -> Dict:
    launch_state = f(x, a, x.Tests)
    node_dict = {'value': launch_state, 'children': [], 'choice': None}
    return node_dict


# value function
# @functools.lru_cache(maxsize=CACHE_MAXSIZE)
def vf(x: State) -> Dict:
    """

    :rtype: Dict
    """
    action_set = actions(x.Tests, x.First)
    action_values = []
    action_list = []

    for a in action_set:
        if a.Test is not None:
            node_dict = vf_test(x, a)
        else:
            node_dict = vf_endpoint(x, a)
        action_list.append(node_dict)
        action_values.append(node_dict['value'].EV)

    best_value = max(action_values)
    choice = action_values.index(best_value)

    new_x = dataclasses.replace(x, EV=action_values[choice],
                                Action=action_list[choice])

    node_dict = {'value': new_x, 'children': action_list, 'choice': choice}

    return node_dict


def tree_start(x: State) -> Dict:
    """
    Entry point for tree algorithm.
    Ensures caches are clear before starting tree calculations.
    :param x:
    :return :
    """
    patent_window_mod.cache_clear()
    g.cache_clear()
    actions.cache_clear()
    success_prob.cache_clear()
    calc_marginal.cache_clear()

    return vf(x)
