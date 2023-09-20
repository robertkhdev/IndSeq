import numpy as np
from itertools import combinations
from typing import List, Optional
import functools
import dataclasses

PATENT_WINDOW = 10
CACHE_MAXSIZE = None


@dataclasses.dataclass(frozen=True)
class State:
    # which indications have been tested and passed (1) or failed (0)
    Tests: tuple[Optional[int], ...] = None
    # which indications have been launched
    Launched: tuple[Optional[int], ...] = None
    Period: Optional[int] = None
    PeriodValue: Optional[float] = None
    EV: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class Action:
    Test: int | None = None
    Launch: tuple[int | None] = None
    Index: int | None = None


@dataclasses.dataclass(frozen=True)
class Node:
    State: State = None
    Action: Action = None
    Children: List = dataclasses.field(default_factory=list)
    Choice: int | None = None


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def patent_window_mod(t_per, launch_per, window_len, r):
    time_left = max(0, launch_per + window_len - t_per)
    if time_left > 0:
        full_value = np.sum((1 + r) ** -np.array(range(window_len)))
        partial_value = np.sum((1 + r) ** -np.array(range(time_left)))
        return partial_value / full_value
    else:
        return 0


# action space
@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def actions(test_results, launched):
    # test_results = x.Tests
    # launched_first = x.First
    end_at_launch = False
    no_tests = False
    K = len(test_results)

    # what can still be tested
    tests: List[Optional[int]] = [i for i, v in enumerate(test_results) if v is None]
    if len(tests) == 0:
        no_tests = True
    tests.append(None)

    # what can be launched
    can_launch = np.array([x if x is not None else 0 for x in test_results])
    launched_01 = np.array([1 if x is not None else 0 for x in launched])
    unlaunched = (can_launch - launched_01) * np.arange(1, K + 1)
    unlaunched_index = [x - 1 for x in unlaunched if x > 0]

    launches = []
    if len(unlaunched_index) > 0:
        combos = [list(combinations(unlaunched_index, i+1)) for i in range(K)]
        launches = [item for combolist in combos for item in combolist]
    launches += [None]

    if end_at_launch:
        action_list = [Action(Test=t, Launch=None, Index=None) for t in tests]
        action_list += [Action(Test=None, Launch=l, Index=None) for l in launches]
    else:
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


def tree_start(x: State, joint_prob, test_costs, ind_demands, prices, discount_factor) -> Node:
    """
    Entry point for tree algorithm.
    :param x:
    :param joint_prob:
    :param test_costs:
    :param ind_demands:
    :param prices:
    :param discount_factor:
    :return:
    """
    # patent_window_mod.cache_clear()
    # actions.cache_clear()
    # success_prob.cache_clear()
    # calc_marginal.cache_clear()

    n_inds = len(ind_demands)

    # payoff function
    @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def g(x: State, a) -> float:
        """
        x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
        a: action = tuple (test index or None to stop, launch index or None)
        test_cost: test cost data
        """

        K = len(x.Tests)
        payoff = 0
        launched = np.array([0 if i is None else i for i in x.Launched])
        total_demand = launched * ind_demands
        if sum(launched) > 0:
            price = np.min(np.array(prices)[launched > 0])
            # value from launched indications
            payoff += np.sum(price * total_demand)

        # test cost
        if a.Test is not None:
            payoff -= test_costs[a.Test]

        return payoff

    # @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def vf_test(x: State, a) -> Node:
        k = a.Test
        success = [0] * n_inds
        success[k] = 1
        failure = [0] * n_inds
        failure[k] = 0

        value_in_period = g(x, a)
        # value_in_period = 0

        x_s = f(x, a, success)
        x_f = f(x, a, failure)
        ps = success_prob(x_s.Tests, joint_prob, x_f.Tests, joint_prob)

        success_node = vf(x_s)
        success_value = success_node.State.EV

        failure_node = vf(x_f)
        failure_value = failure_node.State.EV

        value = value_in_period + discount_factor * (ps * success_value + (1 - ps) * failure_value)
        # value = ps * success_value + (1 - ps) * failure_value
        new_x = dataclasses.replace(x, PeriodValue=value_in_period, EV=value)
        node = Node(State=new_x, Children=[success_node, failure_node], Action=a)
        return node

    # value function
    # @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def vf(x: State) -> Node:
        action_set = actions(x.Tests, x.Launched)
        action_values = []
        action_list = []

        for a in action_set:
            if a.Test is not None:
                child_node = vf_test(x, a)
            else:
                successes = np.sum(np.array([z for z in x.Tests if z is not None]))
                period_value = g(x, a)
                new_x = f(x, a, [0] * n_inds)
                if x.Period < PATENT_WINDOW:
                    next_node = vf(new_x)
                    remaining_value = next_node.State.EV
                else:
                    remaining_value = 0
                value = period_value + discount_factor * remaining_value
                new_x = dataclasses.replace(new_x, PeriodValue=period_value, EV=value)
                child_node = Node(State=new_x, Action=a)
            action_list.append(child_node)
            action_values.append(child_node.State.EV)

        if len(action_set) == 0:
            periods_remaining = PATENT_WINDOW - x.Period
            period_value = g(x, Action(Test=None, Launch=(None,), Index=None))
            disc = np.sum((1 + discount_factor) ** -np.array(range(periods_remaining)))
            remaining_value = np.sum(period_value * disc)

        best_value = 0
        choice = None
        if len(action_values) > 0:
            best_value = max(action_values)
            choice = action_values.index(best_value)

        new_x = dataclasses.replace(x, EV=best_value)
        node_action = None
        if choice is not None:
            node_action = action_list[choice].Action

        node = Node(State=new_x,
                    Action=node_action,
                    Children=action_list,
                    Choice=choice)

        return node

    # state update
    def f(x: State, a, u) -> State:
        """
        x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None, time period)
        a: action = tuple (test index or None to stop, launch index or None)
        u: test outcome = 1 for success, 0 for failure
        """
        test_results = x.Tests
        launched = x.Launched
        t_per = x.Period
        test = a.Test
        if a.Launch is not None:
            successes = np.array([tr if tr is not None else 0 for tr in test_results])
            launched_prev = np.array([launch if launch is not None else 0 for launch in launched])
            launched_01 = np.array([1 if launch is not None else 0 for launch in launched])
            can_launch = successes - launched_01
            launched_now = can_launch
            launched = launched_prev + launched_now
            launched = tuple([launch if launch != 0 else None for launch in launched])

        if test is not None:
            test_results = list(test_results)
            test_results[test] = u[test]
            test_results = tuple(test_results)

        t_per += 1

        new_x = dataclasses.replace(x,
                                    Tests=test_results,
                                    Launched=launched,
                                    Period=t_per)

        return new_x

    return vf(x)
