import numpy as np
from itertools import combinations
from typing import List, Optional
import functools
import dataclasses

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
    IsTesting: bool = True
    IsLaunching: bool = True


@dataclasses.dataclass(frozen=True)
class Action:
    Test: int | None = None
    Launch: tuple[int | None] = None
    # Index: int | None = None


@dataclasses.dataclass(frozen=True)
class Node:
    State: State = None
    Action: Action = None
    Children: List = dataclasses.field(default_factory=list)
    Choice: int | None = None


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def patent_window_mod(t_per, window_len, r):
    time_left = window_len - t_per - 1
    if time_left > 0:
        npv_mult = np.sum(r ** (np.array(range(time_left))))
        return npv_mult
    else:
        return 0


# action space
@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def actions(test_results, launched, allow_testing=True):
    end_at_launch = False
    K = len(test_results)

    # what can still be tested
    tests: List[Optional[int]] = []
    if allow_testing:
        tests = [i for i, v in enumerate(test_results) if v is None]
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
        action_list = [Action(Test=t, Launch=None) for t in tests]
        action_list += [Action(Test=None, Launch=l) for l in launches]
    else:
        action_list = [Action(Test=t, Launch=l) for t in tests for l in launches]

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


def tree_start(x: State, joint_prob, test_costs, ind_demands, prices, discount_factor=1, patent_window=10) -> Node:
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
    #@functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def g(x: State, a) -> float:
        """
        x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
        a: action = tuple (test index or None to stop, launch index or None)
        test_cost: test cost data
        """

        # K = len(x.Tests)
        payoff = 0
        launched = np.array([0 if i is None else i for i in x.Launched])
        # if a.Launch is not None:
        #     for i in a.Launch:
        #         launched[i] = 1
        total_demand = launched * np.array(ind_demands)
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

    def vf_end(x: State, a: Action) -> Node:
        # action_set = actions(x.Tests, x.Launched, allow_testing=False)
        # action_values = []
        # action_list = []

        new_x = f(x, a, [0] * n_inds)
        period_value = g(new_x, a)

        if x.Period >= patent_window:
            # no more testing or launching and no more periods left
            remaining_value = 0
        else:
            # no more testing, but at least one period left
            remaining_value = period_value * patent_window_mod(x.Period, patent_window, discount_factor)

        value = period_value + discount_factor * remaining_value
        new_x = dataclasses.replace(new_x, PeriodValue=period_value, EV=value)
        child_node = Node(State=new_x, Action=a)

        # for a in action_set:
        #     period_value = g(x, a)
        #     value = period_value
        #     remaining_value = period_value * patent_window_mod(x.Period, patent_window, discount_factor)
        #     value += discount_factor * remaining_value
        #     new_x = dataclasses.replace(x, PeriodValue=period_value, EV=value)
        #     child_node = Node(State=new_x, Action=a)
        #     action_list.append(child_node)
        #     action_values.append(child_node.State.EV)
        #
        # best_value = 0
        # choice = None
        # if len(action_values) > 0:
        #     best_value = max(action_values)
        #     choice = action_values.index(best_value)
        #
        # new_x = dataclasses.replace(x, EV=best_value)
        # node_action = None
        # if choice is not None:
        #     node_action = action_list[choice].Action
        #
        # node = Node(State=new_x,
        #             Action=node_action,
        #             Children=action_list,
        #             Choice=choice)

        return child_node


    # value function
    # @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def vf(x: State, tree_start=False) -> Node:
        action_set = actions(x.Tests, x.Launched, x.IsTesting)
        action_values = []
        action_list = []

        # if not the first period but nothing has been tested, then the action_set = {} and quit
        if x.Period > 0 and all(t is None for t in x.Tests):
            action_set = []

        for a in action_set:
            if a.Test is not None:
                child_node = vf_test(x, a)
            else:
                # Once you stop testing, you can't start again.
                child_node = vf_end(x, a)

            action_list.append(child_node)
            action_values.append(child_node.State.EV)

        best_value = 0
        choice = None
        if len(action_values) > 0:
            best_value = max(action_values)
            choice = action_values.index(best_value)

        new_x = dataclasses.replace(x, EV=best_value)

        node = Node(State=new_x, Action=None, Children=action_list, Choice=choice)

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

    return vf(x, tree_start=True)
