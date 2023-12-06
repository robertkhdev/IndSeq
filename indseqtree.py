import numpy as np
import itertools
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
    Action: Optional[Action] = None
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
# @functools.lru_cache(maxsize=CACHE_MAXSIZE)
def possible_actions(state: State) -> List[Action]:
    """
    Generate a list of all possible actions that can be taken from a given state,
    including testing, launching, both, or neither.

    Parameters:
    state (State): The current state of the decision process.

    Returns:
    List[Action]: A list of possible actions.
    """
    actions = []
    num_indications = len(state.Tests)

    # Possible tests (including the option of not testing anything)
    possible_tests = [i for i in range(num_indications) if state.Tests[i] is None] + [None]

    # Possible launches (including the option of not launching anything)
    launchable = [i for i, test_result in enumerate(state.Tests) if test_result == 1 and (state.Launched is None or state.Launched[i] is None)]
    possible_launches = [None] + [tuple(comb) for r in range(1, len(launchable) + 1) for comb in itertools.combinations(launchable, r)]

    # Generate all combinations of tests and launches
    for test in possible_tests:
        for launch in possible_launches:
            actions.append(Action(Test=test, Launch=launch))

    return actions


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
        total_demand = launched * np.array(ind_demands)
        if sum(launched) > 0:
            price = np.min(np.array(prices)[launched > 0])
            # value from launched indications
            payoff += np.sum(price * total_demand)

        # test cost
        if a.Test is not None:
            payoff -= test_costs[a.Test]

        return payoff

    def vf_combined(x: State) -> Node:
        # Determine possible actions from the current state
        action_set = possible_actions(x)

        # If no actions are possible, return the current state as a terminal node
        if not action_set:
            return Node(State=x)

        # List to store child nodes and their corresponding EVs
        action_values = []
        action_list = []

        for a in action_set:
            new_x = f(x, a, None)  # Update the state for any launches
            value_in_period = g(new_x, a)  # Calculate the immediate payoff of the action

            # Determine the expected value based on the action type
            if a.Test is not None:
                # Handle test actions
                k = a.Test
                success = [0] * n_inds
                success[k] = 1
                failure = [0] * n_inds
                failure[k] = 0

                x_s = f(x, a, success)
                x_f = f(x, a, failure)
                success_node = vf_combined(x_s)
                failure_node = vf_combined(x_f)
                success_value = success_node.State.EV  # Recursive call
                failure_value = failure_node.State.EV  # Recursive call

                ps = success_prob(x_s.Tests, joint_prob, x_f.Tests, joint_prob)
                value = value_in_period + discount_factor * (ps * success_value + (1 - ps) * failure_value)
                new_x = dataclasses.replace(new_x, PeriodValue=value_in_period, EV=value)
                child_node = Node(State=new_x, Children=[success_node, failure_node], Action=a)
            else:
                # Handle end actions or no action
                remaining_value = value_in_period * patent_window_mod(x.Period, patent_window, discount_factor)
                value = value_in_period + discount_factor * remaining_value
                new_x = dataclasses.replace(new_x, PeriodValue=value_in_period, EV=value)
                child_node = Node(State=new_x, Action=a)

            # add to the action list
            action_list.append(child_node)
            action_values.append(value)

        # Choose the action with the highest EV
        best_value, choice = max((value, idx) for idx, value in enumerate(action_values))
        new_x = dataclasses.replace(x, EV=best_value)
        return Node(State=new_x, Children=action_list, Choice=choice)

    # state update
    def f(x: State, a, u) -> State:
        """
        Function to update the state based on the action taken and the test outcome.

        Parameters:
        x (State): The current state, including test results, launched indications, and the current time period.
        a (Action): The action to be taken, including the indication to test and to launch.
        u (Test Outcome): Outcome of the test, 1 for success, 0 for failure.

        Returns:
        State: The new updated state.
        """

        # Extract current test results and launched indications from the state
        test_results = x.Tests
        launched = x.Launched
        t_per = x.Period
        test = a.Test

        # If there's an action to launch indications
        if a.Launch is not None:
            launching = np.array([1 if i in a.Launch else 0 for i in range(n_inds)])

            # Convert current launch status to binary (launched or not)
            launched_prev = np.array([launch if launch is not None else 0 for launch in x.Launched])

            # Update the launched status based on current action
            launched = launched_prev + launching

            # Convert back to the original format (None for not launched)
            launched = tuple([launch if launch != 0 else None for launch in launched])

        # If there's an action to test an indication
        if test is not None and u is not None:
            # Update the test results with the new outcome
            test_results = list(test_results)
            test_results[test] = u[test]
            test_results = tuple(test_results)

            # Increment the time period as a test consumes a time unit
            t_per += 1

        # Create a new state with updated values
        new_x = dataclasses.replace(x,
                                    Tests=test_results,
                                    Launched=launched,
                                    Period=t_per)

        return new_x

    return vf_combined(x)
