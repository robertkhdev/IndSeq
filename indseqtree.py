import numpy as np
import itertools
from typing import List, Optional
import functools
import dataclasses

CACHE_MAXSIZE = None


# Global configuration
class Config:
    ALLOW_SIMULTANEOUS_TESTING = False


# Example of changing the setting
# Config.ALLOW_SIMULTANEOUS_TESTING = True  # Allow simultaneous testing
Config.ALLOW_SIMULTANEOUS_TESTING = False  # Disallow simultaneous testing


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
    # vector of prices for each indication
    Prices: tuple[float, ...] = None


@dataclasses.dataclass(frozen=True)
class Action:
    Test: tuple[int, ...] | None = None
    Launch: tuple[int | None] = None
    # Index: int | None = None


@dataclasses.dataclass(frozen=True)
class Node:
    State: State = None
    Action: Optional[Action] = None
    Children: List = dataclasses.field(default_factory=list)
    Choice: int | None = None
    Probabilities: tuple[float, ...] = None


@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def patent_window_mod(t_per, window_len, r):
    time_left = window_len - t_per - 1
    if time_left > 0:
        npv_mult = np.sum(r ** (np.array(range(time_left))))
        return npv_mult
    else:
        return 0


def possible_actions(state: State) -> List[Action]:
    actions = []
    num_indications = len(state.Tests)
    testable = [i for i in range(num_indications) if state.Tests[i] is None]

    if Config.ALLOW_SIMULTANEOUS_TESTING:
        # Generate combinations of tests if simultaneous testing is allowed
        actions.extend(generate_action_combinations((None,), state))
        for r in range(1, len(testable) + 1):
            test_combinations = list(itertools.combinations(testable, r))
            for test_combination in test_combinations:
                actions.extend(generate_action_combinations(test_combination, state))
    else:
        # Only individual tests or no test if simultaneous testing is not allowed
        for test in testable + [None]:
            actions.extend(generate_action_combinations((test,), state))

    return actions


def generate_action_combinations(test_combination, state):
    # Generate all combinations of launches for a given test combination
    launch_combinations = []
    launchable = [i for i, result in enumerate(state.Tests) if
                  result == 1 and (state.Launched is None or state.Launched[i] is None)]
    possible_launches = [None] + [tuple(comb) for r in range(1, len(launchable) + 1) for comb in
                                  itertools.combinations(launchable, r)]

    for launch in possible_launches:
        launch_combinations.append(
            Action(Test=test_combination if test_combination != (None,) else None, Launch=launch))

    return launch_combinations


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


def tree_start(x: State,
               joint_prob,
               test_costs,
               ind_demands,
               prices,
               discount_factor=1,
               patent_window=10,
               allow_simult_tests=None,
               pricing_method='minimum') -> Node:
    """
    Entry point for tree algorithm.

    """
    # patent_window_mod.cache_clear()
    # actions.cache_clear()
    # success_prob.cache_clear()
    # calc_marginal.cache_clear()

    if allow_simult_tests is not None:
        Config.ALLOW_SIMULTANEOUS_TESTING = allow_simult_tests

    # payoff function
    #@functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def period_cash_flow(x: State, a) -> float:
        """
        x: state = ([test results 1=success, 0=fail, None=not tested], [indications launched 1=launch 0=not], first launched index or None)
        a: action = tuple (test index or None to stop, launch index or None)
        test_cost: test cost data
        """

        # K = len(x.Tests)
        payoff = 0
        launched = np.array([0 if i is None else i for i in x.Launched])
        effective_demands = launched * np.array(ind_demands)
        if sum(launched) > 0:
            # value from launched indications
            payoff += np.sum(x.Prices * effective_demands)

        # test cost
        if a.Test is not None:
            payoff -= sum(test_costs[i] for i in a.Test)

        return payoff

    def vf_combined(x: State) -> Node:
        action_set = possible_actions(x)

        action_values, action_list = [], []
        for a in action_set:
            child_node, value = process_action(x, a)
            action_list.append(child_node)
            action_values.append(value)

        best_value, choice = max((value, idx) for idx, value in enumerate(action_values))
        new_x = dataclasses.replace(x, EV=best_value)
        return Node(State=new_x, Children=action_list, Choice=choice)

    def process_action(x: State, a: Action) -> (Node, float):
        """
        Process a single action to determine its outcome and expected value.
        """
        new_x = state_update(x, a, None)  # Update state for any launches
        value_in_period = period_cash_flow(new_x, a)

        if a.Test is not None and x.Period + 1 < patent_window:
            outcome_nodes, outcome_probs = calculate_outcomes(x, a)
            total_prob = sum(outcome_probs)
            value = value_in_period + discount_factor * sum(
                outcome['Prob'] / total_prob * outcome['Value'] for outcome in outcome_nodes)
            new_x = dataclasses.replace(new_x, PeriodValue=value_in_period, EV=value)
            return Node(State=new_x, Children=outcome_nodes, Action=a, Probabilities=tuple(outcome_probs)), value
        else:
            return process_end_action(new_x, value_in_period)

    def calculate_outcomes(x: State, a: Action) -> (List[Node], List[float]):
        """
        Calculate all possible outcomes for a given action.
        """
        test_outcomes = list(itertools.product([0, 1], repeat=len(a.Test)))
        outcome_nodes, outcome_probs = [], []

        for outcome in test_outcomes:
            test_outcome = update_test_results(x.Tests, a.Test, outcome)
            new_x_updated = state_update(x, a, test_outcome)
            new_node = vf_combined(new_x_updated)  # Recursive call
            marginal_prob = calc_marginal(test_outcome, joint_prob)
            outcome_probs.append(marginal_prob)
            outcome_nodes.append({'Node': new_node, 'Prob': marginal_prob, 'Value': new_node.State.EV})

        return outcome_nodes, outcome_probs

    def update_test_results(tests: tuple, tests_to_update: tuple, outcomes: tuple) -> tuple:
        """
        Update test results based on outcomes.
        """
        test_results = list(tests)
        for test, outcome in zip(tests_to_update, outcomes):
            test_results[test] = outcome
        return tuple(test_results)

    def process_end_action(new_x: State, value_in_period: float) -> (Node, float):
        """
        Process an action that signifies the end of testing.
        """
        remaining_value = patent_window_mod(new_x.Period, patent_window, discount_factor)
        remaining_value = value_in_period * patent_window_mod(new_x.Period, patent_window, discount_factor)
        value = value_in_period + discount_factor * remaining_value
        new_x = dataclasses.replace(new_x, PeriodValue=value_in_period, EV=value)
        return Node(State=new_x, Action=None), value

    def state_update(x: State, a, u) -> State:
        test_results = list(x.Tests)
        launched = list(x.Launched)
        t_per = x.Period
        if x.Prices is None:
            effective_prices = np.zeros(len(x.Tests))
        else:
            effective_prices = x.Prices

        # Handle multiple tests if simultaneous testing is allowed
        if a.Test is not None and u is not None:
            for test in a.Test:
                test_results[test] = u[test]  # Update based on the outcome of each test
            # Increment the time period
            t_per += 1

        # Handle launching
        if a.Launch is not None:
            for launch in a.Launch:
                if launch is not None:
                    launched[launch] = 1
            # if this is the first launch and first_launched pricing is used
            if pricing_method == 'first_launched' and x.Launched is None:
                # use max price of launched indications, if multiple launched "first"
                launched_first_price = max([prices[i] for i in a.Launch])
                # set all prices to the first launched indication price
                effective_prices = np.ones(len(prices)) * launched_first_price

        # Update prices
        launched01 = np.array([0 if i is None else i for i in launched])
        if sum(launched01) > 0:
            if pricing_method == 'minimum':
                effective_prices = np.min(np.array(prices)[launched01 > 0]) * np.ones_like(prices)
            elif pricing_method == 'average':
                effective_prices = np.mean(np.array(prices)[launched01 > 0]) * np.ones_like(prices)
            elif pricing_method == 'indication':
                effective_prices = np.array(prices) * launched01
            elif pricing_method == 'first_launched':
                effective_prices = x.Prices
            else:
                raise ValueError('Invalid pricing method specified.')

        return State(Tests=tuple(test_results), Launched=tuple(launched), Period=t_per, Prices=tuple(effective_prices))

    return vf_combined(x)
