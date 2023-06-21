import numpy as np
import pandas as pd
import json
import time
from collections import Counter, namedtuple
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessPool as Pool
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional
import operator
from numba import jit, vectorize

import randmodels
from indseqtree import *


def format_state(x, str_end=''):
    test_results = np.array(x[0])
    launched = np.array(x[1])
    launched_first = x[2]
    t_per = x[3]
    ret_str = ('T=' + ''.join([str(i) if i is not None else '_' for i in test_results]) +
               ' | L=' + ''.join('1' if i == 1 else '0' for i in launched) +
               ' | F=' + str(launched_first if launched_first is not None else '_') +
               ' | Per=' + str(t_per) + ' |' +
               str_end)

    return ret_str


def print_diagnostic(diagnostic_on, x, t_per, value_in_period, ev, prob, prob_str):
    if diagnostic_on:
        str_end = 'CF=' + str(round(value_in_period, 3)) + '| ' + prob_str + '=' + str(
            round(prob, 3)) + ' | ' + 'EV=' + str(
            round(ev, 3))
        print('\t' * (t_per * 2 + 1), format_state(x, str_end))


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

            tic = time.perf_counter()
            tree = vf(pickle.dumps(x))
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
    # for t in tree.items():
    #     print(t)
    #
    # print('Joint Probabilities of Test Success')
    # print(joint_probs)
    # print('Value: ', val, '   Seconds:', str(toc - tic))


def vf2(x) -> Dict:
    t_per = x[3]
    action_set = actions(x)
    action_values = []
    action_list = []
    node_dict = dict()

    if len(action_set) > 0:
        # there are one or more actions
        for a in action_set:
            if a[0] is not None:
                # do a test
                k = a[0]
                t_per = x[3]
                r = x[8]
                n_tests = len(x[0])
                success = [0] * n_tests
                success[k] = 1
                failure = [0] * n_tests
                failure[k] = 0

                value_in_period = g(x, a)
                x_s = f(x, a, success)
                x_f = f(x, a, failure)
                ps = success_prob(x_s, x_f)
                success_dict = vf(x_s)
                success_value = success_dict['value'].EV
                failure_dict = vf(x_f)
                failure_value = failure_dict['value'].EV

                value = value_in_period + r * (ps * success_value + (1 - ps) * failure_value)
                current_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                      PeriodValue=value_in_period, EV=value, JointProb=1,
                                      Action={'Test': a[0], 'Launch': a[1], 'Index': None})

                node_dict = {'value': current_state, 'children': [success_dict, failure_dict]}
            elif a[1] is not None:
                # case of no test but launch something -> at end node in tree
                t_per = x[3]
                value_in_period = g(x, a)
                value = value_in_period
                launch_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                     PeriodValue=g(x, a), EV=value, JointProb=1,
                                     Action={'Test': a[0], 'Launch': a[1], 'Index': None})
                node_dict = {'value': launch_state, 'children': []}
            else:
                # no test or launch, so this is an endpoint where we stop
                t_per = x[3]
                value_in_period = g(x, a)
                value = value_in_period
                stop_state = State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                   PeriodValue=g(x, a), EV=value, JointProb=0,
                                   Action={'Test': None, 'Launch': None, 'Index': None})
                node_dict = {'value': stop_state, 'children': []}
            action_values.append(node_dict['value'].EV)
            action_list.append(node_dict)

        ret_val = max(action_values)
        choice_idx = action_values.index(ret_val)
        value_in_period = action_list[choice_idx]['value'].EV
        action = {'Test': None, 'Launch': None, 'Index': choice_idx}
        node_dict = {'value': State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                    PeriodValue=value_in_period, EV=ret_val, JointProb=0, Action=action),
                     'children': action_list}

    if len(action_values) == 0:
        # uncertainty node
        action = {'Test': None, 'Launch': None, 'Index': None}
        node_dict = {'value': State(Tests=tuple(x[0]), Launched=tuple(x[1]), First=x[2], Period=t_per,
                                    PeriodValue=0, EV=0, JointProb=0, Action=action),
                     'children': []}

    return node_dict


def run_rand(K=3, n_samples=100, diagnostic=False, return_tree=False):
    np.random.seed(1234)
    r = 1 / (1 + 0.1)  # discount factor

    failures = 0
    times = []
    values = []
    samples = []
    step_count_monitor = 0
    tic = time.perf_counter()
    for index in range(n_samples):
        test_results = tuple([None] * K)
        launched = tuple([None] * K)
        launched_first = None
        launch_period = None
        t_per = 0

        joint_probs = np.random.dirichlet(np.ones(2 ** K)).reshape(*([2] * K))
        test_costs = np.random.random(K) * 0.2  # varying this multipleir results in very different numbers of policies
                                                # for the same number of samples taken.
        # test_costs = np.ones(K) * 0.1
        # ind_values = np.random.random(K)
        ind_values = np.append(np.array([1]), np.random.random(K - 1))
        pricing_mults = np.append(np.array([1]), np.random.random(K - 1))

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

        tree_tic = time.perf_counter()
        tree = vf(pickle.dumps(x))
        tree_toc = time.perf_counter()
        times.append(tree_toc - tree_tic)
        values.append(tree['value'].EV)
        samples.append(tree)
        if diagnostic:
            print(json.dumps(tree))
        # print(json.dumps(tree, indent=4))

        # step_count_monitor += 1
        # if step_count_monitor == 1_000:
        #     toc = time.perf_counter()
        #     print('{0:0.0%}'.format(round(index / n_samples, 2)), round(toc - tic, 0), 'seconds')
        #     step_count_monitor = 0
        #     tic = time.perf_counter()

    print('Avg. Time = ', np.mean(times))
    print('Avg. EV = ', np.mean(values))
    print('St. Dev. EV = ', np.std(values))

    # with open('tree.json', 'w') as f:
    #     f.write(json.dumps(tree, indent=4))

    if return_tree:
        return tree
    else:
        return samples


def run_one(x):
    ...


def run_rand_parallel(K=3, n=1000):
    r = 1 / (1 + 0.1)  # discount factor
    test_results = [None] * K
    launched = [None] * K
    launched_first = None
    t_per = 0
    input_samples = randmodels.generate_samples(K, n)
    x_list = []

    for input_sample in input_samples:
        joint_probs = input_sample['joint_probs']
        test_costs = input_sample['test_costs']
        ind_values = input_sample['ind_values']
        pricing_mults = input_sample['pricing_mults']
        x = (test_results,
             launched,
             launched_first,
             t_per,
             joint_probs,
             test_costs,
             ind_values,
             pricing_mults,
             r)
        x_list.append(x)

    with Pool(nodes=20) as p:
        results = p.map(vf, x_list)

    return results


def make_policy(tree):
    node = tree['value']
    children = tree['children']
    action = node.Action
    test = action['Test']
    launch = action['Launch']
    launched = node.Launched
    tested = node.Tests
    first_launch = node.First
    index = action['Index']
    period = node.Period

    state = {'Tested': tested,
             'Launched': launched,
             'FirstLaunch': first_launch,
             'Period': period}

    # decision node
    if index is not None:
        # the node state is duplicated at decisions
        node = children[node.Action['Index']]['value']
        action = node.Action
        period = node.Period
        # index = action['Index']
        state = {'Tested': tested,
                 'Launched': launched,
                 'FirstLaunch': first_launch,
                 'Period': period}
        if index is not None:
            children = children[index]['children']
            children = [make_policy(c) for c in children]
            return {'state': state, 'action': node.Action, 'children': children}
        else:
            children = []
        return {'state': state, 'action': action, 'children': children}

    # end node
    return {'state': state, 'action': None, 'children': []}


def compactify_state(state):
    state_string = ''.join(str(s) for s in state)
    return state_string.replace('None', '_')


def make_compact_policy(tree):
    node = tree['value']
    children = tree['children']
    action = node.Action
    launched = node.Launched
    tested = node.Tests
    first_launch = node.First
    t_per = node.Period
    index = action['Index']

    tested = compactify_state(tested)
    launched = compactify_state(launched)
    first_launch = '_' if first_launch is None else str(first_launch)
    state = str(t_per) + ';' + tested + ';' + launched + ';' + first_launch
    if index is not None:  # decision node
        act_test = children[index]['value'].Action['Test']
        act_test_str = '_' if act_test is None else str(act_test)
        act_launch_str = '_' if action['Launch'] is None else str(action['Launch'])
        choice = make_compact_policy(children[index])
        return {'state': state, 'action': act_test_str + ';' + act_launch_str, 'children': [choice]}
    elif len(children) > 1:  # probability node
        children = [make_compact_policy(c) for c in children]
        return {'state': state, 'action': '_;_', 'children': children}
    else:  # end node
        launched = node.Launched
        tested = node.Tests
        first_launch = node.First
        tested = compactify_state(tested)
        launched = compactify_state(launched)
        first_launch = '_' if first_launch is None else str(first_launch)
        state = str(t_per) + ';' + tested + ';' + launched + ';' + first_launch
        return {'state': state, 'action': '_;_', 'children': []}


def extract_decision_rules(policy):
    rule = ''.join(policy['state']) + ':' + policy['action']
    children = policy['children']
    if len(children) == 1 and policy['action'] != '_;_':
        # go one level down
        # in the policy, decision nodes have 1 branch for the selected alternative
        children = policy['children'][0]['children']
        child_decisions = [extract_decision_rules(c) for c in children]
        if any(type(cd) == list for cd in child_decisions):
            flat_list = [cd for sublist in child_decisions for cd in sublist if type(sublist) == list]
        else:
            flat_list = child_decisions
        as_list = [rule] + flat_list
        return as_list
    return [rule]


def make_policy_all_data(tree):
    """
    Make policy tree and include all node data (e.g., EV, probabilities, etc.)
    :param tree:
    :return:
    """
    node = tree['value']
    children = tree['children']
    action = node.Action
    # test = action['Test']
    # launch = action['Launch']
    index = action['Index']
    # action_str = ''
    # if test is not None:
    #     action_str += 'Test ' + str(test) + ' '
    # if launch is not None:
    #     action_str += 'Launch ' + str(launch)
    # decision node
    if index is not None:
        # the node state is duplicated at decisions
        node = children[node.Action['Index']]['value']
        action = node.Action
        # index = action['Index']
        if index is not None:
            children = children[index]['children']
            children = [make_policy(c['value']) for c in children]
            return {'state': node, 'action': node.Action, 'children': children}
        else:
            children = []
        return {'state': node, 'action': action, 'children': children}

    # end node
    return {'state': node, 'action': None, 'children': []}


def print_policy(policy):
    """
    output policy tree to console
    """
    node = policy
    action = node['action']
    if action is not None and len(action) > 0:
        test = action['Test']
        launch = action['Launch']
        action_str = ''
        if test is not None:
            action_str += 'Test ' + str(test) + ' '
        if launch is not None:
            action_str += 'Launch ' + str(launch)
        print('\t'*policy['state']['Period'],
              'T:' + ''.join([str(t if t is not None else '_') for t in node['state']['Tested']]),
              'L:' + ''.join([str(l if l is not None else '_') for l in node['state']['Launched']]),
              'P:' + str(node['state']['Period']), end=' ')
        if len(action_str) > 0:
            print('\t\t\tAct:', action_str)
        else:
            print('\n', end=' ')
    if len(policy['children']) == 0:
        return
    for c in policy['children']:
        # print('\t'*policy['state'].Period, end='')
        print_policy(c)


if __name__ == '__main__':
    diagnostic = False
    # diagnostic = True

    # run_tests()

    # diagnostic = True
    # tree = run_rand(3, diagnostic, return_tree=True)
    # policy = make_policy(tree)
    # print_policy(policy)

    # for i in range(2, 6):
    #     print(i)
    #     run_rand(i, False)



    tic = time.perf_counter()
    K = 3
    n_samples = 1_000
    samples = run_rand(K, n_samples, diagnostic=False, return_tree=False)
    policies = [make_compact_policy(s) for s in samples]
    # policy_strs = [str(p) for p in policies]
    rule_lists = [extract_decision_rules(p) for p in policies]
    [r.sort(reverse=True, key=lambda x: x.count('_')) for r in rule_lists]
    rule_sets = [frozenset(r) for r in rule_lists]

    counts = Counter(rule_sets)
    counts_df = pd.DataFrame(list(counts.items()), columns=['policy', 'samples_opt'])
    counts_df = counts_df.sort_values('samples_opt', ascending=False).reset_index()
    counts_df['Cumulative'] = counts_df['samples_opt'].cumsum()
    counts_df.to_excel('indseq_k' + str(K) + '_s' + str(n_samples) + '.xlsx')

    counts_df.plot.hist(y='samples_opt', bins=50, title='Num. policies optimal for given no. of samples.')
    counts_df.sort_values('samples_opt', ascending=False)\
        .reset_index().plot.line(y='samples_opt', title='Policies ranked by # opt. policies (desc.).')
    counts_df.plot.line(y='Cumulative', title='Cumulative policies by # opt. samples.')
    toc = time.perf_counter()
    print(toc - tic)



    # def run_trial(K, n_samples):
    #     samples = run_rand(K, n_samples, diagnostic=False, return_tree=False)
    #     policies = [make_compact_policy(s) for s in samples]
    #     # policy_strs = [str(p) for p in policies]
    #     rule_lists = [extract_decision_rules(p) for p in policies]
    #     [r.sort(reverse=True, key=lambda x: x.count('_')) for r in rule_lists]
    #     rule_sets = [frozenset(r) for r in rule_lists]
    #
    #     counts = Counter(rule_sets)
    #     counts_df = pd.DataFrame(list(counts.items()), columns=['policy', 'samples_opt'])
    #     counts_df = counts_df.sort_values('samples_opt', ascending=False).reset_index()
    #     counts_df['Cumulative'] = counts_df['samples_opt'].cumsum()
    #     counts_df.to_excel('indseq_k' + str(K) + '_s' + str(n_samples) + '.xlsx')
    #
    #
    # for K in [2, 3, 4]:
    #     for n in [1_000, 10_000, 100_000]:
    #         print('K =', str(K), ' | n =', str(n))
    #         tic = time.perf_counter()
    #         run_trial(K, n)
    #         toc = time.perf_counter()
    #         print(toc - tic, 'seconds\n')



    # extract_decision_rules(make_compact_policy(samples[3]))

    # K = 3
    # n = 1_000
    # tic = time.perf_counter()
    # r = 1 / (1 + 0.1)  # discount factor
    # test_results = [None] * K
    # launched = [None] * K
    # launched_first = None
    # t_per = 0
    # input_samples = randmodels.generate_samples(K, n)
    # x_list = []
    #
    # for input_sample in input_samples:
    #     joint_probs = input_sample['joint_probs']
    #     test_costs = input_sample['test_costs']
    #     ind_values = input_sample['ind_values']
    #     pricing_mults = input_sample['pricing_mults']
    #     x = (test_results,
    #          launched,
    #          launched_first,
    #          t_per,
    #          joint_probs,
    #          test_costs,
    #          ind_values,
    #          pricing_mults,
    #          r)
    #     x_list.append(x)
    #
    # with Pool(nodes=15) as p:
    #     samples = p.imap(vf, x_list)
    # toc = time.perf_counter()
    # print(toc - tic)
    #
    # policies = [make_compact_policy(s) for s in samples]
    # # policy_strs = [str(p) for p in policies]
    # rule_lists = [extract_decision_rules(p) for p in policies]
    # [r.sort(reverse=True, key=lambda x: x.count('_')) for r in rule_lists]
    # rule_sets = [frozenset(r) for r in rule_lists]
    #
    # counts = Counter(rule_sets)
    # counts_df = pd.DataFrame(list(counts.items()), columns=['policy', 'samples_opt'])
    # counts_df = counts_df.sort_values('samples_opt', ascending=False).reset_index()
    # counts_df['Cumulative'] = counts_df['samples_opt'].cumsum()
    # counts_df.to_excel('indseq_k' + str(K) + '_s' + str(n) + '.xlsx')
    #
    # counts_df.plot.hist(y='samples_opt', bins=50, title='Num. policies optimal for given no. of samples.')
    # counts_df.sort_values('samples_opt', ascending=False)\
    #     .reset_index().plot.line(y='samples_opt', title='Policies ranked by # opt. policies (desc.).')
    # counts_df.plot.line(y='Cumulative', title='Cumulative policies by # opt. samples.')

    # K = 2
    # r = 1 / (1 + 0.1)
    # test_results = [None] * K
    # launched = [None] * K
    # launched_first = None
    # t_per = 0
    #
    # joint_probs = np.array([[0.07, 0.13],[0.03, 0.77]])
    # test_costs = np.ones(K) * 0.1
    # ind_values = np.array([0.95, 0.1])
    # pricing_mults = np.array([1.0, 0.5])
    #
    # x = (test_results,
    #      launched,
    #      launched_first,
    #      t_per,
    #      joint_probs,
    #      test_costs,
    #      ind_values,
    #      pricing_mults,
    #      r)
    #
    # tree = vf(x)
    # policy = make_compact_policy(tree)
    # rule_list = extract_decision_rules(policy)
    #
