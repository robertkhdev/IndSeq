import numpy as np
import pandas as pd



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


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a



def make_policy(tree):
    node = tree['value']
    children = tree['children']
    action = node.Action
    launched = node.Launched
    tested = node.Tests
    first_launch = node.First
    index = tree['choice']
    period = node.Period

    state = {'Tested': tested,
             'Launched': launched,
             'FirstLaunch': first_launch,
             'Period': period}

    # decision node
    if index is not None:
        # the node state is duplicated at decisions
        node = children[index]['value']
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
    return {'state': state, 'action': node.Action, 'children': []}


def compactify_state(state):
    state_string = ''.join(str(s) for s in state)
    return state_string.replace('None', '_')


def make_compact_policy(tree):
    children = tree['children']
    index = tree['choice']

    # the node state is duplicated at decisions
    node = children[index]['value']
    children = children[index]['children']
    tested = compactify_state(node.Tests)
    launched = compactify_state(node.Launched)
    first_launch = '_' if node.First is None else str(node.First)
    state = 'P' + str(node.Period) + 'T' + tested + 'L' + launched + 'F' + first_launch

    act_test = node.Action.Test
    act_test_str = '_' if act_test is None else str(act_test)
    act_launch_str = '_' if node.Action.Launch is None else str(node.Action.Launch)

    children = [make_compact_policy(c) for c in children]
    return {'state': state, 'action': 'T' + act_test_str + 'L' + act_launch_str, 'children': children}


def extract_decision_rules(policy):
    rule = ''.join(str(policy['state'])) + ':' + str(policy['action'])
    children = policy['children']
    if len(children) >= 1:  # and policy['action'] != '_;_':
        # go one level down
        # in the policy, decision nodes have 1 branch for the selected alternative
        children = policy['children']
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

