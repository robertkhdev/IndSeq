from collections import Counter
from multiprocessing import Pool
import json
import pandas as pd
import time

import randmodels
from indseqtree import *
from utils import *


def run_rand(K=3, n_samples=100, block_size=10_000, diagnostic=False, return_tree=False):
    np.random.seed(1234)
    r = 1 / (1 + 0.1)  # discount factor

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
        test_costs = np.random.random(K) * 0.1  # varying this multipleir results in very different numbers of policies
                                                # for the same number of samples taken.
        # test_costs = np.ones(K) * 0.1
        # ind_values = np.random.random(K)
        ind_values = np.append(np.array([1]), np.random.random(K - 1))
        pricing_mults = np.append(np.array([1]), np.random.random(K - 1))

        x = State(Tests=test_results,
                  Launched=launched,
                  First=launched_first,
                  Period=t_per,
                  PeriodValue=0,
                  EV=0,
                  JointProb=to_tuple(joint_probs),
                  Action=Action(Test=None, Launch=None, Index=None),
                  LaunchPer=launch_period,
                  TestCost=tuple(test_costs),
                  IndicationValues=tuple(ind_values),
                  PricingMults=tuple(pricing_mults),
                  DiscountRate=r)

        tree_tic = time.perf_counter()
        tree = tree_start(x)
        tree_toc = time.perf_counter()
        times.append(tree_toc - tree_tic)
        values.append(tree['value'].EV)
        samples.append(tree)
        if diagnostic:
            print(json.dumps(tree))

        step_count_monitor += 1
        if step_count_monitor % block_size == 0:
            pass
        # if step_count_monitor == 10_000:
        #     toc = time.perf_counter()
        #     print('{0:0.0%}'.format(round(index / n_samples, 2)), round(toc - tic, 0), 'seconds')
        #     step_count_monitor = 0
        #     tic = time.perf_counter()

    print('Avg. Time = ', np.mean(times))
    print('Avg. EV = ', np.mean(values))
    print('St. Dev. EV = ', np.std(values))

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
        results = p.map(tree_start, x_list)

    return results


def timing_test_instance(K: int, n_samples: int):
    print('Running K =', str(K), ' n_samples =', n_samples)
    tic = time.perf_counter()
    # K = 3
    # n_samples = 1_000
    samples = run_rand(K, n_samples, diagnostic=False, return_tree=False)
    compact_policies = [make_compact_policy(s) for s in samples]
    policies = [make_policy(s) for s in samples]
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
    # counts_df.sort_values('samples_opt', ascending=False)\
    #     .reset_index().plot.line(y='samples_opt', title='Policies ranked by # opt. policies (desc.).')
    # counts_df.plot.line(y='Cumulative', title='Cumulative policies by # opt. samples.')
    toc = time.perf_counter()
    print('Unique optimal policies: ', len(counts_df))
    print('Time (sec.): ', toc - tic)


def timing_test():
    ks = [2, 3, 4]
    ns = [1_000, 10_000, 100_000, 1_000_000]
    for k in [2, 3]:
        for n in ns:
            print('#'*100)
            timing_test_instance(K=k, n_samples=n)


def recycle_cache(fn):
    print(fn.__name__, ' ' * (20 - len(fn.__name__)), fn.cache_info())
    fn.cache_clear()


def cache_test():
    for k in [2, 3, 4]:
        tic = time.perf_counter()
        _ = run_rand(K=k, n_samples=100)
        toc = time.perf_counter()
        print('total seconds ', toc - tic)
        recycle_cache(patent_window_mod)
        recycle_cache(g)
        recycle_cache(actions)
        recycle_cache(success_prob)
        # recycle_cache(calc_marginal)
        # recycle_cache(vf_test)
        # recycle_cache(vf_endpoint)
        # recycle_cache(vf)