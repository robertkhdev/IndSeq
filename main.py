import randmodels
from indseqtree import *
import experiments
import tests


if __name__ == '__main__':
    diagnostic = False
    # diagnostic = True

    # tests.run_tests()

    # diagnostic = True
    # tree = run_rand(3, diagnostic, return_tree=True)
    # policy = make_policy(tree)
    # print_policy(policy)

    # for i in range(2, 6):
    #     print(i)
    #     run_rand(i, False)

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
