import experiments
#import tests
import unittest


if __name__ == '__main__':
    diagnostic = False
    # diagnostic = True

    # tests.run_tests()

    experiments.timing_test_instance(K=2, n_samples=100_000)

    # diagnostic = True
    # tree = run_rand(3, diagnostic, return_tree=True)
    # policy = make_policy(tree)
    # print_policy(policy)

    # experiments.timing_test()

    # experiments.cache_test()

    # experiments.timing_test_instance(K=2, n_samples=10_000)