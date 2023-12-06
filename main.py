import experiments
#import tests
import unittest


if __name__ == '__main__':
    diagnostic = False
    # diagnostic = True

    # tests.run_tests()

    # experiments.timing_test_instance(K=2, n_samples=10)

    diagnostic = True
    tree = experiments.run_rand(2, diagnostic, return_tree=True)
    policy = experiments.make_policy(tree)
    experiments.print_policy(policy)

    # experiments.timing_test()

    # experiments.cache_test()

    # experiments.timing_test_instance(K=2, n_samples=10_000)