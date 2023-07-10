import experiments


if __name__ == '__main__':
    diagnostic = False
    # diagnostic = True

    # tests.run_tests()

    # diagnostic = True
    # tree = run_rand(3, diagnostic, return_tree=True)
    # policy = make_policy(tree)
    # print_policy(policy)

    experiments.timing_test()

    # experiments.cache_test()