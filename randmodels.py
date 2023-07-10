import numpy as np
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional

def generate_sample(K: int) -> Dict:
    joint_probs = np.random.dirichlet(np.ones(2 ** K)).reshape(*([2] * K))
    test_costs = np.random.random(K) * 0.2  # varying this multipleir results in very different numbers of policies
    # for the same number of samples taken.
    # test_costs = np.ones(K) * 0.1
    # ind_values = np.random.random(K)
    ind_values = np.append(np.array([1]), np.random.random(K - 1))
    pricing_mults = np.append(np.array([1]), np.random.random(K - 1))
    sample = {'joint_probs': joint_probs,
              'test_costs': test_costs,
              'ind_values': ind_values,
              'pricing_mults': pricing_mults}
    return sample


def generate_samples(K, n):
    return [generate_sample(K) for i in range(n)]

# np.random.dirichlet(np.ones(2 ** (K*n))).reshape(*([n] + [2] * K))
