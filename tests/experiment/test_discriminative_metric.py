import numpy as np
from scipy.stats import entropy

from geobench.experiment.discriminative_metric import pairwise_entropy, prob_higher, rank_entropy


def test_p_higher():
    prob = prob_higher([0.1, 0.2, 0.3], [0.09, 0.12])
    assert prob == 5 / 6


def test_pw_entropy():
    scores = [[0.1, 0.2, 0.3], [0.09, 0.12]]
    prob = prob_higher(scores[0], scores[1])
    entropy_ = entropy([prob, 1 - prob], base=2)
    assert entropy_ == pairwise_entropy(scores)

    scores = [[0.1, 0.2, 0.3], [0.09, 0.12], [0.12, 0.23, 0.29]]
    discr = pairwise_entropy(scores)

    assert discr > 0
    assert discr < 1

    discr = pairwise_entropy([[1, 2], [3, 4]])
    assert discr == 0

    discr = pairwise_entropy([[1, 4], [2, 3]])
    assert discr == 1


def test_rank_entropy():
    algo1 = [0.5, 0.51]
    algo2 = [0.49, 0.505, 0.52]
    entropy, counter = rank_entropy([algo1, algo2], n_samples=100, return_counter=True)

    assert np.sum(list(counter.values())) == 100
    assert entropy > 0.9
    assert entropy <= 1.0

    algo1 = [0.5, 0.51]
    algo2 = [1.49, 1.505, 1.52]
    entropy, counter = rank_entropy([algo1, algo2], n_samples=100, return_counter=True)

    assert entropy == 0
    assert np.sum(list(counter.values())) == 100

    # entropy, counter = rank_entropy([[], algo2], n_samples=100, return_counter=True)


if __name__ == "__main__":
    # test_p_higher()
    # test_pw_entropy()
    test_rank_entropy()
