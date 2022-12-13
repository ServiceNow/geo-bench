"""Tools to estimate how discriminative is a dataset."""

from collections import Counter
from itertools import combinations
from typing import Any, List, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import pair_confusion_matrix


def prob_higher(scores_a: ArrayLike, scores_b: ArrayLike):
    """Compute the probability that a is higher than b from two list of samples."""
    scores_a = np.asarray(scores_a).reshape(-1, 1)
    scores_b = np.asarray(scores_b).reshape(1, -1)
    return np.mean(scores_a > scores_b)


def pairwise_entropy(all_scores: List[List[float]]):
    """Compute average entropy of prob_higher for each pair of score list in `all_scores`.

    Note: the entropy of prob_lower_or_equal is the same as the entropy of prob_higher.

    Args:
        all_scores: 2d array of scores or iterable of iterable.

    Returns:
        average entropy of pairwise comparisons.
    """
    entropy_list = []
    for scores_a, scores_b in combinations(all_scores, 2):
        prob = prob_higher(scores_a, scores_b)
        entropy_list.append(entropy([prob, 1 - prob], base=2))
    return np.mean(entropy_list)


def boostrap_pw_entropy(all_scores, repeat=10, std_ratio=0.1):
    """Bootstrap version."""
    values = []
    for i in range(repeat):
        bootstraped_scores = []
        for scores in all_scores:
            std = np.std(scores)
            scores_ = np.random.choice(scores, size=len(scores), replace=True)
            scores_ += std * std_ratio * np.random.randn(len(scores))
            bootstraped_scores.append(scores_)

        # values.append(rank_entropy(bootstraped_scores))
        values.append(pairwise_entropy(bootstraped_scores))

    return values


def get_rank(scores, axis=0):
    """Return the rank of `scores` across `axis`."""
    order = np.argsort(scores, axis=axis)
    return np.argsort(order, axis=axis)


def rank_entropy(all_scores, n_samples=100, return_counter=False):
    """Estimate the entropy of the rank."""
    samples = []
    for seeds in all_scores:
        if len(seeds) == 0:
            samples.append([np.nan] * n_samples)
        else:
            samples.append(np.random.choice(seeds, size=n_samples, replace=True))

    counter = Counter()
    for rank in get_rank(samples).T:
        counter.update([tuple(rank)])

    p_list = []
    for rank, count in counter.items():
        p_list.append(count / n_samples)

    if return_counter:
        return entropy(p_list, base=2), counter
    else:
        return entropy(p_list, base=2)
