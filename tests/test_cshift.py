import numpy as np
from cshift import CShift


def test_validation():
    np.random.seed(0)
    clusters = np.random.choice(5, 100)
    groups = np.random.choice(10, 100)
    reference = np.array([0, 1])
    cshift = CShift(clusters, groups, reference)


def test_broken_validation():
    np.random.seed(0)
    clusters = np.random.choice(5, 100)
    groups = np.random.choice(10, 101)
    reference = np.array([0, 1])
    try:
        cshift = CShift(clusters, groups, reference)
        raise ValueError("Validation failed")
    except AssertionError:
        assert True


def test_missing_reference():
    np.random.seed(0)
    clusters = np.random.choice(5, 100)
    groups = np.random.choice(10, 100)
    reference = np.array([11, 12])
    try:
        cshift = CShift(clusters, groups, reference)
        raise ValueError("Validation failed")
    except AssertionError:
        assert True


def test_group_counts():
    clusters = np.array([0, 1, 2, 0, 1, 2])
    groups = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([0, 1])
    cshift = CShift(clusters, groups, reference)
    assert np.sum(cshift.g_unique - np.array([0, 1])) == 0
    assert np.sum(cshift.g_counts - np.array([3, 3])) == 0
    assert cshift.g_size == 2


def test_cluster_counts():
    clusters = np.array([0, 1, 2, 0, 1, 2])
    groups = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([0, 1])
    cshift = CShift(clusters, groups, reference)
    assert np.sum(cshift.c_unique - np.array([0, 1, 2])) == 0
    assert np.sum(cshift.c_counts - np.array([2, 2, 2])) == 0
    assert cshift.c_size == 3


def test_reference_idx():
    clusters = np.array([0, 1, 2, 0, 1, 2])
    groups = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([0, 1])
    cshift = CShift(clusters, groups, reference)
    assert np.sum(cshift.ref_idx - np.array([0, 1])) == 0


def test_distributions():
    clusters = np.array([0, 1, 2, 0, 1, 2])
    groups = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([0, 1])
    cshift = CShift(clusters, groups, reference)
    assert np.sum(cshift.distributions - np.array([[1, 1, 1], [1, 1, 1]])) == 0
