import numpy as np

from smartsim import acquisition as acq


def test_ucb_favors_high_uncertainty_when_mean_equal():
    mean = np.array([1.0, 1.0])
    std = np.array([0.1, 1.0])
    scores = acq.upper_confidence_bound(mean, std, y_best=1.0, kappa=2.0, minimize=True)
    assert scores[1] > scores[0]


def test_ei_zero_when_no_uncertainty():
    mean = np.array([5.0])
    std = np.array([0.0])
    ei = acq.expected_improvement(mean, std, y_best=1.0, minimize=True)
    assert ei[0] == 0.0


def test_pi_between_zero_and_one():
    mean = np.array([0.5, 5.0])
    std = np.array([1.0, 1.0])
    pi = acq.probability_of_improvement(mean, std, y_best=1.0, minimize=True)
    assert np.all(pi >= 0.0) and np.all(pi <= 1.0)


def test_max_variance_ignores_mean():
    mean = np.array([100.0, -100.0])
    std = np.array([0.2, 0.9])
    scores = acq.max_variance(mean, std)
    assert np.array_equal(scores, std)


def test_registry_lookup():
    for name in ["ucb", "ei", "pi", "variance"]:
        assert callable(acq.get(name))
    try:
        acq.get("no_existe")
        assert False
    except ValueError:
        pass
