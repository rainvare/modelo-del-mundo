import numpy as np

from smartsim.surrogate import SurrogateModel


def test_fit_predict_shapes():
    param_space = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}
    history = [
        ({"x1": 0.0, "x2": 0.0}, 1.0),
        ({"x1": 5.0, "x2": 5.0}, 2.0),
        ({"x1": -5.0, "x2": 15.0}, 3.0),
        ({"x1": 10.0, "x2": 0.0}, 0.5),
    ]
    model = SurrogateModel(param_space)
    model.fit(history)

    test_points = [{"x1": 1.0, "x2": 1.0}, {"x1": 2.0, "x2": 3.0}]
    mean, std = model.predict(test_points)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std >= 0)


def test_predict_before_fit_raises():
    param_space = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}
    model = SurrogateModel(param_space)
    try:
        model.predict([{"x1": 0.0, "x2": 0.0}])
        assert False, "debería haber lanzado RuntimeError"
    except RuntimeError:
        pass


def test_uncertainty_shrinks_near_observed_points():
    """La incertidumbre debe ser menor cerca de puntos ya observados que lejos de ellos."""
    param_space = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}
    history = [({"x1": 0.0, "x2": 0.0}, 1.0), ({"x1": 1.0, "x2": 1.0}, 1.1)]
    model = SurrogateModel(param_space)
    model.fit(history)

    near = {"x1": 0.5, "x2": 0.5}
    far = {"x1": 9.0, "x2": 14.0}
    mean, std = model.predict([near, far])

    assert std[0] < std[1]
