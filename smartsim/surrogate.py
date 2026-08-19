"""Modelo sustituto: Proceso Gaussiano con media + incertidumbre.

Envuelve GaussianProcessRegressor de scikit-learn. Normaliza X a [0,1]^d
usando param_space y normaliza y a media 0 / varianza 1 antes de ajustar.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class SurrogateModel:
    def __init__(self, param_space: Dict[str, Tuple[float, float]], alpha: float = 1e-6):
        self.param_space = param_space
        self.param_names: List[str] = list(param_space.keys())
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
            length_scale=np.ones(len(param_space)),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=0,
        )
        self._y_mean = 0.0
        self._y_std = 1.0
        self._fitted = False

    def _to_matrix(self, points: List[Dict[str, float]]) -> np.ndarray:
        return np.array([[p[name] for name in self.param_names] for p in points])

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        lo = np.array([self.param_space[n][0] for n in self.param_names])
        hi = np.array([self.param_space[n][1] for n in self.param_names])
        return (X - lo) / (hi - lo)

    def fit(self, history: List[Tuple[Dict[str, float], float]]) -> None:
        """Entrena el GP con todo el historial acumulado (params, resultado)."""
        points = [h[0] for h in history]
        y = np.array([h[1] for h in history], dtype=float)
        X = self._normalize_X(self._to_matrix(points))
        self._gp.fit(X, y)
        self._fitted = True

    def predict(self, points: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (media, desvío estándar) para cada punto."""
        if not self._fitted:
            raise RuntimeError("El modelo sustituto no fue entrenado todavía (llamar a fit primero).")
        X = self._normalize_X(self._to_matrix(points))
        mean, std = self._gp.predict(X, return_std=True)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        return float(self._gp.log_marginal_likelihood_value_)
