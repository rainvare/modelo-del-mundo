"""Funciones de adquisición.

Cada función devuelve un score por candidato donde mayor es mejor; el
orquestador maximiza ese score para elegir el próximo punto. El flag
`minimize` indica si el objetivo del simulador es minimizar o maximizar;
`max_variance` lo ignora porque solo depende de la incertidumbre del modelo.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, y_best: float = None,
                            kappa: float = 2.0, minimize: bool = False) -> np.ndarray:
    """UCB/LCB: balancea explotación (media) y exploración (incertidumbre)."""
    return (-mean + kappa * std) if minimize else (mean + kappa * std)


def expected_improvement(mean: np.ndarray, std: np.ndarray, y_best: float,
                          xi: float = 0.01, minimize: bool = False) -> np.ndarray:
    """EI: mejora esperada respecto al mejor valor visto hasta ahora."""
    std_safe = np.maximum(std, 1e-9)
    imp = (y_best - mean - xi) if minimize else (mean - y_best - xi)
    z = imp / std_safe
    ei = imp * norm.cdf(z) + std_safe * norm.pdf(z)
    return np.where(std > 1e-9, ei, 0.0)


def probability_of_improvement(mean: np.ndarray, std: np.ndarray, y_best: float,
                                xi: float = 0.01, minimize: bool = False) -> np.ndarray:
    """PI: probabilidad de superar el mejor valor visto (más conservadora que EI)."""
    std_safe = np.maximum(std, 1e-9)
    z = ((y_best - mean - xi) if minimize else (mean - y_best - xi)) / std_safe
    return norm.cdf(z)


def max_variance(mean: np.ndarray, std: np.ndarray, y_best: float = None,
                  minimize: bool = False) -> np.ndarray:
    """Exploración pura: devuelve la desviación estándar de cada candidato."""
    return std


REGISTRY = {
    "ucb": upper_confidence_bound,
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "variance": max_variance,
}


def get(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Función de adquisición desconocida: '{name}'. Opciones: {list(REGISTRY)}")
