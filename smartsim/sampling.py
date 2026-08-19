"""Muestreo inicial del espacio de parámetros (Latin Hypercube / aleatorio)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc


def latin_hypercube(param_space: Dict[str, Tuple[float, float]], n: int, seed: int = 0) -> List[Dict[str, float]]:
    """Genera n puntos con Latin Hypercube Sampling, bien distribuidos en el espacio."""
    names = list(param_space.keys())
    d = len(names)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n)
    lo = np.array([param_space[k][0] for k in names])
    hi = np.array([param_space[k][1] for k in names])
    scaled = qmc.scale(unit_samples, lo, hi)
    return [dict(zip(names, row)) for row in scaled]


def random_uniform(param_space: Dict[str, Tuple[float, float]], n: int, seed: int = 0) -> List[Dict[str, float]]:
    """Muestreo aleatorio uniforme, usado para generar candidatos al optimizar la adquisición."""
    rng = np.random.default_rng(seed)
    names = list(param_space.keys())
    points = []
    for _ in range(n):
        points.append({k: rng.uniform(param_space[k][0], param_space[k][1]) for k in names})
    return points
