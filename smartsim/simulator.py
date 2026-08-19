"""Simuladores: wrapper genérico + funciones sintéticas de prueba (Branin, Hartmann6)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict


@dataclass
class SimulatorWrapper:
    """Envuelve una función de simulación y cuenta cuántas veces se ejecuta.

    fn: recibe un dict {nombre_param: valor} y devuelve un float.
    noise_std: desvío de un ruido gaussiano opcional sumado al resultado.
    """

    fn: Callable[[Dict[str, float]], float]
    noise_std: float = 0.0
    n_calls: int = field(default=0, init=False)
    total_time: float = field(default=0.0, init=False)
    _rng: "object" = field(default=None, init=False, repr=False)

    def __post_init__(self):
        import numpy as np

        self._rng = np.random.default_rng(0)

    def run(self, params: Dict[str, float]) -> float:
        start = time.perf_counter()
        result = float(self.fn(params))
        if self.noise_std > 0:
            result += float(self._rng.normal(0, self.noise_std))
        self.total_time += time.perf_counter() - start
        self.n_calls += 1
        return result


# ---------------------------------------------------------------------------
# Funciones sintéticas de benchmark
# ---------------------------------------------------------------------------

def branin(params: Dict[str, float]) -> float:
    """Función de Branin, benchmark clásico de optimización 2D.

    Dominio recomendado: x1 in [-5, 10], x2 in [0, 15].
    Tres mínimos globales con valor ~0.397887.
    """
    x1, x2 = params["x1"], params["x2"]
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5 / math.pi, 6.0, 10.0, 1 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def hartmann6(params: Dict[str, float]) -> float:
    """Función de Hartmann 6D, benchmark de optimización de mayor dimensión.

    Dominio: cada xi in [0, 1]. Mínimo global ~ -3.32237.
    """
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    P = [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
    x = [params[f"x{i+1}"] for i in range(6)]
    outer = 0.0
    for i in range(4):
        inner = sum(A[i][j] * (x[j] - P[i][j] * 1e-4) ** 2 for j in range(6))
        outer += alpha[i] * math.exp(-inner)
    return -outer
