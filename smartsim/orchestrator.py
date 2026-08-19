"""Orquestador: bucle de aprendizaje activo.

Alterna entre muestreo inicial, ajuste del modelo sustituto, selección del
próximo punto por la función de adquisición, y ejecución del simulador real
en ese punto. Cada punto del historial fue evaluado por `simulator.run(...)`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import acquisition as acq_module
from .sampling import latin_hypercube, random_uniform
from .simulator import SimulatorWrapper
from .storage import HistoryStore
from .surrogate import SurrogateModel

logger = logging.getLogger("smartsim")


@dataclass
class IterationLog:
    iteration: int
    phase: str
    params: Dict[str, float]
    result: float
    acquisition_score: Optional[float]
    max_candidate_std: Optional[float]
    decision_time_s: Optional[float]


@dataclass
class SmartSimOrchestrator:
    simulator: SimulatorWrapper
    param_space: Dict[str, Tuple[float, float]]
    n_initial: Optional[int] = None
    max_iterations: int = 50
    acquisition: str = "ucb"
    acquisition_kwargs: dict = field(default_factory=dict)
    minimize: bool = True
    n_candidates: int = 2000
    seed: int = 0
    uncertainty_stop_threshold: Optional[float] = None
    verbose: bool = True

    def __post_init__(self):
        self.param_names = list(self.param_space.keys())
        if self.n_initial is None:
            self.n_initial = max(4, 4 * len(self.param_names))
        self.store = HistoryStore(self.param_names)
        self.surrogate = SurrogateModel(self.param_space)
        self.acquisition_fn: Callable = acq_module.get(self.acquisition)
        self.iteration_logs: List[IterationLog] = []
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logger.setLevel(logging.INFO)

    # -- fases del flujo -----------------------------------------------
    def _sample_initial_points(self) -> List[Dict[str, float]]:
        return latin_hypercube(self.param_space, self.n_initial, seed=self.seed)

    def _current_best(self) -> float:
        results = [r for _, r in self.store.as_history_tuples()]
        return min(results) if self.minimize else max(results)

    def _optimize_acquisition(self) -> Tuple[Dict[str, float], float, float]:
        """Devuelve (mejor_punto, su_score, incertidumbre_max_entre_candidatos)."""
        candidates = random_uniform(self.param_space, self.n_candidates, seed=self.seed + len(self.store))
        mean, std = self.surrogate.predict(candidates)
        y_best = self._current_best()
        scores = self.acquisition_fn(mean, std, y_best, minimize=self.minimize, **self.acquisition_kwargs)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], float(scores[best_idx]), float(np.max(std))

    def _log_iteration(self, log: IterationLog) -> None:
        self.iteration_logs.append(log)
        if self.verbose:
            score_txt = f", score={log.acquisition_score:.4f}" if log.acquisition_score is not None else ""
            std_txt = f", max_std_candidatos={log.max_candidate_std:.4f}" if log.max_candidate_std is not None else ""
            logger.info(
                f"[{log.phase:7s}] it={log.iteration:3d} params={ {k: round(v,3) for k,v in log.params.items()} } "
                f"resultado={log.result:.4f}{score_txt}{std_txt}"
            )

    # -- bucle principal --------------------------------------------------
    def run(self) -> Tuple[HistoryStore, SurrogateModel]:
        # Fase 1: muestreo inicial (siempre simulador real)
        for i, point in enumerate(self._sample_initial_points()):
            result = self.simulator.run(point)
            self.store.add(i, point, result, phase="initial")
            self._log_iteration(IterationLog(i, "initial", point, result, None, None, None))

        # Fase 2: bucle de aprendizaje activo
        for it in range(self.max_iterations):
            self.surrogate.fit(self.store.as_history_tuples())

            t0 = time.perf_counter()
            next_point, score, max_std = self._optimize_acquisition()
            decision_time = time.perf_counter() - t0

            if self.uncertainty_stop_threshold is not None and max_std < self.uncertainty_stop_threshold:
                if self.verbose:
                    logger.info(
                        f"Parada por incertidumbre: max_std={max_std:.4f} < "
                        f"umbral={self.uncertainty_stop_threshold:.4f} (it={it})"
                    )
                break

            result = self.simulator.run(next_point)
            iteration_idx = self.n_initial + it
            self.store.add(iteration_idx, next_point, result, phase="active", acquisition_score=score)
            self._log_iteration(
                IterationLog(iteration_idx, "active", next_point, result, score, max_std, decision_time)
            )

        return self.store, self.surrogate

    # -- métricas -----------------------------------------------------
    def summary(self) -> dict:
        n_real = self.simulator.n_calls
        best = self._current_best()
        return {
            "simulaciones_reales": n_real,
            "iteraciones_activas": max(0, n_real - self.n_initial),
            "mejor_resultado": best,
            "tiempo_total_simulador_s": self.simulator.total_time,
            "log_marginal_likelihood_gp": self.surrogate.log_marginal_likelihood(),
        }
