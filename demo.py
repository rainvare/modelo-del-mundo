"""Demo: SmartSim Active Learner sobre la función de Branin.

Compara el número de simulaciones "costosas" necesarias para alcanzar una
precisión objetivo usando aprendizaje activo vs. muestreo aleatorio (proxy
de fuerza bruta), y valida la calibración de incertidumbre del GP.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smartsim import SmartSimOrchestrator, SimulatorWrapper, branin
from smartsim.sampling import random_uniform

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PARAM_SPACE = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}
TRUE_MIN = 0.397887
ERROR_THRESHOLD = 0.05
BUDGET = 60  # máximo de evaluaciones "costosas" permitidas
N_INITIAL = 8
N_SEEDS_BASELINE = 8


def best_so_far(results):
    out, best = [], float("inf")
    for r in results:
        best = min(best, r)
        out.append(best)
    return out


def run_active_learning(seed: int = 0):
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim,
        param_space=PARAM_SPACE,
        n_initial=N_INITIAL,
        max_iterations=BUDGET - N_INITIAL,
        acquisition="ei",
        acquisition_kwargs={"xi": 0.01},
        minimize=True,
        n_candidates=2000,
        seed=seed,
        verbose=False,
    )
    store, _ = orch.run()
    results = [r for _, r in store.as_history_tuples()]
    return results, orch


def run_random_baseline(seed: int = 0):
    sim = SimulatorWrapper(fn=branin)
    points = random_uniform(PARAM_SPACE, BUDGET, seed=1000 + seed)
    return [sim.run(p) for p in points]


def evaluations_to_reach(errors, threshold):
    for i, e in enumerate(errors):
        if e <= threshold:
            return i + 1
    return None


def main():
    print("=== SmartSim Active Learner: demo sobre función de Branin ===\n")
    print(f"Espacio de parámetros: {PARAM_SPACE}")
    print(f"Mínimo global conocido: {TRUE_MIN}")
    print(f"Umbral de error objetivo: {ERROR_THRESHOLD}\n")

    t0 = time.perf_counter()
    al_results, orch = run_active_learning(seed=0)
    al_time = time.perf_counter() - t0
    al_errors = [b - TRUE_MIN for b in best_so_far(al_results)]
    al_n = evaluations_to_reach(al_errors, ERROR_THRESHOLD)

    baseline_curves, baseline_n_list = [], []
    for s in range(N_SEEDS_BASELINE):
        res = run_random_baseline(seed=s)
        errors = [b - TRUE_MIN for b in best_so_far(res)]
        baseline_curves.append(errors)
        n = evaluations_to_reach(errors, ERROR_THRESHOLD)
        baseline_n_list.append(n if n is not None else BUDGET)
    baseline_n_avg = float(np.mean(baseline_n_list))

    print("--- Resultados ---")
    print(
        f"Aprendizaje activo: alcanzó error <= {ERROR_THRESHOLD} en "
        f"{al_n if al_n else '>' + str(BUDGET)} simulaciones reales "
        f"(tiempo total incl. decisiones: {al_time:.2f}s)"
    )
    print(
        f"Fuerza bruta (muestreo aleatorio, promedio de {N_SEEDS_BASELINE} corridas): "
        f"{baseline_n_avg:.1f} simulaciones"
    )

    if al_n is not None:
        reduction = 100 * (1 - al_n / baseline_n_avg)
        print(f"\n>>> Reducción de simulaciones costosas: {reduction:.1f}% <<<\n")
    else:
        print(
            "\nEl aprendizaje activo no alcanzó el umbral dentro del presupuesto; "
            "subir BUDGET o max_iterations.\n"
        )

    print("--- Resumen del orquestador ---")
    for k, v in orch.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Calibración de incertidumbre del GP ---")
    test_points = random_uniform(PARAM_SPACE, 300, seed=999)
    true_vals = np.array([branin(p) for p in test_points])
    mean, std = orch.surrogate.predict(test_points)
    within_2sigma = np.abs(true_vals - mean) <= 2 * std
    coverage = 100 * within_2sigma.mean()
    print(
        f"Cobertura empírica de mu +/- 2*sigma sobre 300 puntos de test: {coverage:.1f}% "
        f"(objetivo del spec: ~95%)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(al_errors) + 1), al_errors, label="Aprendizaje activo (EI)", color="#1f77b4", linewidth=2)
    baseline_arr = np.array([c[:BUDGET] for c in baseline_curves])
    baseline_mean = baseline_arr.mean(axis=0)
    baseline_std = baseline_arr.std(axis=0)
    x = range(1, len(baseline_mean) + 1)
    ax.plot(x, baseline_mean, label=f"Muestreo aleatorio (promedio de {N_SEEDS_BASELINE})", color="#ff7f0e", linewidth=2)
    ax.fill_between(x, baseline_mean - baseline_std, baseline_mean + baseline_std, color="#ff7f0e", alpha=0.2)
    ax.axhline(ERROR_THRESHOLD, color="gray", linestyle="--", label=f"Umbral de error ({ERROR_THRESHOLD})")
    ax.set_xlabel("Simulaciones reales ejecutadas")
    ax.set_ylabel("Error respecto al mínimo global")
    ax.set_yscale("log")
    ax.set_title("Convergencia: aprendizaje activo vs. fuerza bruta (función de Branin)")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "convergencia_branin.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nGráfico de convergencia guardado en: {out_path}")

    csv_path = OUT_DIR / "historial_branin.csv"
    orch.store.to_csv(csv_path)
    print(f"Historial de evaluaciones (trazabilidad) guardado en: {csv_path}")


if __name__ == "__main__":
    main()
