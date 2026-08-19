import time

from smartsim import SmartSimOrchestrator, SimulatorWrapper, branin

PARAM_SPACE = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}


def test_run_produces_exact_history_length():
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim, param_space=PARAM_SPACE, n_initial=5, max_iterations=10,
        acquisition="ucb", seed=1, verbose=False,
    )
    store, _ = orch.run()
    assert len(store) == 15
    assert sim.n_calls == 15


def test_every_history_point_is_a_real_simulation():
    """Verifica que n_calls del simulador sea siempre igual al largo del historial."""
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim, param_space=PARAM_SPACE, n_initial=4, max_iterations=6,
        acquisition="ei", seed=2, verbose=False,
    )
    store, _ = orch.run()
    assert sim.n_calls == len(store)


def test_uncertainty_stop_criterion_can_end_early():
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim, param_space=PARAM_SPACE, n_initial=6, max_iterations=200,
        acquisition="variance", seed=3, verbose=False,
        uncertainty_stop_threshold=1000.0,
    )
    store, _ = orch.run()
    assert len(store) == 6


def test_active_learning_reduces_error_vs_initial_only():
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim, param_space=PARAM_SPACE, n_initial=8, max_iterations=20,
        acquisition="ucb", minimize=True, seed=0, verbose=False,
    )
    store, _ = orch.run()
    results = [r for _, r in store.as_history_tuples()]
    best_after_initial = min(results[:8])
    best_final = min(results)
    assert best_final <= best_after_initial


def test_acquisition_decision_time_under_5s():
    sim = SimulatorWrapper(fn=branin)
    orch = SmartSimOrchestrator(
        simulator=sim, param_space=PARAM_SPACE, n_initial=8, max_iterations=1,
        acquisition="ucb", seed=0, n_candidates=2000, verbose=False,
    )
    orch._sample_initial_points()
    for i, point in enumerate(orch._sample_initial_points()):
        result = orch.simulator.run(point)
        orch.store.add(i, point, result, phase="initial")
    orch.surrogate.fit(orch.store.as_history_tuples())

    t0 = time.perf_counter()
    orch._optimize_acquisition()
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
