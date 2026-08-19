# SmartSim Active Learner

## El problema

Entrenar o correr un simulador de un sistema físico o dinámico complejo
(robótica, sistemas autónomos, cualquier proceso regido por ecuaciones de
estado) es costoso computacionalmente. Para caracterizar cómo se comporta el
sistema bajo distintas condiciones iniciales o parámetros hay que correr el
simulador una y otra vez, y ese costo escala rápido con la cantidad de
parámetros y de puntos a explorar.

## La solución propuesta

Un modelo sustituto (Proceso Gaussiano) que aprende de las simulaciones ya
ejecutadas y estima su propia incertidumbre en cada predicción. Un bucle de
aprendizaje activo usa esa incertidumbre para decidir en cada iteración si
conviene confiar en la predicción barata del modelo o si hace falta correr
el simulador real. El simulador de alta fidelidad nunca se sustituye por la
aproximación: solo se ejecuta en el punto que la función de adquisición
(UCB, EI, PI, o exploración pura por varianza) señala como el más valioso.

Sobre el benchmark sintético de Branin, el prototipo alcanza el mismo error
que un muestreo aleatorio con ~65% menos simulaciones reales, y el 94.7% de
las predicciones del modelo caen dentro de μ ± 2σ (incertidumbre calibrada).

## Cómo levantarla

### Instalación

```bash
cd "modelo del mundo"
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

### Correr la demo

```bash
.venv/Scripts/python demo.py
```

Corre el sistema sobre la función de Branin, compara contra muestreo
aleatorio en consola, y genera:

- `outputs/convergencia_branin.png` — curva de convergencia (error vs. nº de
  simulaciones reales), aprendizaje activo vs. fuerza bruta.
- `outputs/historial_branin.csv` — cada punto evaluado, con su fase
  (`initial`/`active`), resultado y score de adquisición.

### Correr los tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

### Usar con tu propio simulador

`SimulatorWrapper` solo necesita una función `fn(params: dict) -> float`.
Podés envolver un binario externo, una API, o un notebook.

```python
from smartsim import SmartSimOrchestrator, SimulatorWrapper

def mi_simulador(params: dict) -> float:
    ...
    return resultado

sim = SimulatorWrapper(fn=mi_simulador)
param_space = {"param1": (0.0, 10.0), "param2": (-5.0, 5.0)}

orch = SmartSimOrchestrator(
    simulator=sim,
    param_space=param_space,
    n_initial=8,             # ejecuciones reales para el muestreo inicial (LHS)
    max_iterations=50,       # tope de iteraciones activas
    acquisition="ei",        # "ucb" | "ei" | "pi" | "variance"
    minimize=True,           # False si el objetivo es maximizar
    uncertainty_stop_threshold=None,  # ej. 0.05 para cortar cuando el GP ya está seguro
)
store, surrogate = orch.run()
print(orch.summary())
```

`acquisition="variance"` explora sin noción de "mejor resultado": prioriza
siempre el candidato con mayor incertidumbre del modelo, apto para
caracterizar el sistema en toda una región en vez de buscar un óptimo.
`"ucb"`, `"ei"` y `"pi"` en cambio balancean exploración con explotación
del mejor valor encontrado, apto para optimización.

## Estructura

```
smartsim/
  simulator.py      # SimulatorWrapper + funciones sintéticas Branin/Hartmann6
  surrogate.py       # Proceso Gaussiano: media + incertidumbre
  acquisition.py       # UCB, EI, PI, exploración por varianza
  sampling.py            # Latin Hypercube (inicial) + uniforme (candidatos)
  storage.py                # Historial de evaluaciones -> CSV
  orchestrator.py             # Bucle de aprendizaje activo
demo.py                        # Demo end-to-end sobre Branin
tests/                          # pytest: surrogate, acquisition, orchestrator
```

## Fuera de alcance en este prototipo

- Multi-fidelidad (combinar un simulador barato + uno caro).
- Selección de lotes de puntos en paralelo (batch acquisition).
- Optimización multi-objetivo.
