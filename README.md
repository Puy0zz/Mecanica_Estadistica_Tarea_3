# Simulación Monte Carlo del modelo de Ising — Algoritmo de Metropolis

Este documento describe la implementación contenida en `implementacion_metropolis.ipynb`, un notebook que simula la evolución de una red cuadrada de espines 1/2 (modelo de Ising) mediante el algoritmo de Metropolis-Hastings, incluyendo el cálculo de energía y magnetización en función de la temperatura.

## Requisitos

```python
numpy
matplotlib
numba
scipy
tqdm
```

## Parámetros físicos utilizados

| Símbolo | Descripción | Valor de ejemplo |
|---|---|---|
| `N` | Tamaño de la red (N x N) | 50 |
| `J` | Constante de acoplamiento (μeV) | 15000 |
| `Kb` | Constante de Boltzmann (μeV/K) | 86.17333262 |
| `beta` | Factor `1/(Kb*T)` | — |

---

## Funciones implementadas

### 1. Inicialización de la grilla

La grilla inicial se genera fuera de una función, muestreando valores aleatorios uniformes en `[0,1)` y asignando el espín según:

```
s[i,j] = +1  si random[i,j] <= 0.75
s[i,j] = -1  si random[i,j] > 0.75
```

Esto produce una configuración con ~75% de espines "arriba" y 25% "abajo", es decir, un estado inicial fuera del equilibrio térmico.

```python
N = 50
prop_spin_up = 0.75
grilla_random = np.random.rand(N, N)
grilla_init = np.where(grilla_random < prop_spin_up, 1, -1)
```

---

### 2. `metropolis_step(grilla, beta, i, j, J)`

Realiza **un único paso** del algoritmo de Metropolis-Hastings sobre la posición `(i, j)` de la grilla.

- **Decorador:** `@njit` (Numba) para acelerar la ejecución, dado que esta función se llama millones de veces.
- **Condiciones de frontera:** periódicas, implementadas mediante el operador módulo (`% N`) al acceder a los vecinos.
- **Lógica:**
  1. Calcula el cambio de energía `delta_E` que resultaría de invertir el espín en `(i, j)`, considerando la interacción con sus 4 primeros vecinos (arriba, abajo, izquierda, derecha):
     ```
     delta_E = 2 * J * spin_actual * (suma de los 4 vecinos)
     ```
  2. Aplica el criterio de aceptación de Metropolis:
     - Si `delta_E <= 0`, el cambio se acepta siempre (reduce o mantiene la energía).
     - Si `delta_E > 0`, se acepta con probabilidad `exp(-beta * delta_E)` (factor de Boltzmann).
  3. Si el cambio es rechazado, la grilla no se modifica y se retorna `delta_E = 0`.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|---|---|---|
| `grilla` | `np.ndarray` (N×N) | Estado actual de la red de espines |
| `beta` | `float` | Temperatura inversa `1/(Kb*T)` |
| `i`, `j` | `int` | Coordenadas del espín a evaluar |
| `J` | `float` | Constante de acoplamiento |

**Retorna:** `(grilla, delta_E)` — la grilla (posiblemente actualizada) y el cambio de energía efectivo.

---

### 3. `get_init_E(grilla, J)`

Calcula la **energía total inicial** del sistema de forma eficiente usando convolución 2D en lugar de un doble bucle `for` sobre toda la red.

- Se define un kernel de primeros vecinos:
  ```
  [[0, 1, 0],
   [1, 0, 1],
   [0, 1, 0]]
  ```
- Se convoluciona la grilla con este kernel (`scipy.signal.convolve2d`, `mode='same'`, `boundary='wrap'` para condiciones periódicas).
- La energía se calcula como:
  ```
  E = -J * sum(grilla * convolucion) / 2
  ```
  (se divide entre 2 para no contar cada interacción dos veces).

**Complejidad:** O(N² log N) mediante convolución, frente a O(N²) por vecino con un enfoque de doble loop ingenuo pero con mayor constante — ventajoso para redes grandes.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|---|---|---|
| `grilla` | `np.ndarray` (N×N) | Estado de la red de espines |
| `J` | `float` | Constante de acoplamiento |

**Retorna:** `E` (float) — energía total del sistema.

---

### 4. `evolucion_metropolis(grilla, beta, J, epochs)`

Evoluciona el sistema durante un número determinado de épocas, aplicando en cada una un paso de Metropolis sobre una posición aleatoria de la grilla, y registra la evolución temporal de la energía y la magnetización.

**Lógica:**
1. Calcula la energía inicial con `get_init_E`.
2. Calcula la magnetización inicial como `M = sum(grilla) / N²`.
3. Para cada época (`step` de `0` a `epochs-1`):
   - Elige aleatoriamente una posición `(i, j)`.
   - Aplica `metropolis_step`.
   - Actualiza la energía acumulando `delta_E`.
   - Recalcula la magnetización.
   - Guarda ambos valores en los arreglos de salida.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|---|---|---|
| `grilla` | `np.ndarray` (N×N) | Grilla inicial (se recomienda pasar una copia, `.copy()`) |
| `beta` | `float` | Temperatura inversa `1/(Kb*T)` |
| `J` | `float` | Constante de acoplamiento |
| `epochs` | `int` | Número de pasos de Metropolis a ejecutar |

**Retorna:**
| Valor | Tipo | Descripción |
|---|---|---|
| `grilla` | `np.ndarray` | Grilla final tras la evolución |
| `energias` | `np.ndarray` (epochs+1,) | Energía del sistema en cada época (incluye la inicial) |
| `magnetizaciones` | `np.ndarray` (epochs+1,) | Magnetización del sistema en cada época |

**Ejemplo de uso:**
```python
Temp = 293.15  # K
J = 15000      # mueV
Kb = 86.17333262  # mueV/K
grilla_final, energias, magnetizaciones = evolucion_metropolis(
    grilla_init.copy(), 1/(Kb*Temp), J, epochs=int(1e6)
)
```

---

### 5. `temperature_sweep(grilla, T_values, J, epochs)`

Realiza un **barrido de temperatura**: para cada valor de `T` en `T_values`, evoluciona el sistema durante `epochs` pasos y calcula los promedios de energía, magnetización y la desviación estándar de la energía, tomando solo las últimas 100 000 épocas (asumiendo que el sistema ya alcanzó el equilibrio térmico en ese tramo).

**Lógica:**
1. Para cada temperatura `T_values[t]` (con barra de progreso `tqdm`):
   - Calcula `beta = 1/(Kb*T)`.
   - Llama a `evolucion_metropolis` reutilizando la grilla resultante del paso anterior (`grilla_copy`), de modo que cada temperatura parte del estado final de la anterior.
   - Calcula:
     - `avg_E[t]`: promedio de energía en las últimas 100 000 épocas.
     - `std_E[t]`: desviación estándar de la energía en las últimas 100 000 épocas.
     - `avg_M[t]`: promedio de magnetización en las últimas 100 000 épocas.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|---|---|---|
| `grilla` | `np.ndarray` (N×N) | Grilla inicial |
| `T_values` | `np.ndarray` / lista | Valores de temperatura (K) a recorrer |
| `J` | `float` | Constante de acoplamiento |
| `epochs` | `int` | Número de épocas de Metropolis por temperatura |

**Retorna:**
| Valor | Tipo | Descripción |
|---|---|---|
| `avg_E` | `np.ndarray` (len(T_values),) | Energía promedio en equilibrio por temperatura |
| `avg_M` | `np.ndarray` (len(T_values),) | Magnetización promedio en equilibrio por temperatura |
| `std_E` | `np.ndarray` (len(T_values),) | Desviación estándar de la energía por temperatura |

**Ejemplo de uso:**
```python
J = 15000
T_values = np.linspace(200, 500, 300)  # de 200 K a 500 K
E_eq, M_eq, std_E = temperature_sweep(grilla_init, T_values, J, epochs=int(1e6))
```

> **Nota:** esta función es la más costosa computacionalmente del notebook (barrido sobre 300 temperaturas × 1e6 épocas cada una tomó ~28 minutos en las pruebas del notebook original).

---

## Flujo general del notebook

1. **Inicialización** de la grilla de espines con 75% "arriba" / 25% "abajo".
2. **Visualización** del estado inicial con `plt.imshow`.
3. **Evolución** del sistema a una temperatura fija (`evolucion_metropolis`) y graficación de energía y magnetización contra el número de épocas, para distintas temperaturas (293.15 K, 395 K, 450 K).
4. **Barrido de temperatura** (`temperature_sweep`) para construir las curvas de energía, magnetización y varianza de la energía en función de la temperatura (relacionadas con el calor específico y la transición de fase del sistema).

## Notas de implementación

- El uso de `@njit` de Numba en `metropolis_step` es crítico para el rendimiento, ya que esta función se ejecuta una vez por cada época simulada (hasta 3×10⁸ veces en el barrido completo).
- Las condiciones de frontera periódicas se implementan consistentemente tanto en `metropolis_step` (vía `% N`) como en `get_init_E` (vía `boundary='wrap'` en la convolución).
- Se recomienda pasar siempre una **copia** de la grilla (`grilla.copy()`) al llamar a `evolucion_metropolis`, ya que la función modifica el arreglo *in place*.
