from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def polynomial_mutation_numba(
    X: np.ndarray,
    prob: float,
    eta: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """
    Apply polynomial mutation in-place.
    """
    n_ind, n_var = X.shape
    mut_pow = 1.0 / (eta + 1.0)
    
    # Pre-calculate span for optimization if needed, but per-gene is safer for varying bounds
    # Assuming bounds are (n_var,)
    
    for i in range(n_ind):
        for j in range(n_var):
            if np.random.random() <= prob:
                y = X[i, j]
                yl = lower[j]
                yu = upper[j]
                
                if yl >= yu:
                    continue
                    
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rnd = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (yu - yl)
                
                if y < yl:
                    y = yl
                if y > yu:
                    y = yu
                    
                X[i, j] = y


@njit(cache=True)
def sbx_crossover_numba(
    X_parents: np.ndarray,
    prob: float,
    eta: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Apply SBX crossover. 
    X_parents: (N, 2, n_var)
    Returns: offspring (N, 2, n_var) # Note: SBX normally returns same shape
    """
    # X_parents is shaped (n_pairs, 2, n_vars) based on operators.real.crossover usage
    # Wait, numpy config might flatten it? 
    # Let's check numba_backend.py: 
    #   Np, D = X_parents.shape
    #   pairs = X_parents.reshape(Np // 2, 2, D)
    # So the input to the actual op logic should probably be the Pairs.
    # Numba backend implementation of `sbx_crossover` currently receives (Np, D) flattened.
    # It reshapes it to (Np//2, 2, D).
    
    n_parents, n_var = X_parents.shape
    # Handle odd number of parents by ignoring the last one or duplicating.
    # The caller (numba_backend.py) handles resizing if needed.
    
    # We will assume X_parents is (N, n_var) and we process in pairs (0,1), (2,3)...
    # This avoids reshaping overhead inside the JIT function if possible, or we just reshape.
    # Actually, JIT works fine with 3D arrays. Let's accept (n_pairs, 2, n_var) for clarity.
    
    # BUT `numba_backend.sbx_crossover` signature receives `X_parents` (N, D).
    # I should align with that or do the reshape inside/outside. 
    # Doing it inside is cleaner.
    
    # Let's stick to the signature: receives flattened parents, returns flattened offspring.
    
    if n_parents % 2 != 0:
        # Should have been handled by caller, but safety check
        n_parents -= 1
        
    offspring = np.empty_like(X_parents)
    
    # Process pairs
    for i in range(0, n_parents, 2):
        # pair i and i+1
        
        # Check probability for this pair
        if np.random.random() <= prob:
            for j in range(n_var):
                y1 = X_parents[i, j]
                y2 = X_parents[i+1, j]
                
                yl = lower[j]
                yu = upper[j]
                
                if np.abs(y1 - y2) < 1.0e-14:
                    offspring[i, j] = y1
                    offspring[i+1, j] = y2
                    continue
                
                if y1 < y2:
                    y1_val = y1
                    y2_val = y2
                else:
                    y1_val = y2
                    y2_val = y1
                    
                # y1_val is min, y2_val is max
                    
                beta = 1.0 + (2.0 * (y1_val - yl) / (y2_val - y1_val))
                alpha = 2.0 - beta ** -(eta + 1.0)
                rand = np.random.random()
                
                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                c1 = 0.5 * ((y1_val + y2_val) - betaq * (y2_val - y1_val))
                
                beta = 1.0 + (2.0 * (yu - y2_val) / (y2_val - y1_val))
                alpha = 2.0 - beta ** -(eta + 1.0)
                
                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                c2 = 0.5 * ((y1_val + y2_val) + betaq * (y2_val - y1_val))
                
                # Clipping
                if c1 < yl: c1 = yl
                if c1 > yu: c1 = yu
                if c2 < yl: c2 = yl
                if c2 > yu: c2 = yu
                
                # Random swap is implied by the fact we sorted y1/y2? 
                # Standard SBX often creates two children from y1, y2.
                # The order in output depends on implementation. 
                # Standard pymoo/vamos: 
                # child1 = 0.5*((y1+y2) - betaq*(y2-y1))
                # child2 = 0.5*((y1+y2) + betaq*(y2-y1))
                # And then typically we swap them with 0.5 prob per variable or per individual.
                # VAMOS real.py implementation does swap per variable (prob 0.5).
                
                if np.random.random() <= 0.5:
                    offspring[i, j] = c2
                    offspring[i+1, j] = c1
                else:
                    offspring[i, j] = c1
                    offspring[i+1, j] = c2
        else:
            # Copy parents
            for j in range(n_var):
                offspring[i, j] = X_parents[i, j]
                offspring[i+1, j] = X_parents[i+1, j]

    return offspring
