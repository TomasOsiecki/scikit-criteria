#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils import hidden

with hidden():
    import numpy as np
    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank

# =============================================================================
# MAIRCA
# =============================================================================

# decision_matrix = np.array(
#     [
#         [3.828,	5.000,	3.720,	2.723,	1.000,	4.255],
#         [4.675,	5.000,	3.000,	3.452,	1.000,	2.587],
#         [4.515,	4.836,	3.289,	3.491,	1.000,	4.069],
#         [4.421,	5.000,	3.555,	2.839,	1.000,	4.397],
#         [4.717,	5.000,	3.430,	4.401,	1.000,	1.000],
#         [4.695,	5.000,	3.925,	3.847,	1.000,	1.000],
#         [4.688,	5.000,	2.000,	4.99,	1.000,	1.000],
#         [4.688,	5.000,	3.971,	4.001,	1.000,	1.000],
#     ]
# )

# n columnas, m filas
# caso de test: la suma de los p_ai's debe ser 1

# si hay una columna con todos 1, se hace:
# decision_matrix = np.delete(decision_matrix, ...)
# decision_matrix = np.delete(decision_matrix, column_index, axis=1)
# weights = np.delete(weights, column_index)
# objectives = np.delete(objectives, column_index)
# weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])
# objectives = np.array([-1, 1, -1, -1, -1])


def mairca(matrix, objectives, weights, P_ai):
    """ 
    Execute MAIRCA without any validation.
    MAIRCA (MultiAtributive Ideal-Real Comparative Analysis)
    """

    # Step 2
    # Calculate  P_ai
    m = len(matrix)  
    if P_ai is None:
        P_ai = np.ones(m) / m

    # Step 3
    # Define the theoretical ranking matrix (Tp)
    T_p = P_ai[:, np.newaxis] * weights[np.newaxis, :]

    # Step 4
    # Define the real rating matrix (Tr)
    mask = objectives == Objective.MAX.value  # 1
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    T_r = np.zeros_like(T_p)
    
    for j in range(matrix.shape[1]):
        if min_vals[j] == max_vals[j]:  # All alternatives have same value for this criterion
            # Use theoretical evaluation (no gap)
            T_r[:, j] = T_p[:, j]
        else:
            if mask[j]:  # Benefit criterion
                T_r[:, j] = T_p[:, j] * (
                    (matrix[:, j] - min_vals[j]) / (max_vals[j] - min_vals[j])
                )
            else:  # Cost criterion
                T_r[:, j] = T_p[:, j] * (
                    (matrix[:, j] - max_vals[j]) / (min_vals[j] - max_vals[j])
                )
    
    # Step 5
    # Calculate Total Gap Matrix
    G = T_p - T_r

    # Step 6
    # Calculate the final values of criteria functions (Qi) by alternatives
    Q_i = G.sum(axis=1)

    ranking = rank.rank_values(Q_i, reverse=False)  # or reverse=True depending on your method
    return ranking, Q_i

class MAIRCA(SKCDecisionMakerABC):
    """
    MAIRCA (MultiAtributive Ideal-Real Comparative Analysis)
    """
    _skcriteria_parameters = ["P_ai"]  # Declare P_ai as a valid parameter

    def __init__(self, P_ai=None):
        if P_ai is not None:
            if not isinstance(P_ai, np.ndarray):
                raise ValueError("P_ai must be numpy array.")
            if np.any(P_ai < 0):
                raise ValueError("P_ai must be non-negative.")
            if not np.isclose(np.sum(P_ai), 1):
                raise ValueError("Sum of P_ai must be 1.")
        self._P_ai = P_ai

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError("MAIRCA can't operate with values <= 0")
        if self._P_ai is not None and len(self._P_ai) != len(matrix):
            raise ValueError("Length of P_ai must match number of alternatives")
        rank, q_i = mairca(
            matrix,
            objectives,
            weights,
            self._P_ai,
        )
        return rank, {
            "values": q_i,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MAIRCA",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )