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
    if not P_ai:
        m = len(matrix)  
        P_ai = 1 / m

    # Step 3
    # Define the theoretical ranking matrix (Tp)
    T_p = np.tile(weights * P_ai, (m, 1))

    # Step 4
    # Define the real rating matrix (Tr)
    mask = objectives == Objective.MAX.value  # 1
    T_r = np.where(mask, 
                   T_p * ((matrix-matrix.min(axis=0))/(matrix.max(axis=0)-matrix.min(axis=0))), 
                   T_p * ((matrix-matrix.max(axis=0))/(matrix.min(axis=0)-matrix.max(axis=0)))
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
    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, P_ai=None, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError("MAIRCA can't operate with values <= 0")
        if not P_ai:
            raise Warning(
                "Preferences for the choice of alternatives not found."
                "Using default value."
            )
        rank, q_i = mairca(
            matrix,
            objectives,
            weights,
            P_ai,
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