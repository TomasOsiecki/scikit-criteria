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
# weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])

def calculate_real_evaluation(matrix, a, b):
    """
    (x_ij - a) / (b - a)
    in the maximization case, a is the minimum value of the column, b is the maximum value of the column
    in the minimization case, b is the maximum value of the column, a is the minimum value of the column
    """
    # TODO: intentar con for's
    pass

def mairca(matrix, objectives, weights):
    """ 
    Execute MAIRCA without any validation.
    MAIRCA (MultiAtributive Ideal-Real Comparative Analysis)
    """

    # Step 1
    # Define decision matrix X
    # X = matrix

    # Step 2
    # Calculate  P_ai
    # m = len(X)
    # P_ai = 1 / m

    # Step 3
    # Define the theoretical ranking matrix (Tp)
    # T_p = np.tile(weights * P_ai, (m, 1))

    # Step 4
    # Define the real rating matrix (Tr)
    # x_i_min = decision_matrix.min(axis=0)
    # x_i_max = decision_matrix.max(axis=0)
    # mask = objectives == 1
    # real_value = calculate_real_evaluation(X)
    # T_r = T_p * real_value

    # Step 5
    # Calculate Total Gap Matrix

    # Step 6
    # Calculate the final values of criteria functions (Qi) by alternatives


    pass

class MAIRCA(SKCDecisionMakerABC):
    """
    MAIRCA (MultiAtributive Ideal-Real Comparative Analysis)

    
    Parameters
    ----------
    matrix : ndarray
        Decision matrix where each row is an alternative and each column is a criterion.
    objectives : ndarray
        Array indicating if each criterion is to be maximized (1) or minimized (-1).
    weights : ndarray
        Array of weights for each criterion.

    """
    pass
