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
    # P_ai = 1 / len(X)

    # Step 3
    # Define the theoretical ranking matrix (Tp)

    # Step 4
    # Define the real rating matrix (Tr)

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
