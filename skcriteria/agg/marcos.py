#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils import hidden

with hidden():
    import numpy as np
    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank

# =============================================================================
# MARCOS
# =============================================================================


def marcos(matrix, objectives, weights):
    """
    Execute MARCOS without any validation.
    MARCOS (Measurement Alternatives and Ranking according to COmpromise Solution)
    """
    mask = objectives == Objective.MAX.value
    ideal = np.where(mask, matrix.max(axis=0), matrix.min(axis=0))
    anti_ideal = np.where(mask, matrix.min(axis=0), matrix.max(axis=0))

    # Step 1: Add ideal and anti-ideal to the matrix
    extended_data = np.vstack([anti_ideal, matrix, ideal])

    # Step 2: Normalize with respect to the ideal
    normalized = np.zeros_like(extended_data, dtype=float)
    for i in range(extended_data.shape[1]):
        if objectives[i] == Objective.MAX.value:
            normalized[:, i] = extended_data[:, i] / ideal[i]
        else:
            normalized[:, i] = ideal[i] / extended_data[:, i]

    # Step 3: Weight the normalized values
    weighted = normalized * weights

    # Step 4: Sum the weighted values (Si)
    S = weighted.sum(axis=1)
    Si = S[1:-1]  # values for actual alternatives
    S_aideal = S[0]  # value for anti-ideal
    S_ideal = S[-1]  # value for ideal

    # Step 5: Compute utility functions
    K_minus = Si / S_aideal  # Relation to anti-ideal
    K_plus = Si / S_ideal  # Relation to ideal

    # Step 6: Final utility function
    f_K_plus = K_minus / (K_plus + K_minus)
    f_K_minus = K_plus / (K_plus + K_minus)
    f_K = (
        (K_plus + K_minus)
        / ((1 - f_K_plus) * (1 - f_K_minus))
        * (1 / (1 + f_K_plus / f_K_minus))
    )

    return Si, K_minus, K_plus, f_K


class MARCOS(SKCDecisionMakerABC):
    """
    MARCOS (Measurement Alternatives and Ranking according to COmpromise Solution).

    This method evaluates and ranks alternatives based on their compromise distance
    to the ideal and anti-ideal solutions. It is suitable for multi-criteria decision
    analysis (MCDA) problems with both maximizing and minimizing criteria.

    References
    ----------
    Stević, Z., Pamučar, D., Puška, A., Chatterjee, P., Sustainable supplier selection in
    healthcare industries using a new MCDM method: Measurement Alternatives and Ranking according to
    COmpromise Solution (MARCOS), Computers & Industrial Engineering (2019)

    Parameters
    ----------
    weights : array-like of shape (n_criteria,)
        The weights assigned to each criterion.

    Examples
    --------
    >>> from skcriteria.agg.marcos import MARCOS
    >>> dm = MARCOS(weights=[0.3, 0.7])
    >>> result = dm.evaluate(matrix)
    >>> print(result.rank)
    """

    _skcriteria_parameters = [] 

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if weights is None:
            raise ValueError(
                "weights parameter is needed."
            )
        if len(weights) != matrix.shape[1]:
            raise ValueError(
                "Number of weights must match number of criteria."
            )
        Si, K_minus, K_plus, f_K = marcos(matrix, objectives, weights)
        ranking = rank.rank_values(f_K, reverse=True)
        return ranking, {
            "utility_scores": f_K,
            "Si": Si,
            "K_minus": K_minus,
            "K_plus": K_plus,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MARCOS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
