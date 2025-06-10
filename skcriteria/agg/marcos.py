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

    extended_data = np.vstack([anti_ideal, matrix, ideal])
    normalized = extended_data / ideal
    weighted = normalized * weights
    S = weighted.sum(axis=1)
    Si = S[1:-1]
    S_ai = S[0]
    S_i = S[-1]

    Ki = Si / S_ai
    f_Ki = Si / S_i

    return f_Ki, Ki, ideal, anti_ideal


class MARCOS(SKCDecisionMakerABC):
    """
    MARCOS (Measurement Alternatives and Ranking according to COmpromise Solution).

    This method evaluates and ranks alternatives based on their compromise distance
    to the ideal and anti-ideal solutions. It is suitable for multi-criteria decision
    analysis (MCDA) problems with both maximizing and minimizing criteria.

    References
    ----------
    Željko Stević, Dragan Pamučar, Adis Puška, Prasenjit Chatterjee.
    "Sustainable supplier selection in healthcare industries using a new MCDM method:
    Measurement Alternatives and Ranking according to COmpromise Solution (MARCOS)".
    Computers & Industrial Engineering, Volume 140, 2020, 106231.
    https://doi.org/10.1016/j.cie.2019.106231

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

    _skcriteria_parameters = ["weights"]

    def __init__(self, weights):
        self.weights = np.asarray(weights)

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, **kwargs):
        if len(self.weights) != matrix.shape[1]:
            raise ValueError(
                "Number of weights must match number of criteria."
            )
        f_Ki, Ki, ideal, anti_ideal = marcos(matrix, objectives, self.weights)
        ranking = rank.rank_values(f_Ki, reverse=True)
        return ranking, {
            "utility_scores": f_Ki,
            "ki_scores": Ki,
            "ideal": ideal,
            "anti_ideal": anti_ideal,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MARCOS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
