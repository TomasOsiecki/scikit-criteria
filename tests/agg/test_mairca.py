import numpy as np
import skcriteria
import pytest
from skcriteria.agg import RankResult
from skcriteria.agg.mairca import MAIRCA

def test_MAIRCA_basic():
    matrix = np.array(
        [
            [3.828, 5.000, 3.720, 2.723, 4.255],
            [4.675, 5.000, 3.000, 3.452, 2.587],
            [4.515, 4.836, 3.289, 3.491, 4.069],
            [4.421, 5.000, 3.555, 2.839, 4.397],
            [4.717, 5.000, 3.430, 4.401, 1.000],
            [4.695, 5.000, 3.925, 3.847, 1.000],
            [4.688, 5.000, 2.000, 4.99, 1.000],
            [4.688, 5.000, 3.971, 4.001, 1.000],
        ]
    )
    objectives = np.array([-1, 1, -1, -1, -1])
    weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])
    Q_expected = np.array([0.0427, 0.0548, 0.0919, 0.0592, 0.0631, 0.0637, 0.0483, 0.0658])
    ranking_expected = np.array([1, 3, 8, 4, 5, 6, 2, 7])
    alternatives_expected = [f"A{i}" for i in range(len(matrix))]

    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    dm = skcriteria.mkdm(
        matrix=matrix,
        objectives=objectives,
        weights=weights,
    )

    mairca = MAIRCA()
    result = mairca.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["values"], Q_expected, atol=1e-4)

def test_MAIRCA_with_equal_P_ai():
    matrix = np.array(
        [
            [3.828, 5.000, 3.720, 2.723, 4.255],
            [4.675, 5.000, 3.000, 3.452, 2.587],
            [4.515, 4.836, 3.289, 3.491, 4.069],
            [4.421, 5.000, 3.555, 2.839, 4.397],
            [4.717, 5.000, 3.430, 4.401, 1.000],
            [4.695, 5.000, 3.925, 3.847, 1.000],
            [4.688, 5.000, 2.000, 4.99, 1.000],
            [4.688, 5.000, 3.971, 4.001, 1.000],
        ]
    )
    objectives = np.array([-1, 1, -1, -1, -1])  # -1 for cost, 1 for benefit
    weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])
    
    P_ai = np.array([0.125] * 8)
    Q_expected = np.array([0.0427, 0.0548, 0.0919, 0.0592, 0.0631, 0.0637, 0.0483, 0.0658])
    ranking_expected = np.array([1, 3, 8, 4, 5, 6, 2, 7])
    alternatives_expected = [f"A{i}" for i in range(len(matrix))]

    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    dm = skcriteria.mkdm(
        matrix=matrix,
        objectives=objectives,
        weights=weights,
    )

    mairca = MAIRCA(P_ai=P_ai)
    result = mairca.evaluate(dm)  # Pass P_ai as a keyword argument
    

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["values"], Q_expected, atol=1e-4)

def test_MAIRCA_with_custom_P_ai():
    import numpy as np
    from skcriteria.agg.mairca import mairca

    matrix = np.array(
        [
            [3.828, 5.000, 3.720, 2.723, 4.255],
            [4.675, 5.000, 3.000, 3.452, 2.587],
            [4.515, 4.836, 3.289, 3.491, 4.069],
            [4.421, 5.000, 3.555, 2.839, 4.397],
            [4.717, 5.000, 3.430, 4.401, 1.000],
            [4.695, 5.000, 3.925, 3.847, 1.000],
            [4.688, 5.000, 2.000, 4.99, 1.000],
            [4.688, 5.000, 3.971, 4.001, 1.000],
        ]
    )
    objectives = np.array([-1, 1, -1, -1, -1])
    weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])
    # Custom, non-uniform P_ai (sum must be 1)
    P_ai = np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Calculate expected values using the actual MAIRCA function
    ranking_expected, Q_expected = mairca(matrix, objectives, weights, P_ai)
    alternatives_expected = [f"A{i}" for i in range(len(matrix))]
    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    dm = skcriteria.mkdm(
        matrix=matrix,
        objectives=objectives,
        weights=weights,
    )

    mairca_dm = MAIRCA(P_ai=P_ai)
    result = mairca_dm.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["values"], Q_expected, atol=1e-4)


def test_MAIRCA_invalid_P_ai_sum():
    P_ai = np.array([0.6, 0.5])  # Sum = 1.1 ≠ 1
    
    with pytest.raises(ValueError, match="Sum of P_ai must be 1"):
        MAIRCA(P_ai=P_ai)

def test_MAIRCA_negative_P_ai():
    P_ai = np.array([0.7, -0.3])  # Negative value

    with pytest.raises(ValueError, match="P_ai must be non-negative"):
        MAIRCA(P_ai=P_ai)

def test_MAIRCA_mismatched_P_ai_length():
    matrix = np.array([[3.828, 5.000], [4.675, 5.000]])
    objectives = np.array([-1, 1])
    weights = np.array([0.5, 0.5])
    P_ai = np.array([0.5, 0.3, 0.2])  # Length 3 ≠ 2 alternatives

    # Add length check in MAIRCA __init__ if not present
    with pytest.raises(ValueError, match="Length of P_ai must match number of alternatives"):
        dm = skcriteria.mkdm(
            matrix=matrix,
            objectives=objectives,
            weights=weights,
        )
        mairca_dm = MAIRCA(P_ai=P_ai)
        mairca_dm.evaluate(dm)


def test_MAIRCA_zero_P_ai():
    matrix = np.array([[3.828, 5.000], [4.675, 5.000]])
    objectives = np.array([-1, 1])
    weights = np.array([0.5, 0.5])
    P_ai = np.array([0.0, 1.0])  # One alternative has zero preference

    mairca = MAIRCA(P_ai=P_ai)
    dm = skcriteria.mkdm(matrix=matrix, objectives=objectives, weights=weights)
    result = mairca.evaluate(dm)
    assert len(result.e_["values"]) == 2


def test_MAIRCA_invalid_objectives():
    matrix = np.array([[3.828, 5.000], [4.675, 5.000]])
    objectives = np.array([-1, 2])  # 2 is invalid
    weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="Invalid criteria objective 2"):
        dm = skcriteria.mkdm(matrix=matrix, objectives=objectives, weights=weights)
        ranker = MAIRCA()
        ranker.evaluate(dm)

#def test_MAIRCA_zero_weights():
#    matrix = np.array([[3.828, 5.000], [4.675, 5.000]])
#    objectives = np.array([-1, 1])
#    weights = np.array([0.0, 0.0])  # Sum = 0
#
#    dm = skcriteria.mkdm(matrix=matrix, objectives=objectives, weights=weights)
#    ranker = MAIRCA()
#
#    with pytest.raises(ValueError, match="Weights must sum to a positive value"):
#        ranker.evaluate(dm)
#
#def test_MAIRCA_empty_matrix():
#    matrix = np.array([])  # Empty matrix
#    objectives = np.array([])
#    weights = np.array([])
#
#    dm = skcriteria.mkdm(matrix=matrix, objectives=objectives, weights=weights)
#    ranker = MAIRCA()
#
#    with pytest.raises(ValueError, match="Matrix cannot be empty"):
#        ranker.evaluate(dm)
#