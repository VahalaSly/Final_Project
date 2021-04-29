from analysis.ProcessResults import is_error_under_threshold
from analysis.ProcessResults import get_valid_prediction_features


def test_is_error_under_threshold():
    low_error = 0.1
    high_error = 0.3
    assert is_error_under_threshold(high_error) is False
    assert is_error_under_threshold(low_error) is True


def test_get_valid_prediction_features():
    feature_importance_pairs = [('A', 0.02), ('B', 0.1), ('C', 0), ('D', 0.7)]
    assert get_valid_prediction_features(True, feature_importance_pairs) == [('B', 0.1), ('D', 0.7)]
    assert get_valid_prediction_features(False, feature_importance_pairs) == []

    # edge cases
    assert get_valid_prediction_features(True, []) == []
