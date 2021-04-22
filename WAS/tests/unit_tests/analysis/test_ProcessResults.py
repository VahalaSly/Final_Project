from analysis.ProcessResults import is_error_under_threshold
from analysis.ProcessResults import get_valid_prediction_features


def test_is_error_under_threshold():
    classif_low_error = 0.1
    classif_high_error = 0.3
    regress_sample_label_values = [2, 5, 7, 3, 8, 9]
    classif_sample_values = [0, 0, 1, 0, 0]
    regress_low_error = 1
    regress_high_error = 2
    assert is_error_under_threshold('regressor', regress_sample_label_values, regress_high_error) is False
    assert is_error_under_threshold('regressor', regress_sample_label_values, regress_low_error) is True
    assert is_error_under_threshold('classifier', classif_sample_values, classif_high_error) is False
    assert is_error_under_threshold('classifier', classif_sample_values, classif_low_error) is True


def test_get_valid_prediction_features():
    feature_importance_pairs = [('A', 0.02), ('B', 0.1), ('C', 0), ('D', 0.7)]
    assert get_valid_prediction_features(True, feature_importance_pairs) == [('B', 0.1), ('D', 0.7)]
    assert get_valid_prediction_features(False, feature_importance_pairs) == []
