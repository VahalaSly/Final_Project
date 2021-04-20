import pandas

from machine_learning.classes.Label import Label
from machine_learning.RandomForest import get_features_importance
from machine_learning.RandomForest import random_forest
from machine_learning.RandomForest import predict

sample_dataframe = pandas.DataFrame({
    'id': ['0', '1', '2', '3', 4],
    'name': ['name0', 'name1', 'name2', 'name3', 'name3'],
    'state': ['executed', 'executed', 'executed', 'failed', 'failed'],
    'duration': [3, 2, 3, 3, 4]
})

sample_larger_dataframe = pandas.DataFrame({
    'id': ['1', '1', '3', 4, 4, 4, 4],
    'name': ['name1', 'name2', 'name3', 'name3', 'name3', 'name3', 'name3'],
    'state': ['failed', 'executed', 'failed', 'failed', 'failed', 'executed', 'executed'],
    'duration': [2, 3, 1, 1, 2, 3, 3]
})

mock_predictions = [4, 2, 0, 1, 2]
mock_importance = [0.3, 0.1]
mock_feature_importance = [('id', 0.3), ('name', 0.1)]


# we don't want to actually call random forest, therefore we create a mock instance
class MockRandomForest:
    def __init__(self):
        self.feature_importances_ = mock_importance

    def fit(self, a, b):
        pass

    def predict(self, a):
        return mock_predictions


def test_get_features_importance():
    rf = MockRandomForest()
    features = ['id', 'name']
    assert get_features_importance(rf, features) == mock_feature_importance


def test_random_forest():
    # label, train_features, test_features, rf_instance
    rf_instance = MockRandomForest()
    train_features = sample_larger_dataframe
    test_features = sample_dataframe
    label = Label(name='duration', rf_type='regressor', train_values=sample_larger_dataframe['duration'],
                  test_values=sample_dataframe['duration'], encoder=None)
    result = random_forest(label, train_features, test_features, rf_instance)
    assert result.label_name == 'duration'
    assert result.predictions == mock_predictions
    assert result.features_importance == mock_feature_importance
    assert result.error == 1.6


def test_predict():
    # historical_data, new_data, rf_labels
    rf_labels = {'classifier': ['state', 'name'], 'regressor': ['duration']}
    classifier_results = predict(sample_larger_dataframe, sample_dataframe, rf_labels)['classifier']
    regressor_results = predict(sample_larger_dataframe, sample_dataframe, rf_labels)['regressor']
    assert len(classifier_results) == 2
    assert len(regressor_results) == 1
