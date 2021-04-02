import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def get_numerical_feature_importance(rf, feature_columns):
    importance_list = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 2)) for feature, importance in
                          zip(feature_columns, importance_list)]
    # Sort the feature importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return importance_list, feature_importance


def random_forest(data_sets, rf_instance):
    feature_headers = list(data_sets['test_features'].columns)
    rf_instance.fit(data_sets['train_features'], data_sets['train_labels'])
    predictions = rf_instance.predict(data_sets['test_features'])
    errors = abs(predictions - data_sets['test_labels'])
    mean_error = round(np.mean(errors), 2)
    importance_list, features_importance = get_numerical_feature_importance(rf_instance, feature_headers)

    return {'predictions': predictions, 'error': mean_error,
            'features_importance': features_importance}


def get_t_sets(target_label, all_labels, historical_data, new_data):
    # get columns present in new data but not in historical data
    new_columns = list(new_data.columns.difference(historical_data.columns))
    # get columns present in historical data but not new data
    missing_historical_columns = list(historical_data.columns.difference(new_data.columns))

    try:
        train_features = historical_data.drop(missing_historical_columns + all_labels, axis=1)
        test_features = new_data.drop(new_columns + all_labels, axis=1)
        train_labels = np.array(historical_data[target_label])
        test_labels = np.array(new_data[target_label])

        return {'train_features': train_features, 'test_features': test_features,
                'test_labels': test_labels, 'train_labels': train_labels}
    except KeyError as e:
        sys.stderr.write(str(e) + "\n")
        raise KeyError


def predict(historical_data, new_data, rf_labels):
    results_dict = {}
    all_labels = list(set().union(*rf_labels.values()))
    for rf, labels in rf_labels.items():
        results_dict[rf] = {}
        for target_label in labels:
            try:
                sets = get_t_sets(target_label, all_labels, historical_data, new_data)
                rf_type = None
                if rf == "classifier":
                    rf_type = RandomForestClassifier(n_estimators=128)
                if rf == "regressor":
                    rf_type = RandomForestRegressor(n_estimators=128)
                results = random_forest(sets, rf_type)
                results_dict[rf][target_label] = results
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return results_dict
