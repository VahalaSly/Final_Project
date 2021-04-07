import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


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

    if data_sets['encoder'] is not None:
        predictions = data_sets['encoder'].inverse_transform(predictions)

    return {'predictions': predictions, 'error': mean_error,
            'features_importance': features_importance}


def get_t_sets(rf_type, target_label, historical_data, new_data):
    train = historical_data.copy(deep=True)
    test = new_data.copy(deep=True)

    le = None
    if rf_type == 'classifier':
        le = LabelEncoder()
        label_values = list(train[target_label]) + list(test[target_label])
        le.fit(label_values)
        train[target_label] = le.transform(train[target_label])
        test[target_label] = le.transform(test[target_label])

    # hot encode the categorical features
    hotenc_hist_data = pd.get_dummies(pd.DataFrame.from_records(train).fillna(-1),
                                      prefix_sep="!-->")
    hotenc_new_data = pd.get_dummies(pd.DataFrame.from_records(test).fillna(-1),
                                     prefix_sep="!-->")

    final_train, final_test = hotenc_hist_data.align(hotenc_new_data, join='inner', axis=1)

    try:
        train_features = final_train.drop(target_label, axis=1)
        test_features = final_test.drop(target_label, axis=1)
        train_labels = np.array(final_train[target_label])
        test_labels = np.array(final_test[target_label])

        return {'train_features': train_features, 'test_features': test_features,
                'test_labels': test_labels, 'train_labels': train_labels, 'encoder': le}
    except KeyError as e:
        sys.stderr.write(str(e) + "\n")
        raise KeyError


def predict(historical_data, new_data, rf_labels):
    results_dict = {}
    for rf, labels in rf_labels.items():
        results_dict[rf] = {}
        for target_label in labels:
            try:
                sets = get_t_sets(rf, target_label, historical_data, new_data)
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
