import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def get_features_importance(rf, feature_names):
    # get the list of features and their gini importance
    importance_list = list(rf.feature_importances_)
    # round the importance and tuple feature name with its importance value
    feature_importance = [(feature, round(importance, 2)) for feature, importance in
                          zip(feature_names, importance_list)]
    return feature_importance


def random_forest(label_set, train_features, test_features, rf_instance, rf_type):
    feature_names = list(test_features.columns)
    rf_instance.fit(train_features, label_set['train_labels'])
    predictions = rf_instance.predict(test_features)
    if rf_type == 'classifier':
        c = list(np.array(predictions) == np.array(label_set['test_labels']))
        mean_error = 1 - c.count(True)/len(c)
    else:
        errors = abs(predictions - label_set['test_labels'])
        mean_error = np.mean(errors)
    features_importance = get_features_importance(rf_instance, feature_names)

    if label_set['encoder'] is not None:
        predictions = label_set['encoder'].inverse_transform(predictions)

    return {'predictions': predictions, 'error': round(mean_error, 2),
            'features_importance': features_importance}


def get_label_set(rf_type, train_label_col, test_label_col):
    le = None
    if rf_type == 'classifier':
        le = LabelEncoder()
        label_values = list(train_label_col) + list(test_label_col)
        le.fit(label_values)
        train_label_col = le.transform(train_label_col)
        test_label_col = le.transform(test_label_col)

    try:
        train_labels = np.array(train_label_col)
        test_labels = np.array(test_label_col)

        return {'test_labels': test_labels, 'train_labels': train_labels, 'encoder': le}
    except KeyError as e:
        sys.stderr.write(str(e) + "\n")
        raise KeyError


def predict(historical_data, new_data, rf_labels):
    results_dict = {}

    # drop the labels from features data
    all_labels = list(set().union(*rf_labels.values()))
    train_features = historical_data.drop(all_labels, axis=1)
    test_features = new_data.drop(all_labels, axis=1)

    # hot encode the categorical features and fill empty cells with -1, representing the lack of data
    hotenc_train_feat = pd.get_dummies(pd.DataFrame.from_records(train_features).fillna(-1),
                                       prefix_sep="!-->")
    hotenc_test_feat = pd.get_dummies(pd.DataFrame.from_records(test_features).fillna(-1),
                                      prefix_sep="!-->")

    # inner join to match the shape between the two dataframes
    hotenc_train_feat, hotenc_test_feat = hotenc_train_feat.align(hotenc_test_feat, join='inner', axis=1)

    # for each label, encode the label and run RF
    for rf, labels in rf_labels.items():
        results_dict[rf] = {}
        for target_label in labels:
            try:
                train_label = historical_data[target_label].fillna(-1)
                test_label = new_data[target_label].fillna(-1)
                label_set = get_label_set(rf, train_label,
                                          test_label)
                rf_instance = None
                if rf == "classifier":
                    rf_instance = RandomForestClassifier(n_estimators=128)
                if rf == "regressor":
                    rf_instance = RandomForestRegressor(n_estimators=128)
                results = random_forest(label_set, hotenc_train_feat, hotenc_test_feat, rf_instance, rf)
                results_dict[rf][target_label] = results
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return results_dict
