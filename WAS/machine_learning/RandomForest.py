import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from machine_learning.classes.RF_Result import RFResult
from machine_learning.classes.Label import Label


def get_features_importance(rf, features):
    # get the list of features and their gini importance
    importance_list = list(rf.feature_importances_)
    # round the importance and tuple feature name with its importance value
    feature_importance = [(feature, round(importance, 2)) for feature, importance in
                          zip(features, importance_list)]
    return feature_importance


def random_forest(label, train_features, test_features, rf_instance):
    train_label_values, test_label_values = label.encoded_train_test()
    train_label_values = np.array(train_label_values)
    test_label_values = np.array(test_label_values)
    rf_instance.fit(train_features, train_label_values)
    predictions = rf_instance.predict(test_features)
    if label.rf_type == 'classifier':
        matches = list(np.array(predictions) == test_label_values)
        error = matches.count(False) / len(matches)
    else:
        # we calculate the total mean error,
        # then divide that by the mean value of the label to get a difference in percentage
        total_prediction_error = sum(abs(predictions - test_label_values))
        total_label_values = sum(value for value in test_label_values)
        error = total_prediction_error / total_label_values
        print(error)
    features_importance = get_features_importance(rf_instance, test_features.columns)
    predictions = label.decode(predictions)
    rf_result = RFResult(label_name=label.name, predictions=predictions,
                         features_importance=features_importance, error=round(error, 2))
    return rf_result


def predict(historical_data, new_data, rf_labels):
    results_dict = {}
    # drop the labels from features testing data
    all_labels = list(set().union(*rf_labels.values()))
    test_features = new_data.drop(all_labels, axis=1)
    # hot encode the categorical features and fill empty cells with -1, representing the lack of data
    hotenc_train_feat = pd.get_dummies(pd.DataFrame.from_records(historical_data).fillna(0), prefix_sep="!-->")
    hotenc_test_feat = pd.get_dummies(pd.DataFrame.from_records(test_features).fillna(0), prefix_sep="!-->")
    # inner join to match the shape between the two dataframes
    hotenc_train_feat, hotenc_test_feat = hotenc_train_feat.align(hotenc_test_feat, join='inner', axis=1)

    if hotenc_train_feat.empty or hotenc_test_feat.empty:
        sys.stderr.write("The dataframe couldn't provide any feature. The labels cannot be inferred. \n")
        return results_dict
    # for each label, encode the label and run RF
    for rf_name, labels in rf_labels.items():
        results_dict[rf_name] = []
        for label_name in labels:
            try:
                if label_name not in historical_data.columns:
                    historical_data[label_name] = np.NAN
                if rf_name == "classifier":
                    encoder = LabelEncoder()
                    train_label_values = historical_data[label_name].astype(str).fillna('None')
                    test_label_values = new_data[label_name].astype(str).fillna('None')
                    rf_instance = RandomForestClassifier(n_estimators=128)
                else:
                    encoder = None
                    train_label_values = historical_data[label_name].fillna(0)
                    test_label_values = new_data[label_name].fillna(0)
                    rf_instance = RandomForestRegressor(n_estimators=128)
                label_instance = Label(name=label_name, rf_type=rf_name,
                                       train_values=train_label_values,
                                       test_values=test_label_values,
                                       encoder=encoder)
                results = random_forest(label_instance,
                                        hotenc_train_feat,
                                        hotenc_test_feat,
                                        rf_instance)
                results_dict[rf_name].append(results)
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return results_dict
