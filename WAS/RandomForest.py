import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def get_numerical_feature_importance(rf, feature_columns):
    importance_list = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 2)) for feature, importance in
                          zip(feature_columns, importance_list)]
    # Sort the feature importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return importance_list, feature_importance


def show_graph(importance, feature_headers):
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importance)))
    # Make a bar chart
    plt.bar(x_values, importance, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_headers, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.show()


def random_forest(data_sets, rf_instance):
    feature_headers = list(data_sets['test_features'].columns)
    rf_instance.fit(data_sets['train_features'], data_sets['train_labels'])
    predictions = rf_instance.predict(data_sets['test_features'])
    errors = abs(predictions - data_sets['test_labels'])
    mean_error = round(np.mean(errors), 2)
    importance_list, features_importance = get_numerical_feature_importance(rf_instance, feature_headers)

    return {'predictions': predictions, 'error': mean_error,
            'features_importance': features_importance, 'importance_list': importance_list}


def get_t_sets(rf_type, labels, historical_data, new_data):
    # get columns present in new data but not in historical data
    new_columns = list(new_data.columns.difference(historical_data.columns))
    # get columns present in historical data but not new data
    missing_historical_columns = list(historical_data.columns.difference(new_data.columns))

    all_labels = []
    for label_type, label in labels.items():
        all_labels.append(label)

    train_features = historical_data.drop(missing_historical_columns + all_labels, axis=1)
    test_features = new_data.drop(new_columns + all_labels, axis=1)
    train_labels = np.array(historical_data[labels[rf_type]])
    test_labels = np.array(new_data[labels[rf_type]])

    return {'train_features': train_features, 'test_features': test_features,
            'test_labels': test_labels, 'train_labels': train_labels}


def predict(historical_data, new_data, labels):
    results_dict = {}
    for key in labels.keys():
        sets = get_t_sets(key, labels, historical_data, new_data)
        rf_type = None
        if key == "classifier":
            rf_type = RandomForestClassifier(n_estimators=128)
        if key == "regressor":
            rf_type = RandomForestRegressor(n_estimators=128)
        results = random_forest(sets, rf_type)
        results_dict[key] = results
    return results_dict
