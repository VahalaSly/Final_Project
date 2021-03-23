import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def random_forest(train_features, test_features, train_labels, test_labels):
    rf = RandomForestClassifier(n_estimators=128, random_state=42)
    feature_headers = list(test_features.columns)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    # get absolute error
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    importance_list, features_importance = get_numerical_feature_importance(rf, feature_headers)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in features_importance]


def main(historical_data_path, latest_execution_path):
    current_workflow = pd.get_dummies(pd.read_csv(latest_execution_path),
                                      columns=["node_name"])
    historical_data = False
    try:
        historical_data = pd.read_csv(historical_data_path)

        # we need to compare the columns between historical and current workflow data
        # the columns are compared between the two dataframes
        # the extra columns are dropped from the current workflow data
        # then the historical columns missing from the current data are also dropped
        # this is because the training (historical_data) and testing (current_workflow) columns need to match
        # and the algo would not be able to infer anything form the differing columns anyway
        # the new columns will be added later as part of the new historical data by the WAS

        new_columns = list(current_workflow.columns.difference(historical_data.columns))
        missing_historical_columns = list(historical_data.columns.difference(current_workflow.columns))
        label_columns = ['has_failed', 'execution_duration']

        # the columns added to the current_workflow are given value 0
        # since the new columns are always nodes
        # and if the workflow doesn't have the column, it didn't have the node either
        # which is represented by the value of 0

        train_features = historical_data.drop(missing_historical_columns + label_columns, axis=1)
        test_features = current_workflow.drop(new_columns + label_columns, axis=1)
        test_labels = np.array(current_workflow['has_failed'])
        train_labels = np.array(historical_data['has_failed'])

        random_forest(train_features, test_features, train_labels, test_labels)
        # adaptive_random_forest(features, labels, feature_headers, arf_filename)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("No historical data available. Exiting...")

    return historical_data, current_workflow
