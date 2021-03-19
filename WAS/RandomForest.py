import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultiflow.meta import AdaptiveRandomForestRegressor
import matplotlib.pyplot as plt
import pickle


def save_adaptive_classifier(filename, arfc_instance):
    file_object = open(filename, 'wb')
    pickle.dump(arfc_instance, file_object)


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
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    feature_headers = list(test_features.columns)
    # Train the model on training data
    try:
        rf.fit(train_features, train_labels)
        predictions = rf.predict(test_features)

        # get absolute error
        errors = abs(predictions - test_labels)
        print('Mean Absolute Error:', round(np.mean(errors), 2))

        importance_list, features_importance = get_numerical_feature_importance(rf, feature_headers)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in features_importance]

        print(predictions)
        show_graph(importance_list, feature_headers)
    except ValueError as e:
        print("Error encountered:")
        print(e)
        print("Exiting...")


def get_adaptive_instance(filename):
    try:
        filehandler = open(filename, 'rb')
        arf = pickle.load(filehandler)
    except (FileNotFoundError, EOFError):
        arf = AdaptiveRandomForestRegressor(n_estimators=1000, random_state=42)
    return arf


def main(historical_data_path, latest_execution_path):
    arf_filename = 'adaptive_classifier.txt'
    latest_execution = pd.get_dummies(pd.read_csv(latest_execution_path),
                                      columns=["parent_workflow", "class_name"])
    historical_data = False
    try:
        historical_data = pd.read_csv(historical_data_path)
        test_labels = np.array(latest_execution['is_executed'])
        test_features = latest_execution.drop('is_executed', axis=1)
        train_labels = np.array(historical_data['is_executed'])
        train_features = historical_data.drop('is_executed', axis=1)

        random_forest(train_features, test_features, train_labels, test_labels)
        # adaptive_random_forest(features, labels, feature_headers, arf_filename)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        is_first_workflow = True
        print("No historical data available. Exiting...")

    return historical_data, latest_execution
