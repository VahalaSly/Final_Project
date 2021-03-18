import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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


def random_forest(features, labels, feature_headers):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    # get absolute error
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    importance_list, features_importance = get_numerical_feature_importance(rf, feature_headers)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in features_importance]

    show_graph(importance_list, feature_headers)


def get_adaptive_instance(filename):
    try:
        filehandler = open(filename, 'rb')
        arf = pickle.load(filehandler)
    except (FileNotFoundError, EOFError):
        arf = AdaptiveRandomForestRegressor(n_estimators=1000, random_state=42)
    return arf


def adaptive_random_forest(features, labels, feature_headers, filename):
    # Imports
    import numpy as np
    arf = get_adaptive_instance(filename)
    arf.fit(features, labels)
    predictions = arf.predict(features)
    errors = abs(predictions - labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    importance_list, features_importance = get_numerical_feature_importance(arf, feature_headers)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in features_importance]


def main():
    arf_filename = 'adaptive_classifier.txt'
    data_frame = pd.read_csv('csvs/summary.csv')
    # hot-encode categorical columns
    numeric_dataset = pd.get_dummies(data_frame,
                                     columns=["parent_workflow", "class_name"])

    labels = np.array(numeric_dataset['is_executed'])
    features = numeric_dataset.drop('is_executed', axis=1)
    feature_headers = list(features.columns)
    features = np.array(features)

    random_forest(features, labels, feature_headers)
    # adaptive_random_forest(features, labels, feature_headers, arf_filename)


if __name__ == "__main__":
    main()
