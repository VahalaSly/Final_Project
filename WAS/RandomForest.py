import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle


def save_adaptive_classifier(filename, arfc_instance):
    file_object = open(filename, 'wb')
    pickle.dump(arfc_instance, file_object)


def get_labels_and_features(dataframe):
    labels = np.array(dataframe['is_executed'])
    features = dataframe.drop('is_executed', axis=1)
    feature_columns = list(features.columns)
    features = np.array(features)
    return labels, features, feature_columns


def get_numerical_feature_importance(rf, feature_columns):
    importance_list = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 2)) for feature, importance in
                          zip(feature_columns, importance_list)]
    # Sort the feature importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return importance_list, feature_importance


def show_graph(importance, feature_columns):
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importance)))
    # Make a bar chart
    plt.bar(x_values, importance, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_columns, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.show()


def main():
    arf_filename = 'adaptive_classifier.txt'
    data_frame = pd.read_csv('csvs/summary.csv')
    # hot-encode categorical columns
    numeric_dataset = pd.get_dummies(data_frame,
                                     columns=["parent_workflow", "class_name"])

    labels, features, feature_columns = get_labels_and_features(numeric_dataset)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    # get absolute error
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    importance_list, features_importance = get_numerical_feature_importance(rf, feature_columns)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in features_importance]

    show_graph(importance_list, feature_columns)

    # try:
    #     filehandler = open(arf_filename, 'rb')
    #     arf = pickle.load(filehandler)
    # except (FileNotFoundError, EOFError):
    #     arf = AdaptiveRandomForestClassifier()
    # arf.predict(numeric_dataset)
    # save_adaptive_classifier(arf_filename, arf)
    # print('Adaptive Random Forest ensemble classifier example')
    # print(arf.get_info())


def example_adaptive():
    # Imports
    from skmultiflow.data import RegressionGenerator
    from skmultiflow.meta import AdaptiveRandomForestRegressor
    import numpy as np
    # Setup a data stream
    stream = RegressionGenerator(random_state=1, n_samples=200)
    # Prepare stream for use
    # Setup the Adaptive Random Forest regressor
    arf_reg = AdaptiveRandomForestRegressor(random_state=123456)
    # Auxiliary variables to control loop and track performance
    n_samples = 0
    max_samples = 200
    y_pred = np.zeros(max_samples)
    y_true = np.zeros(max_samples)
    # Run test-then-train loop for max_samples and while there is data
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        y_true[n_samples] = y[0]
        y_pred[n_samples] = arf_reg.predict(X)[0]
        arf_reg.partial_fit(X, y)
        n_samples += 1
    # Display results
    print('Adaptive Random Forest regressor example')
    print('{} samples analyzed.'.format(n_samples))
    print('Mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))


if __name__ == "__main__":
    main()
