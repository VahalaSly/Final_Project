import sys


def is_error_under_threshold(rf_type, label_rows, label_results):
    print("Error = {}".format(label_results['error']))
    if rf_type == 'classifier' and label_results['error'] < 0.2:
        return True
    elif rf_type == 'regressor':
        mean_label_value = sum(abs(number) for number in label_rows) / len(label_rows)
        if label_results['error'] < mean_label_value * 0.2:
            return True
    return False


def get_correct_prediction_features(low_error, result):
    features = []
    if low_error:
        for feature, importance in result['features_importance']:
            if importance > 0.05:
                features.append((feature, importance))
    return features


def analyse(labels_results, execution_data, rf_labels):
    label_features = {}
    for rf_type, rf_result in labels_results.items():
        # get the label corresponding to classifier or regressor
        labels = rf_labels[rf_type]
        for label in labels:
            try:
                label_results = rf_result[label]
                # find position of label column to put "[label] prediction" column next to it
                idx = execution_data.columns.get_loc(label) + 1
                execution_data.insert(loc=idx, column="predicted {}".format(label),
                                            value=label_results['predictions'])
                # get all rows of label to calculate error threshold
                print("Label: {}".format(label))
                label_rows = execution_data[label].to_list()
                low_error = is_error_under_threshold(rf_type, label_rows, label_results)
                # if error is too high, the features are not returned
                features = get_correct_prediction_features(low_error, label_results)
                label_features[label] = features
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return execution_data, label_features
