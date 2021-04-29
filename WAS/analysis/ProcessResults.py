import sys


def is_error_under_threshold(error):
    print("Error = {}".format(error))
    if error < 0.2:
        return True
    return False


def get_valid_prediction_features(is_error_low, features):
    imp_features = []
    if is_error_low:
        for feature, importance in features:
            if importance >= 0.1:
                imp_features.append((feature, importance))
    return imp_features


def process(rf_results, dataframe):
    label_features = {}
    for rf_type, rf_result in rf_results.items():
        for label_result in rf_result:
            try:
                label_name = label_result.label_name
                # find position of label column to put "[label] prediction" column next to it
                idx = dataframe.columns.get_loc(label_name) + 1
                dataframe.insert(loc=idx, column="predicted {}".format(label_name),
                                 value=label_result.predictions)
                # get all rows of label to calculate error threshold
                print("Label: {}".format(label_name))
                label_values = dataframe[label_name].to_list()
                is_error_low = is_error_under_threshold(label_result.error)
                # if error is too high, the features are not returned
                features = get_valid_prediction_features(is_error_low, label_result.features_importance)
                label_features[label_name] = features
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return dataframe, label_features
