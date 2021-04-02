import pandas as pd
import sys


def is_error_under_threshold(rf_type, label_rows, label_results):
    mean_label_value = sum(abs(number) for number in label_rows) / len(label_rows)
    if (rf_type == 'classifier' and label_results['error'] < 0.2) or (
            rf_type == 'regressor' and label_results['error'] < mean_label_value * 0.2):
        return True
    return True


def get_correct_prediction_features(low_error, result):
    features = []
    if low_error:
        for feature, importance in result['features_importance']:
            if importance > 0.1:
                features.append((feature, importance))
    return features


def undummify(df, prefix_sep):
    prefix_columns = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, prefix in prefix_columns.items():
        if prefix:
            undummified = (
                df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
            )
            series_list.insert(0, undummified)
        else:
            series_list.insert(0, df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def analyse(labels_results, input_execution_data, rf_labels, column_value_separator):
    # data = undummify(input_execution_data)
    label_features = {}
    problematic_cells = []
    for rf_type, rf_result in labels_results.items():
        # get the label corresponding to classifier or regressor
        labels = rf_labels[rf_type]
        for label in labels:
            try:
                label_results = rf_result[label]
                # find position of label column to put "[label] prediction" column next to it
                idx = input_execution_data.columns.get_loc(label) + 1
                input_execution_data.insert(loc=idx, column="predicted {}".format(label),
                                            value=label_results['predictions'])
                # get all rows of label to calculate threshold
                label_rows = input_execution_data[label].to_list()
                low_error = is_error_under_threshold(rf_type, label_rows, label_results)
                # if error is too high, the features are not returned
                features = get_correct_prediction_features(low_error, label_results)
                # problematic_cells += get_problematic_cells(data, features, label)
                label_features[label] = features
            except KeyError as e:
                sys.stderr.write(str(e) + "\n")
                raise KeyError
    return undummify(input_execution_data, column_value_separator), label_features
