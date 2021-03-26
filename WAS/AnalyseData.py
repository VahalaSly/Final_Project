from tabulate import tabulate
import pandas as pd


def analyse(labels_results, data, labels):
    data = undummify(data)
    label_features = {}
    problematic_cells = []
    for rf_type, rf_result in labels_results.items():
        # get the label corresponding to classifier or regressor
        label = labels[rf_type]
        # find position of label column to put "[label] prediction" column next to it
        idx = data.columns.get_loc(label) + 1
        data.insert(loc=idx, column="{} prediction".format(label), value=rf_result['predictions'])
        # get all rows of label to calculate threshold
        label_rows = data[label].to_list()
        low_error = is_error_under_threshold(rf_type, label_rows, rf_result)
        # if error is too high, the features are not returned
        features = get_correct_prediction_features(low_error, rf_result)
        problematic_cells += get_problematic_cells(data, features, label)
        label_features[label] = features
    return data, label_features, problematic_cells

    # print(tabulate(data, headers='keys', tablefmt='psql'))


def get_problematic_cells(dataframe, features, label):
    cells = []
    for index in dataframe.index:
        prediction_column = dataframe.iloc[index]["{} prediction".format(label)]
        failure_issue = prediction_column == 1
        exec_dur_issue = dataframe.iloc[index][label] > (prediction_column * 1.5)
        tasks_sec_isse = (dataframe.iloc[index][label] * 1.5) < prediction_column
        if (label == 'failure' and failure_issue)\
                or (label == 'execution duration (ms)' and exec_dur_issue)\
                or (label == 'tasks per second' and tasks_sec_isse):
            for feature, importance in features:
                if "_" in feature:
                    split_feat = feature.split("_", 1)
                    column = split_feat[0]
                    value = split_feat[1]
                    if dataframe.iloc[index][column] == value:
                        cells.append((index, column))
                else:
                    cells.append((index, feature))
    return cells


def is_error_under_threshold(rf_type, label_rows, rf_result):
    mean_label_value = sum(abs(number) for number in label_rows) / len(label_rows)
    if (rf_type == 'classifier' and rf_result['error'] < 0.2) or (
            rf_type == 'regressor' and rf_result['error'] < mean_label_value * 0.2):
        return True
    return False


def get_correct_prediction_features(low_error, result):
    features = []
    if low_error:
        for feature, importance in result['features_importance']:
            if importance > 0.1:
                features.append((feature, importance))
    return features


def undummify(df, prefix_sep="_"):
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
