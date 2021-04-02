from argparse import ArgumentParser
import pandas as pd
import pathlib
import os
import sys
import ProcessInputJson
import RandomForest
import FeedbackSuite
import AnalyseResults


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filepath",
                        help="KNIME workflow summary in xml format", metavar="FILE PATH",
                        required=True)
    parser.add_argument("-ct", "--classify_task", dest="task_classifier",
                        help="List of node/task targets Column:Value pair to be classified", metavar="COLUMN:VALUE",
                        required=False, nargs="*", default='')
    parser.add_argument("-rt", "--regress_task", dest="task_regressor",
                        help="List of node/task targets continuous numeric column to be regressed", metavar="COLUMN",
                        required=False, nargs="*", default='')
    parser.add_argument("-cw", "--classify_workflow", dest="wk_classifier",
                        help="List of workflow targets Column:Value pair to be classified", metavar="COLUMN:VALUE",
                        required=False, nargs="*", default='')
    parser.add_argument("-rw", "--regress_workflow", dest="wk_regressor",
                        help="List of workflow targets continuous numeric column to be regressed", metavar="COLUMN",
                        required=False, nargs="*", default='')
    return parser.parse_args()


def add_latest_exec_to_historical_data(historical_data_path, historical_data, latest_execution):
    try:
        if historical_data is not None:
            # pd.concat creates a union of the two dataframes, adding any new column
            latest_execution = pd.concat([historical_data, latest_execution], ignore_index=True, sort=False)
            # make sure to fill any NAN cell with 0
            latest_execution.fillna(0, inplace=True)
        latest_execution.to_csv(historical_data_path, index=False)
    except ValueError as e:
        print("Encountered error while trying to save new data to historical data.")
        print(e)
        print("Exiting...")


def main():
    report_path = '{}/../reports'.format(absolute_path)
    # tasks variables
    task_historical_data_path = 'csvs/tasks_historical_data.csv'
    # workflows variables
    workflow_historical_data_path = 'csvs/workflows_historical_data.csv'

    tasks_historical_data = None
    workflow_historical_data = None
    try:
        try:
            print("Initialising data pre-processing step...")
            tasks, workflows = ProcessInputJson.main(xml_file_path)
            print("Data pre-processing step successful! \n")
        except KeyError as e:
            sys.stderr.write("Data pre-processing step unsuccessful :( \n")
            sys.stderr.write(str(e))
            raise KeyError

        if workflows is not None and tasks is not None:
            hotenc_task_data = pd.get_dummies(pd.DataFrame.from_records(tasks),
                                              prefix_sep=column_value_separator).fillna(0)
            hotenc_workflow_data = pd.get_dummies(pd.DataFrame.from_records(workflows),
                                                  prefix_sep=column_value_separator).fillna(0)
            if os.path.isfile(task_historical_data_path) and os.path.isfile(workflow_historical_data_path):
                tasks_historical_data = pd.read_csv(task_historical_data_path)
                workflow_historical_data = pd.read_csv(workflow_historical_data_path)

                try:
                    print("Initialising Random Forest step...")
                    tasks_results = RandomForest.predict(tasks_historical_data,
                                                         hotenc_task_data,
                                                         task_rf_label_map)
                    workflow_results = RandomForest.predict(workflow_historical_data,
                                                            hotenc_workflow_data,
                                                            workflow_rf_label_map)
                    print("Random Forest step successful! \n")
                except KeyError as e:
                    sys.stderr.write("Random Forest step unsuccessful :( \n")
                    sys.stderr.write(str(e))
                    raise KeyError

                try:
                    print("Initialising results analysis step...")
                    new_task_dataframe, task_features = AnalyseResults.analyse(tasks_results,
                                                                               hotenc_task_data,
                                                                               task_rf_label_map,
                                                                               column_value_separator)
                    new_workflow_dataframe, workflow_features = AnalyseResults.analyse(workflow_results,
                                                                                       hotenc_workflow_data,
                                                                                       workflow_rf_label_map,
                                                                                       column_value_separator)
                    print("Results analysis step successful! \n")
                except KeyError as e:
                    sys.stderr.write("Results analysis step unsuccessful :( \n")
                    sys.stderr.write(str(e))
                    raise KeyError

                try:
                    print("Initialising feedback report step...")
                    report = FeedbackSuite.produce_report(task_features,
                                                          workflow_features,
                                                          new_task_dataframe,
                                                          new_workflow_dataframe,
                                                          report_path)
                    if report is not None:
                        print("Feedback report step successful! \n")
                        print("Report file:///{} has been saved. \n".format(report.replace('\\', '/')))
                        print("Workflow Analysis Finished!")
                    else:
                        print("Could not produce report. Check for errors and try again.")
                except ValueError as e:
                    sys.stderr.write("Results analysis step unsuccessful :( \n")
                    sys.stderr.write(str(e))
                    raise ValueError
            else:
                print("No historical data available.")
                print("Saving new workflow execution data to historical data...")
                add_latest_exec_to_historical_data(task_historical_data_path,
                                                   tasks_historical_data, hotenc_task_data)
                add_latest_exec_to_historical_data(workflow_historical_data_path,
                                                   workflow_historical_data, hotenc_workflow_data)
    except (ValueError, KeyError):
        pass
    print("Exiting...")


if __name__ == "__main__":
    absolute_path = pathlib.Path(__file__).parent.absolute()
    xml_file_path = get_arguments().filepath
    column_value_separator = "!-->"

    task_rf_label_map = {'classifier': [], 'regressor': []}
    workflow_rf_label_map = {'classifier': [], 'regressor': []}

    task_classifier_values = get_arguments().task_classifier
    task_regressor_values = get_arguments().task_regressor
    wk_classifier_values = get_arguments().wk_classifier
    wk_regressor_values = get_arguments().wk_regressor

    classifier_values = [task_classifier_values, wk_classifier_values]

    # replace the ":" with the more unique separator to avoid confusion when unencoding the values at the end
    for value in task_classifier_values:
        try:
            value_pair = value.split(":")
            task_rf_label_map['classifier'].append(value_pair[0] + column_value_separator + value_pair[1])
        except IndexError:
            sys.stderr.write("Classifier value: '{}' not formatted properly. Expected format: COLUMN:VALUE")
            raise IndexError
    for value in wk_classifier_values:
        try:
            value_pair = value.split(":")
            workflow_rf_label_map['classifier'].append(value_pair[0] + column_value_separator + value_pair[1])
        except IndexError:
            sys.stderr.write("Classifier value: '{}' not formatted properly. Expected format: COLUMN:VALUE")
            raise IndexError
    for value in task_regressor_values:
        task_rf_label_map['regressor'].append(value)
    for value in wk_regressor_values:
        workflow_rf_label_map['regressor'].append(value)

    print("Welcome to the Workflow Analysis System!")
    print("Starting...")
    main()
