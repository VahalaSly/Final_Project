from argparse import ArgumentParser
import pandas as pd
import pathlib
import os
import sys
import data_processing.ProcessInputJson as PI
import data_processing.FeedbackSuite as FS
import analysis.RandomForest as RF
import analysis.AnalyseResults as AR
import analysis.TopologicalAnalysis as TA


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filepath",
                        help="KNIME workflow summary in JSON format", metavar="FILE PATH",
                        required=True)
    parser.add_argument("-ct", "--classify_task", dest="task_classifier",
                        help="Target node/task non-continuous KEY to be classified", metavar="KEY",
                        required=False, nargs="*", default='')
    parser.add_argument("-rt", "--regress_task", dest="task_regressor",
                        help="Target node/task continuous numeric KEY to be regressed", metavar="KEY",
                        required=False, nargs="*", default='')
    parser.add_argument("-cw", "--classify_workflow", dest="wk_classifier",
                        help="Target workflow non-continuous KEY to be classified", metavar="KEY",
                        required=False, nargs="*", default='')
    parser.add_argument("-rw", "--regress_workflow", dest="wk_regressor",
                        help="Target workflow continuous numeric KEY to be regressed", metavar="KEY",
                        required=False, nargs="*", default='')
    return parser.parse_args()


def add_latest_exec_to_historical_data(historical_data_path, historical_data, latest_execution):
    try:
        if historical_data is not None:
            # pd.concat creates a union of the two dataframes, adding any new column
            latest_execution = pd.concat([historical_data, latest_execution], ignore_index=True, sort=False)
        latest_execution.to_csv(historical_data_path, index=False)
    except ValueError as e:
        print("Encountered error while trying to save new data to historical data.")
        print(e)


def analyse(report_path,
            json_file_path,
            task_rf_label_map,
            workflow_rf_label_map):
    tasks_historical_data = None
    workflow_historical_data = None
    try:
        ## STEP 1 ##
        try:
            print("Initialising data pre-processing step...")
            tasks, workflows = PI.json_to_dataframe(json_file_path)
            print("Data pre-processing step successful! \n")
        except (KeyError, ValueError) as e:
            sys.stderr.write("Data pre-processing step unsuccessful :( \n")
            sys.stderr.write(str(e))
            raise e

        if os.path.isfile(task_historical_data_path) and os.path.isfile(workflow_historical_data_path):
            tasks_historical_data = pd.read_csv(task_historical_data_path, low_memory=False)
            workflow_historical_data = pd.read_csv(workflow_historical_data_path, low_memory=False)

            ### STEP 2 ###
            try:
                print("Initialising Random Forest step...")
                tasks_results = RF.predict(tasks_historical_data,
                                           tasks,
                                           task_rf_label_map)
                workflow_results = RF.predict(workflow_historical_data,
                                              workflows,
                                              workflow_rf_label_map)
                print("Random Forest step successful! \n")
            except (KeyError, ValueError) as e:
                sys.stderr.write("Random Forest step unsuccessful :( \n")
                raise e

            ### STEP 3 ##
            try:
                print("Initialising results analysis step...")
                new_task_dataframe, task_imp_features = AR.analyse(tasks_results,
                                                                   tasks,
                                                                   task_rf_label_map)
                new_workflow_dataframe, workflow_imp_features = AR.analyse(workflow_results,
                                                                           workflows,
                                                                           workflow_rf_label_map)
                print("Results analysis step successful! \n")
            except (KeyError, ValueError) as e:
                sys.stderr.write("Results analysis step unsuccessful :( \n")
                raise e

            ### STEP 4 ##
            try:
                print("Initialising topological analysis step...")
                paths_stats = TA.analyse(tasks_historical_data,
                                         new_task_dataframe,
                                         task_imp_features,
                                         task_rf_label_map)
                print("Topological analysis step successful! \n")
            except (KeyError, ValueError) as e:
                sys.stderr.write("Topological analysis step unsuccessful :( \n")
                raise e

            ## STEP 5 ##
            try:
                print("Initialising feedback report step...")
                report = FS.produce_report(task_imp_features,
                                           workflow_imp_features,
                                           new_task_dataframe,
                                           new_workflow_dataframe,
                                           paths_stats,
                                           report_path)
                print("Feedback report step successful! \n")
                print("Report file:///{} has been saved. \n".format(report.replace('\\', '/')))
                print("Workflow Analysis Finished!")
            except (KeyError, ValueError) as e:
                sys.stderr.write("Results analysis step unsuccessful :( \n")
                raise e
        else:
            print("No historical data available.")
        print("Saving new workflow execution data to historical data...")

        ### STEP 6 ###
        add_latest_exec_to_historical_data(task_historical_data_path,
                                           tasks_historical_data, tasks)
        add_latest_exec_to_historical_data(workflow_historical_data_path,
                                           workflow_historical_data, workflows)
    except (ValueError, KeyError) as e:
        print(e)


def main():
    absolute_path = pathlib.Path(__file__).parent.absolute()
    json_file_path = "{}/{}".format(absolute_path, get_arguments().filepath)
    report_path = '{}/../reports'.format(absolute_path)

    task_rf_label_map = {'classifier': [], 'regressor': []}
    workflow_rf_label_map = {'classifier': [], 'regressor': []}

    task_rf_label_map['classifier'] = [arg for arg in get_arguments().task_classifier]
    task_rf_label_map['regressor'] = [arg for arg in get_arguments().task_regressor]
    workflow_rf_label_map['classifier'] = [arg for arg in get_arguments().wk_classifier]
    workflow_rf_label_map['regressor'] = [arg for arg in get_arguments().wk_regressor]

    print("Welcome to the Workflow Analysis System!")
    print("Starting...")
    analyse(report_path,
            json_file_path,
            task_rf_label_map,
            workflow_rf_label_map)

    print("Exiting...")


if __name__ == "__main__":
    # tasks variables
    task_historical_data_path = 'csvs/tasks_historical_data.csv'
    # workflows variables
    workflow_historical_data_path = 'csvs/workflows_historical_data.csv'
    main()
