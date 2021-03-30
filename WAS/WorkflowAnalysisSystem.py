from argparse import ArgumentParser
import DataPreProcessing
import RandomForest
import FeedbackSuite
import AnalyseResults
import pandas as pd
import pathlib


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filepath",
                        help="KNIME workflow summary in xml format", metavar="FILE PATH",
                        required=True)
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
    task_rf_label_map = {'classifier': ['failure', 'warnings'], 'regressor': ['execution duration (ms)', 'execution weekday']}
    # workflows variables
    workflow_historical_data_path = 'csvs/workflows_historical_data.csv'
    workflow_rf_labels_map = {'classifier': ['failure'], 'regressor': ['tasks per second']}

    tasks_historical_data = None
    workflow_historical_data = None

    tasks, workflows = DataPreProcessing.main(xml_file_path)
    if workflows is not None and tasks is not None:
        try:
            tasks_historical_data = pd.read_csv(task_historical_data_path)
            workflow_historical_data = pd.read_csv(workflow_historical_data_path)
            # execute Random Forest
            try:
                tasks_results = RandomForest.predict(tasks_historical_data, tasks, task_rf_label_map)
                workflows_results = RandomForest.predict(workflow_historical_data, workflows,
                                                         workflow_rf_labels_map)
                # analyse RF results
                new_task_dataframe, task_features = AnalyseResults.analyse(
                    tasks_results, tasks, task_rf_label_map)
                new_workflow_dataframe, workflow_features = AnalyseResults.analyse(
                    workflows_results, workflows, workflow_rf_labels_map)
                report = FeedbackSuite.produce_report(task_features, workflow_features, new_task_dataframe,
                                                      new_workflow_dataframe, report_path)
                if report is not None:
                    print("Workflow Analysis Finished!")
                    print("Report file:///{} has been saved.".format(report.replace('\\', '/')))
                else:
                    print("Could not produce report. Check the errors and try again.")
            except KeyError:
                print("Could not train algorithm, some training labels were missing from historical data.")

        except (pd.errors.EmptyDataError, FileNotFoundError):
            print("No historical data available.")
        except ValueError as e:
            print("Error encountered:")
            print(e)

        print("Saving workflow execution data to historical data...")
        add_latest_exec_to_historical_data(task_historical_data_path,
                                           tasks_historical_data, tasks)
        add_latest_exec_to_historical_data(workflow_historical_data_path,
                                           workflow_historical_data, workflows)
        print("Exiting...")


if __name__ == "__main__":
    absolute_path = pathlib.Path(__file__).parent.absolute()
    xml_file_path = get_arguments().filepath
    print("Initialising Workflow Analysis System...")
    main()
