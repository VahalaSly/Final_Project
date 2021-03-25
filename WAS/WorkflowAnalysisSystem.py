from argparse import ArgumentParser
from WAS import DataPreProcessing
from WAS import RandomForest
from WAS import FeedbackSuite
import pandas as pd


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
    xml_file_path = get_arguments().filepath
    # tasks variables
    task_historical_data_path = 'csvs/tasks_historical_data.csv'
    task_labels = {'classifier': 'has_failed', 'regressor': 'execution_duration'}
    # workflows variables
    workflow_historical_data_path = 'csvs/workflows_historical_data.csv'
    workflow_labels = {'classifier': 'has_failed', 'regressor': 'makespan'}

    tasks_historical_data = None
    workflow_historical_data = None

    dataframes = DataPreProcessing.main(xml_file_path)
    if dataframes:
        hotenc_task_data = pd.get_dummies(dataframes['task'])
        hotenc_workflow_data = pd.get_dummies(dataframes['workflow'])
        try:
            tasks_historical_data = pd.read_csv(task_historical_data_path)
            workflow_historical_data = pd.read_csv(workflow_historical_data_path)
            # execute Random Forest
            tasks_results = RandomForest.predict(tasks_historical_data, hotenc_task_data, task_labels)
            workflows_results = RandomForest.predict(workflow_historical_data, hotenc_workflow_data, workflow_labels)
            # get RF results' feedback
            task_feedback = FeedbackSuite.get_feedback(tasks_results, dataframes['task'], task_labels)
            workflow_feedback = FeedbackSuite.get_feedback(workflows_results, dataframes['workflow'], workflow_labels)
            FeedbackSuite.produce_report(task_feedback, workflow_feedback)

        except (pd.errors.EmptyDataError, FileNotFoundError):
            print("No historical data available. Exiting...")
        except ValueError as e:
            print("Error encountered:")
            print(e)
            print("Exiting...")

        add_latest_exec_to_historical_data(task_historical_data_path,
                                           tasks_historical_data, hotenc_task_data)
        add_latest_exec_to_historical_data(workflow_historical_data_path,
                                           workflow_historical_data, hotenc_workflow_data)


if __name__ == "__main__":
    main()
