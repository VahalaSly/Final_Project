from argparse import ArgumentParser
from WAS import DataPreProcessing
from WAS import RandomForest
from WAS import FeedbackSuite
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


def prepare_features(results, instances, labels):
    dataframe = pd.DataFrame.from_records([instance.to_dict() for instance in instances])
    label_features = {}
    for rf_type, label in labels.items():
        features = []
        label_result = results[rf_type]
        error = label_result['error']
        if not (rf_type == 'classifier' and error > 0.2):
            for feature, importance in label_result['features_importance']:
                if importance > 0.02:
                    features.append((feature, importance))
            label_features[label] = features
        dataframe["{} prediction".format(label)] = label_result['predictions']
    return dataframe, label_features


def main():
    report_path = '{}/../reports'.format(absolute_path)
    # tasks variables
    task_historical_data_path = 'csvs/tasks_historical_data.csv'
    task_labels = {'classifier': 'failure', 'regressor': 'execution duration'}
    # workflows variables
    workflow_historical_data_path = 'csvs/workflows_historical_data.csv'
    workflow_labels = {'classifier': 'failure', 'regressor': 'makespan'}

    tasks_historical_data = None
    workflow_historical_data = None

    tasks, workflows = DataPreProcessing.main(xml_file_path)
    if not workflows is False and not tasks is False:
        hotenc_task_data = pd.get_dummies(
            pd.DataFrame.from_records([task.to_ml_ready_dict() for task in tasks]).fillna(0))
        hotenc_workflow_data = pd.get_dummies(
            pd.DataFrame.from_records([workflow.to_ml_ready_dict() for workflow in workflows]).fillna(0))
        try:
            tasks_historical_data = pd.read_csv(task_historical_data_path)
            workflow_historical_data = pd.read_csv(workflow_historical_data_path)
            # execute Random Forest
            tasks_results = RandomForest.predict(tasks_historical_data, hotenc_task_data, task_labels)
            workflows_results = RandomForest.predict(workflow_historical_data, hotenc_workflow_data, workflow_labels)
            # analyse RF results
            new_task_dataframe, task_features = prepare_features(tasks_results,
                                                                 tasks, task_labels)
            new_workflow_dataframe, workflow_features = prepare_features(workflows_results,
                                                                         workflows, workflow_labels)
            report = FeedbackSuite.produce_report(task_features, workflow_features, new_task_dataframe,
                                                  new_workflow_dataframe, report_path)
            print("Workflow Analysis Finished!")
            print("Report file:///{} has been saved.".format(report.replace('\\', '/')))

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
    absolute_path = pathlib.Path(__file__).parent.absolute()
    xml_file_path = get_arguments().filepath
    main()
