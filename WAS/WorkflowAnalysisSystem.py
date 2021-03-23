from argparse import ArgumentParser
from WAS import DataPreProcessing
from WAS import RandomForest
import pandas as pd


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filepath",
                        help="KNIME workflow summary in xml format", metavar="FILE PATH",
                        required=True)
    return parser.parse_args()


def add_latest_exec_to_historical_data(historical_data_path, historical_data, latest_execution):
    if historical_data is not False:
        # pd.concat creates a union of the two dataframes, adding any new column
        latest_execution = pd.concat([historical_data, latest_execution], ignore_index=True, sort=False)
        # make sure to fill any NAN cell with 0
        latest_execution.fillna(0, inplace=True)
    latest_execution.to_csv(historical_data_path, index=False)


def main():
    xml_file_path = get_arguments().filepath
    tasks_csv_data_path = 'csvs/current_workflow_tasks.csv'
    workflows_csv_data_path = 'csvs/current_workflow.csv'
    historical_data_path = 'csvs/historical_data.csv'
    is_dpp_successful = DataPreProcessing.main(xml_file_path, tasks_csv_data_path, workflows_csv_data_path)
    if is_dpp_successful:
        try:
            historical_data, latest_execution = \
                RandomForest.main(historical_data_path, tasks_csv_data_path)
            add_latest_exec_to_historical_data(historical_data_path, historical_data, latest_execution)
        except ValueError as e:
            print("Error encountered:")
            print(e)
            print("Exiting...")


if __name__ == "__main__":
    main()
