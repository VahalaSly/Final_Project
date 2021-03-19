from argparse import ArgumentParser
from WAS import DataPreProcessing
from WAS import RandomForest


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filepath",
                        help="KNIME workflow summary in xml format", metavar="FILE PATH",
                        required=True)
    return parser.parse_args()


def main():
    xml_file_path = get_arguments().filepath
    csv_data_path = 'csvs/latest_execution.csv'
    historical_data_path = 'csvs/historical_data.csv'
    is_pp_successful = DataPreProcessing.main(xml_file_path, csv_data_path)
    if is_pp_successful:
        RandomForest.main(historical_data_path, csv_data_path)


if __name__ == "__main__":
    main()
