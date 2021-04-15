import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import sys


def colourise_cells(dataframe, feature_columns):
    df1 = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
    for index in dataframe.index:
        for feature, importance in feature_columns:
            if "!-->" in feature:
                split_feat = feature.split("!-->", 1)
                column = split_feat[0]
                value = split_feat[1]
                if dataframe.loc[index, column] == value:
                    df1.loc[index, column] = 'background-color:yellow'
            else:
                df1.loc[index, feature] = 'background-color:yellow'
    return df1


def create_xlsx(workflow_figures, task_figures, workflow_dataframe, task_dataframe,
                workflow_features, task_features, branch_stats, task_stats, report_filepath):
    # these values are used to position the figures
    workflow_row = workflow_dataframe.shape[0]
    workflow_col = workflow_dataframe.shape[1]
    task_row = task_dataframe.shape[0]
    task_col = task_dataframe.shape[1]

    # add columns/cells highlight based on features
    workflow_styler = workflow_dataframe.style.apply(colourise_cells,
                                                     feature_columns=workflow_features,
                                                     subset=None, axis=None)
    task_styler = task_dataframe.style.apply(colourise_cells, feature_columns=task_features,
                                             subset=None, axis=None)

    writer = pd.ExcelWriter(report_filepath, engine='xlsxwriter')
    workflow_styler.to_excel(writer, sheet_name='Workflow Analysis')
    task_styler.to_excel(writer, sheet_name='Task Analysis')
    branch_stats.to_excel(writer, sheet_name='Branch Topological Analysis')
    task_stats.to_excel(writer, sheet_name='Task Topological Analysis')

    # add figures
    col = 0
    add_figures_to_sheet(writer.sheets['Workflow Analysis'], workflow_row, workflow_col, workflow_figures)
    add_figures_to_sheet(writer.sheets['Task Analysis'], task_row, task_col, task_figures)

    writer.save()


def add_figures_to_sheet(sheet, max_row, max_col, figures):
    # max_row + 3 represents the row after the table data + 3 rows of space
    row = max_row + 3
    col = 0
    for figure in figures:

        if col >= max_col:
            col = 0
            row += 14
        sheet.insert_image(row=row, col=col, filename=figure,
                           options={'x_scale': 0.7, 'y_scale': 0.7})
        col += 12


def produce_report(task_features, workflow_features, task_dataset, workflow_dataset, branch_stats, task_stats, report_path):
    task_figures = []
    workflow_figures = []
    # task graphs
    i = 0
    for label, features in task_features.items():
        figure_name = "report/figures/task_{}.png".format(i)
        make_graph(label, features, figure_name)
        task_figures.append(figure_name)
        i += 1
    # workflow graphs
    i = 0
    for label, features in workflow_features.items():
        figure_name = "report/figures/workflow_{}.png".format(i)
        make_graph(label, features, figure_name)
        workflow_figures.append(figure_name)
        i += 1

    workflow_features_list = list(set().union(*workflow_features.values()))
    task_features_list = list(set().union(*task_features.values()))

    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d%m%y%H%M%S")
    xlsx_file = "{}/execution_report_{}.xlsx".format(report_path, timestamp)

    try:
        create_xlsx(workflow_figures, task_figures, workflow_dataset, task_dataset,
                    workflow_features_list, task_features_list, branch_stats, task_stats, xlsx_file)
        return xlsx_file
    except TypeError as e:
        sys.stderr.write(str(e) + "\n")
        raise TypeError


def make_graph(label, data_pairs, figure_name):
    # make a square figure and axes
    plt.figure(1, figsize=(10, 4))
    patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
            'hatch.color': 'darkslategrey'
        }
    )

    # prepare labels and fractions
    features = []
    importances = []
    other_percentage = 100
    for pair in data_pairs:
        feature_name = pair[0]
        # cut off names that are too long
        max_length = 100
        half_length = 50
        if len(feature_name) > half_length:
            if len(feature_name) > max_length:
                # cut off names longer than 100 chars
                feature_name = feature_name[:max_length] + "[...]"
            # go on a new line for names longer than 50 chars
            feature_name = feature_name[:half_length] + "-\n" + feature_name[half_length:]
        features.append(feature_name)
        percentage = pair[1] * 100
        importances.append(percentage)
        other_percentage -= percentage
    if other_percentage > 0:
        features.append("Unknown")
        importances.append(other_percentage)

    pie = plt.pie(importances, labels=features,
                  autopct='%1.1f%%', startangle=90)
    # add textures to pie chart wedges for colour clarity
    for i in range(len(pie[0])):
        pie[0][i].set_hatch(patterns[i % len(patterns)])

    plt.title("Which features affect the {}?".format(label), bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig("{}".format(figure_name))
    plt.close()
