import matplotlib.pyplot as plt
from datetime import datetime
import jinja2
import pandas as pd


def colourise_cells(dataframe, cell_index_column):
    df1 = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
    for index, column in cell_index_column:
        df1.loc[index, column] = 'background-color: yellow'
    return df1


def create_html(task_figures, workflow_figures, task_dataframe, workflow_dataframe,
                report_path):
    timestamp = datetime.now()

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template = env.get_template('report/report_template.html')

    html = template.render(timestamp=timestamp, wk_dataframe=workflow_dataframe.render(),
                           tk_dataframe=task_dataframe.render(), wk_images=workflow_figures, tk_images=task_figures)

    timestamp = timestamp.strftime("%d%m%y%H%M%S")
    # Write the HTML file
    html_file = "{}/execution_report_{}.html".format(report_path, timestamp)
    with open(html_file, 'w') as f:
        f.write(html)
    return html_file


def produce_report(task_features, workflow_features, task_dataset, workflow_dataset,
                   tasks_problematic_cells, workflows_problematic_cells, report_path):
    task_figures = []
    workflow_figures = []
    # task graphs
    for label, features in task_features.items():
        figure_name = "report/figures/task_{}.png".format(label)
        make_graph(label, features, figure_name)
        task_figures.append(figure_name)
    # workflow graphs
    for label, features in workflow_features.items():
        figure_name = "report/figures/workflow_{}.png".format(label)
        make_graph(label, features, figure_name)
        workflow_figures.append(figure_name)

    task_dataset = task_dataset.style.apply(colourise_cells, cell_index_column=tasks_problematic_cells,
                                            subset=None, axis=None)
    workflow_dataset = workflow_dataset.style.apply(colourise_cells, cell_index_column=workflows_problematic_cells,
                                                    subset=None, axis=None)

    return create_html(task_figures, workflow_figures, task_dataset, workflow_dataset,
                       report_path)


def make_graph(label, data_pairs, figure_name):
    # make a square figure and axes
    plt.figure(1, figsize=(12, 6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])

    # The slices will be ordered and plotted counter-clockwise.
    labels = []
    fracs = []
    other_percentage = 100
    for pair in data_pairs:
        labels.append(pair[0])
        percentage = pair[1] * 100
        fracs.append(percentage)
        other_percentage -= percentage
    if other_percentage > 0:
        labels.append("Unknown")
        fracs.append(other_percentage)

    plt.pie(fracs, labels=labels,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.title("Which features affect the {}?".format(label), bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig("{}".format(figure_name))  # save the figure to file
    plt.close()
