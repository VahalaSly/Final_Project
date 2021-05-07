import pandas as pd
from analysis.util import Graph_operations as Go


def count_ratios_and_means(df, column):
    if df[column].dtype == 'object':
        count = df[column].value_counts(normalize=True)
        return ["{}, ratio: {} ".format(value, round(ratio, 2)) for
                value, ratio in count.items()]
    else:
        return round(df[column].mean(), 2)


def ids_to_names(ids, dataframe):
    named_tasks = []
    for task_id in ids:
        named_tasks.append(str(dataframe.loc[task_id, 'name']))
    return named_tasks


def get_workflow_branches(df):
    # the successors are divided among multiple columns
    # this finds all of these columns (through regex search)
    graph = {}
    all_successors = df.filter(regex='successors').filter(regex='id')
    if len(all_successors.columns) != 0:
        # puts all of the successors columns together and drops empty
        all_successors = all_successors.T.apply(lambda x: x.dropna().tolist())
        ids = list(df['id'])
        # goes through each task and its successors, and creates a graph in the form of {task:[successors]}
        graph = Go.create_graph(ids, all_successors)
    branches = []
    start_nodes, end_nodes = Go.find_start_end_of_branches(graph)
    for start_node in start_nodes:
        for end_node in end_nodes:
            if start_node != end_node:
                found_paths = Go.find_all_paths(graph, start_node, end_node)
                # if there's more than one path (branch) between two nodes:
                if len(found_paths) > 1:
                    branches += found_paths
    return branches


def get_branch_statistics(new_df, hist_df, imp_columns):
    current_workflow_branches = get_workflow_branches(new_df)
    # once we have all the branches, we retrieve the labels and features information
    statistics = []
    # we set the id to index to facilitate task's id to name conversion
    ided_tsks_new_exec = new_df.set_index('id', inplace=False, drop=True)
    for branch in current_workflow_branches:
        path_statistics = {}
        branch_names = ids_to_names(branch, ided_tsks_new_exec)
        path_statistics['branch_ids'] = branch
        path_statistics['branch_names'] = branch_names

        # for branch specific stats, we need to match both the ID and the workflow name
        workflows_names = set(ided_tsks_new_exec.loc[branch, 'workflow_name'])
        branch_hist_data = hist_df.loc[hist_df['workflow_name'].isin(workflows_names) &
                                       hist_df['id'].astype(str).isin(branch)]

        for column in imp_columns:
            branch_statistics_key = "branch_statistics: {}".format(column)
            # for both workflow specific stats and overall stats, save to dictionary
            path_statistics[branch_statistics_key] = count_ratios_and_means(branch_hist_data, column)
        statistics.append(path_statistics)
    # store the paths statistics results into new dataframe
    return statistics


def get_tasks_statistics(new_df, hist_df, imp_columns):
    all_tasks = new_df['name'].unique()
    statistics = []
    for task in all_tasks:
        task_statistics = {}
        task_statistics['task'] = task
        task_relevant_data = hist_df.loc[hist_df['name'] == task]
        for column in imp_columns:
            task_statistics_key = "task_statistics: {}".format(column)
            task_statistics[task_statistics_key] = count_ratios_and_means(task_relevant_data, column)

        statistics.append(task_statistics)
    return statistics


def analyse(tsk_hist_data,
            tsk_new_exec,
            tsk_imp_feat,
            tsk_rf_label_map):
    # get all important columns based on the unique values of user given labels and ML important features
    imp_columns = []
    for feature_pair in list(set().union(*tsk_imp_feat.values())):
        if len(feature_pair) > 0:
            feature = feature_pair[0]
            # the features are hot-encoded so we need to only take the column original name
            if "!-->" in feature:
                feature = feature.split("!-->")[0]
            imp_columns.append(feature)
    for label in list(set().union(*tsk_rf_label_map.values())):
        imp_columns.append(label)

    # remove any duplicate column
    imp_columns = set(imp_columns)

    branch_stats = get_branch_statistics(tsk_new_exec, tsk_hist_data, imp_columns)
    tasks_stats = get_tasks_statistics(tsk_new_exec, tsk_hist_data, imp_columns)

    return pd.DataFrame(branch_stats), pd.DataFrame(tasks_stats)
