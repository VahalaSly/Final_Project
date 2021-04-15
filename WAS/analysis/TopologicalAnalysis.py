import pandas as pd


def create_graph(dataframe):
    # the successors are divided among multiple columns
    # this finds all of these columns (through regex search)
    graph = {}
    all_successors = dataframe.filter(regex='successors').filter(regex='id')
    if len(all_successors.columns) != 0:
        # puts all of the successors columns together and drops empty
        all_successors = all_successors.T.apply(lambda x: x.dropna().tolist())
        ids = list(dataframe['id'])
        for i in range(0, len(ids)):
            name = ids[i]
            successors = all_successors[i]
            if name in graph.keys():
                graph[name] = list(set(graph[name]).union(set(successors)))
            else:
                graph[name] = successors
    return graph


def find_start_end_of_branches(graph):
    start_nodes = []
    end_nodes = []
    is_successor_of_count = {}
    has_successors_count = {}
    for node, successors in graph.items():
        has_successors_count[node] = len(successors)
        if node not in is_successor_of_count.keys():
            is_successor_of_count[node] = 0
        for successor in successors:
            if successor in is_successor_of_count:
                is_successor_of_count[successor] += 1
            else:
                is_successor_of_count[successor] = 1
    for successor, count in is_successor_of_count.items():
        if count > 1:
            end_nodes.append(successor)
    for successor, count in has_successors_count.items():
        if count > 1:
            start_nodes.append(successor)
    return start_nodes, end_nodes


# https://www.python.org/doc/essays/graphs/
def find_all_paths(graph, start, end, path=None):
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                if newpath not in paths:
                    paths.append(newpath)
    return paths


def get_ratio_or_mean(df, column):
    if df[column].dtype == 'object':
        # get overall tasks ratios
        count = df[column].value_counts(normalize=True)
        return ["{}, ratio: {} | ".format(value, ratio) for value, ratio in count.items()]
    else:
        return df[column].mean()


def get_statistics(branch, target_columns, hist_dataframe, new_dataframe):
    path_statistics = {}
    # assigning path key so it can be first in the dataframe
    # just for aesthetic reasons
    path_statistics['path'] = []
    for column in target_columns:
        # for each column, we create one key for workflow-branch specific stats
        # and one key for overall stats for each task
        overall_tasks_col_key = "overall_tasks: {}".format(column)
        wk_specific_col_key = "branch_specific: {}".format(column)
        if overall_tasks_col_key not in path_statistics.keys():
            path_statistics[overall_tasks_col_key] = []
        if wk_specific_col_key not in path_statistics.keys():
            path_statistics[wk_specific_col_key] = []

        wk_relevant_data = []
        overall_relevant_data = []
        named_tasks = []
        for task in branch:
            # the workflow specific data needs to match both the workflow_name of the task and its ID
            workflow_name = new_dataframe.loc[task, 'workflow_name']
            wk_relevant_data.append(
                (hist_dataframe.loc[
                    (hist_dataframe['id'].astype('string') == str(task)) &
                    (hist_dataframe['workflow_name'] == workflow_name)]
                ).to_dict()
            )
            # for the overall data, it just needs to match the name
            task_name = new_dataframe.loc[task, 'name']
            named_tasks.append(task_name)
            overall_relevant_data.append(
                (hist_dataframe.loc[hist_dataframe['name'] == task_name]
                ).to_dict()
            )

        wk_relevant_data_df = pd.DataFrame(wk_relevant_data)
        overall_relevant_data_df = pd.DataFrame(overall_relevant_data)

        path_statistics['path'] = named_tasks
        path_statistics[overall_tasks_col_key].append(get_ratio_or_mean(overall_relevant_data_df, column))
        path_statistics[wk_specific_col_key].append(get_ratio_or_mean(wk_relevant_data_df, column))
    return path_statistics


def analyse(tsk_hist_data,
            tsk_new_exec,
            tsk_imp_feat,
            tsk_rf_label_map):
    new_data_graph = create_graph(tsk_new_exec)

    # create a dictionary with a key for each feature and label
    # this dictionary will be used to store the ratios and means of each path
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

    # we set the id to index to facilitate task's id to name conversion
    ided_tsks_new_exec = tsk_new_exec.set_index('id', inplace=False, drop=True)
    start_nodes, end_nodes = find_start_end_of_branches(new_data_graph)
    # we get all the nodes that have been recognised as starting or ending a branch
    # for each of the starting/ending pair, we find the paths...
    current_workflow_branches = []
    for start_node in start_nodes:
        for end_node in end_nodes:
            # some start nodes of paths might also be end nodes of other paths.
            # We do not want to look for paths between the same node.
            if start_node != end_node:
                found_paths = find_all_paths(new_data_graph, start_node, end_node)
                # if there's more than one path (branch) between two nodes:
                if len(found_paths) > 1:
                    current_workflow_branches += found_paths
    # once we have all the branches, we retrieve the labels and features information
    statistics = []
    for branch in current_workflow_branches:
        statistics.append(get_statistics(branch, imp_columns, tsk_hist_data, ided_tsks_new_exec))
    # store the paths statistics results into new dataframe
    stats = pd.DataFrame(statistics)
    return stats
