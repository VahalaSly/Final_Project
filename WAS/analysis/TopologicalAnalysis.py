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
        # goes through each task and its successors, and creates a graph in the form of {task:[successors]}
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


def find_all_reachable_nodes(graph, node, reachable_nodes=None):
    if reachable_nodes is None:
        reachable_nodes = []
    for successor in graph[node]:
        if successor not in reachable_nodes and successor != node:
            reachable_nodes.append(successor)
            find_all_reachable_nodes(graph, successor, reachable_nodes)
    # return set to avoid repetition of nodes
    return set(reachable_nodes)


def count_ratios_and_means(df, column):
    if df[column].dtype == 'object':
        count = df[column].value_counts(normalize=True)
        return ["{}, ratio: {} | ".format(value, ratio) for
                value, ratio in count.items()]
    else:
        return df[column].mean()


def ids_to_names(path, dataframe):
    named_tasks = []
    for task in path:
        named_tasks.append(str(dataframe.loc[task, 'name']))
    return named_tasks


def get_branch_statistics(new_df, hist_df, imp_columns):
    new_data_graph = create_graph(new_df)
    # we set the id to index to facilitate task's id to name conversion
    ided_tsks_new_exec = new_df.set_index('id', inplace=False, drop=True)
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
        path_statistics = {}
        branch_names = ids_to_names(branch, ided_tsks_new_exec)
        path_statistics['branch_ids'] = branch
        path_statistics['branch_names'] = branch_names

        # for branch specific stats, we need to match both the ID and the workflow name
        workflows_names = set(ided_tsks_new_exec.loc[branch, 'workflow_name'])
        branch_hist_data = hist_df.loc[hist_df['id'].isin(branch) & hist_df['workflow_name'].isin(workflows_names)]

        for column in imp_columns:
            branch_statistics_key = "branch_statistics: {}".format(column)
            # for both workflow specific stats and overall stats, save to dictionary
            path_statistics[branch_statistics_key] = count_ratios_and_means(branch_hist_data, column)
        statistics.append(path_statistics)
    # store the paths statistics results into new dataframe
    return pd.DataFrame(statistics)


def get_tasks_statistics(hist_df, new_df, imp_columns):
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
    return pd.DataFrame(statistics)


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
    tasks_stats = get_tasks_statistics(tsk_hist_data, tsk_new_exec, imp_columns)

    return branch_stats, tasks_stats
