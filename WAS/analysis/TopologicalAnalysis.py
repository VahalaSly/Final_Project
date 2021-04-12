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


def get_columns_ratio(path, target_columns, dataframe, ratios_and_means):
    for column in target_columns:
        relevant_data = dataframe.loc[dataframe['name'].isin(path)]
        if relevant_data[column].dtype == 'object':
            count = relevant_data[column].value_counts(normalize=True)
            ratios_and_means[column].append(["{}, ratio: {} | ".format(value, ratio) for value, ratio in count.items()])
        else:
            ratios_and_means[column].append(relevant_data[column].mean())
    return ratios_and_means


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


def ids_to_names(path, dataframe):
    named_tasks = []
    for task in path:
        named_tasks.append(str(dataframe.loc[task, 'name']))
    return named_tasks


def analyse(tsk_hist_data,
            tsk_new_exec,
            tsk_imp_feat,
            tsk_rf_label_map):
    new_data_graph = create_graph(tsk_new_exec)

    # create a dictionary with a key for each feature and label
    # this dictionary will be used to store the ratios and means of each path
    paths_statistics = {}
    paths_statistics['path'] = []
    paths_statistics['in current workflow?'] = []
    imp_columns = []
    for feature_pair in list(set().union(*tsk_imp_feat.values())):
        if len(feature_pair) > 0:
            feature = feature_pair[0]
            # the features are hot-encoded so we need to only take the column original name
            if "!-->" in feature:
                feature = feature.split("!-->")[0]
            imp_columns.append(feature)
            paths_statistics[feature] = []
    for label in list(set().union(*tsk_rf_label_map.values())):
        paths_statistics[label] = []
        imp_columns.append(label)

    # remove any duplicate column
    imp_columns = set(imp_columns)

    # we set the id to index to facilitate task id_to_name conversion
    tsk_new_exec.set_index('id', inplace=True, drop=True)
    start_nodes, end_nodes = find_start_end_of_branches(new_data_graph)
    # we get all the nodes that have been recognised as starting or ending a branch
    # for each of the starting/ending pair, we find the paths...
    for start_node in start_nodes:
        for end_node in end_nodes:
            if start_node != end_node:
                # ...find all the paths between these two nodes...
                current_workflow_paths = find_all_paths(new_data_graph, start_node, end_node)
                # ... if there is more than one path, then we know we are interested...
                if len(current_workflow_paths) > 1:
                    for path in current_workflow_paths:
                        # ...finally, for each of the branches calculate the ratios of the features and labels!
                        path = ids_to_names(path, tsk_new_exec)
                        get_columns_ratio(path, imp_columns, tsk_hist_data, paths_statistics)
                        paths_statistics['path'].append(path)

    # store the paths statistics results into new dataframe
    stats = pd.DataFrame.from_dict(paths_statistics, orient='index').transpose()

    return stats
