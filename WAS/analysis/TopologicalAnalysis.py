import pandas as pd


def create_graph(dataframe):
    # the successors are divided among multiple columns
    # this finds all of these columns (through regex search)
    graph = {}
    all_successors = dataframe.filter(regex='successors').filter(regex='id')
    if len(all_successors.columns) != 0:
        # puts all of the successors columns together and drops empty
        all_successors = all_successors.T.apply(lambda x: x.dropna().tolist())
        names = list(dataframe['name'])
        for i in range(0, len(names)):
            name = names[i]
            successors = all_successors[i]
            if name in graph.keys():
                graph[name] = list(set(graph[name]).union(set(successors)))
            else:
                graph[name] = successors
    print(graph)
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
                paths.append(newpath)
    return paths


def find_all_reachable_nodes(reachable_nodes, graph, node):
    if len(graph[node]) == 0:
        return
    for successor in graph[node]:
        if successor not in reachable_nodes:
            reachable_nodes.append(successor)
        if successor != node:
            find_all_reachable_nodes(reachable_nodes, graph, successor)


def get_columns_ratio(hist_path, target_columns, hist_data, ratios_and_means):
    for column in target_columns:
        relevant_data = hist_data.loc[hist_data['name'].isin(hist_path)]
        if relevant_data[column].dtype == 'object':
            count = relevant_data[column].value_counts(normalize=True)
            ratios_and_means[column].append(["{}, ratio: {} | ".format(value, ratio) for value, ratio in count.items()])
        else:
            ratios_and_means[column].append(relevant_data[column].mean())
    return ratios_and_means


def analyse(tsk_hist_data,
            tsk_new_exec,
            tsk_imp_feat,
            tsk_rf_label_map):
    hist_data_graph = create_graph(tsk_hist_data)
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
            if "!-->" in feature:
                feature = feature.split("!-->")[0]
            imp_columns.append(feature)
            paths_statistics[feature] = []
    for label in list(set().union(*tsk_rf_label_map.values())):
        paths_statistics[label] = []
        imp_columns.append(label)

    for node in new_data_graph:
        # first, for each node find all the nodes it can reach (aka find all point A and point B on the new data nodes)
        reachable_nodes = []
        find_all_reachable_nodes(reachable_nodes, new_data_graph, node)
        # then, for each node that can be reached by current node...
        for reachable_node in reachable_nodes:
            # ...find all the paths in historical data between these two nodes
            current_workflow_paths = find_all_paths(new_data_graph, node, reachable_node)
            hist_workflow_paths = find_all_paths(hist_data_graph, node, reachable_node)
            for path in hist_workflow_paths:
                # finally, for each path found calculate the ratios of the features and labels!
                get_columns_ratio(path, imp_columns, tsk_hist_data, paths_statistics)
                paths_statistics['path'].append(path)
                if path in current_workflow_paths:
                    paths_statistics['in current workflow?'].append(True)
                else:
                    paths_statistics['in current workflow?'].append(False)

    # store the paths statistics results into new dataframe
    stats = pd.DataFrame.from_dict(paths_statistics, orient='index').transpose()

    return stats
