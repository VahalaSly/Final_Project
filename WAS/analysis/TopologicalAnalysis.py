import pandas as pd


def create_graph(dataframe):
    successors = dataframe.filter(regex='successors').filter(regex='id')
    successors = successors.T.apply(lambda x: x.dropna().tolist())

    # imp_columns = list(tsk_imp_feat.keys()) + list(set().union(*tsk_rf_label_map.values()))
    # imp_columns.append('name')

    return dict(zip(dataframe['name'], successors))


def analyse(tsk_hist_data,
            tsk_new_exec,
            tsk_imp_feat,
            tsk_rf_label_map):
    hist_data_graph = create_graph(tsk_hist_data)
    new_data_graph = create_graph(tsk_new_exec)

    print(find_all_paths(hist_data_graph, 'Table Reader', 'Normalizations'))

    pass

    # from historical tasks, find all the ones that appear first (any at graph-depth 0 should do)
    # have each know its successors
    # create a large graph in which all these nodes are connected, creating a super-graph of the tasks
    # to do this the cleanest way possible create a successors column when parsing the input initially
    # for any node repeated, return most common target/feature (if object) and mean (if numeric)?
    # then, for each 2 nodes the new execution not adjacent, find_all_paths in the super-graph
    # for each path found, calculate the patterns of target label (if numeric sum up,
    # if object return rates of each value)
    # for each path found, also return each tasks feature_of_interest values
    # potentially, this whole thing will need to be shown as a graph,
    # with each connection of nodes having their own statistics


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
