def create_graph(nodes, successors):
    graph = {}
    for i in range(0, len(nodes)):
        name = nodes[i]
        node_successors = successors[i]
        if name in graph.keys():
            graph[name] = list(set(graph[name]).union(set(node_successors)))
        else:
            graph[name] = node_successors
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
        # if a node is the successor of more than one node, we know it's a branch ending node
        if count > 1:
            end_nodes.append(successor)
    for successor, count in has_successors_count.items():
        # if a node has more than one successor, we know it's a branch starting node
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
