from analysis.util.Graph_operations import create_graph
from analysis.util.Graph_operations import find_start_end_of_branches
from analysis.util.Graph_operations import find_all_paths


def test_create_graph():
    ids = [1, 2, 3, 4, 5]
    successors = [[2, 3], [3], [4, 5], [5], [5]]
    assert create_graph(ids, successors) == {1: [2, 3], 2: [3], 3: [4, 5], 4: [5], 5: [5]}


def test_find_start_end_of_branches():
    graph = {1: [2, 3], 2: [3], 3: [4, 5], 4: [5], 5: [5]}
    start_nodes, end_nodes = find_start_end_of_branches(graph)
    assert start_nodes == [1, 3]
    assert end_nodes == [3, 5]


def test_find_all_paths():
    graph = {1: [2, 3], 2: [3], 3: [4, 5], 4: [5], 5: [5]}
    start_node = 1
    end_node = 3
    assert find_all_paths(graph, start_node, end_node) == [[1, 2, 3], [1, 3]]
