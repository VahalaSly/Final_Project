import xml.etree.cElementTree as ET
import pandas as pd
import os
from tabulate import tabulate
from WAS.Node import Node


def get_workflow_nodes(root):
    nodes = {}
    parent_workflow = ""
    for workflow in root.iter('workflow'):
        parent_workflow = workflow.attrib['name']
    for node in root.iter('node'):
        successors = []
        state = node.attrib['state']
        node_id = node.attrib['id']
        graph_depth = node.attrib['graphDepth']
        warning = False
        error = False
        for successor in node.iter("successor"):
            successors.append(successor.attrib['id'])
        for factoryKey in node.iter("factoryKey"):
            class_name = factoryKey.attrib['className']
        for executionStatistics in node.iter("executionStatistics"):
            execution_duration = executionStatistics.attrib['lastExecutionDuration']
        for nodeMessage in node.iter('nodeMessage'):
            if nodeMessage.attrib['type'] == "WARNING":
                warning = True
            if nodeMessage.attrib['type'] == "ERROR":
                error = True
        new_node = Node(graph_depth=graph_depth,
                        node_id=node_id,
                        parent_workflow=parent_workflow,
                        class_name=class_name,
                        state=state, successors=successors,
                        warnings=warning,
                        errors=error,
                        execution_duration=execution_duration)
        nodes[node_id] = new_node
    return nodes


def xml_to_tree(dir_path):
    nodes_trees = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.xml'):
            fullname = os.path.join(dir_path, filename)
            xtree = ET.parse(fullname)
            nodes_trees.append(xtree)
    return nodes_trees


def infer_predecessor(workflow_nodes):
    for id, node in workflow_nodes.items():
        for successor in node.successors:
            workflow_nodes[successor].predecessors.append(node)


if __name__ == "__main__":
    dir_path = "xml_files/random_tree"
    trees = xml_to_tree(dir_path)
    nodes = []
    for tree in trees:
        workflow_nodes = get_workflow_nodes(tree.getroot())
        infer_predecessor(workflow_nodes)
        for id, node in workflow_nodes.items():
            nodes.append(node)

    df = pd.DataFrame.from_records([node.to_dict() for node in nodes])
    print(tabulate(df, headers='keys', tablefmt='psql', showindex='false'))
    df.to_csv('csvs/summary.csv', index=None)
