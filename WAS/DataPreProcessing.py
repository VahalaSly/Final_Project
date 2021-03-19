import xml.etree.cElementTree as ET
import pandas as pd
import os
from tabulate import tabulate
from WAS.Node import Node


def xml_to_tree(filename):
    if filename.endswith('.xml'):
        try:
            xtree = ET.parse(filename)
            return xtree
        except ET.ParseError as e:
            print(e)
            return False
    return False


def parse_workflow_nodes(root):
    nodes_dict = {}
    parent_workflow = ""
    for workflow in root.iter('workflow'):
        parent_workflow = workflow.attrib['name']
    for node_tag in root.iter('node'):
        successors = []
        state = node_tag.attrib['state']
        node_id = node_tag.attrib['id']
        graph_depth = node_tag.attrib['graphDepth']
        warning = False
        error = False
        for successor in node_tag.iter("successor"):
            successors.append(successor.attrib['id'])
        for factoryKey in node_tag.iter("factoryKey"):
            class_name = factoryKey.attrib['className']
        for executionStatistics in node_tag.iter("executionStatistics"):
            execution_duration = executionStatistics.attrib['lastExecutionDuration']
            execution_datetime = executionStatistics.attrib['lastExecutionStartTime']
        for nodeMessage in node_tag.iter('nodeMessage'):
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
                        execution_duration=execution_duration,
                        execution_datetime=execution_datetime)
        nodes_dict[node_id] = new_node
    return nodes_dict


def main(xml_path, csv_path):
    workflow = xml_to_tree(xml_path)
    nodes = []
    if workflow:
        workflow_nodes = parse_workflow_nodes(workflow.getroot())
        for id, node in workflow_nodes.items():
            node.infer_predecessor(workflow_nodes)
            nodes.append(node)
    else:
        print("The workflow summary provided was not in XML format or was corrupted. "
              "Make sure the path is correct and provide a valid file.")
        return False

    data_frame = pd.DataFrame.from_records([node.to_ml_ready_dict() for node in nodes]).fillna(0)
    # print(tabulate(data_frame, headers='keys', tablefmt='psql', showindex='false'))
    data_frame.to_csv(csv_path, index=False)
    return True
