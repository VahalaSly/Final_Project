import lxml.etree as ET
import pandas as pd
from WAS.Classes.Node import Node
from WAS.Classes.Workflow import Workflow


def xml_to_tree(filename):
    if filename.endswith('.xml'):
        try:
            xtree = ET.parse(filename)
            return xtree
        except ET.ParseError as e:
            print(e)
            return False
    return False


def parse_node(node):
    successors = []
    state = node.attrib['state']
    graph_depth = node.attrib['graphDepth']
    node_name = node.attrib['name']
    node_id = node.attrib['id']
    warning = False
    error = False
    executionStatistics = node.find('executionStatistics')
    # if the executionStatistics tag is missing, it means the task/node was never executed
    # therefore, we skip it
    if executionStatistics is not None:
        execution_duration = int(executionStatistics.attrib['lastExecutionDuration'])
        execution_datetime = executionStatistics.attrib['lastExecutionStartTime']
        for successor in node.iter('successor'):
            successors.append(successor.attrib['id'])
        nodeMessage = node.find('nodeMessage')
        if nodeMessage is not None:
            if nodeMessage.attrib['type'] == "WARNING":
                warning = True
            if nodeMessage.attrib['type'] == "ERROR":
                error = True
        return Node(graph_depth=graph_depth,
                    node_name=node_name,
                    node_id=node_id,
                    state=state, successors=successors,
                    warnings=warning,
                    errors=error,
                    execution_duration=execution_duration,
                    execution_datetime=execution_datetime)


def parse_workflow(root, nodes, workflows_list, user, parent_workflow):
    workflow_tag = None
    workflow = Workflow()
    # check if workflow or sub-workflow, and find name in corresponding tag
    if root.find('workflow') is not None:
        workflow_tag = root.find('workflow')
    if root.find('subWorkflow') is not None:
        workflow_tag = root.find('subWorkflow')
    # if we are inside a workflow...
    if workflow_tag is not None:
        workflow_tag_name = workflow_tag.attrib['name']
        workflow.name = workflow_tag_name
        workflow.user = user
        # if the workflow has a parent (a sub-workflow would)...
        if parent_workflow is not None:
            # add the current workflow as a child to the parent workflow
            parent_workflow.children_workflows.append(workflow)
        all_nodes = workflow_tag.find('nodes').findall('node')
        # go through each node in the workflow
        for node_tag in all_nodes:
            # a component is another workflow
            # recursively call parse_workflow on the node with current workflow as parent workflow
            if 'component' in node_tag.attrib:
                parse_workflow(node_tag, nodes, workflows_list, user, workflow)
            new_node = parse_node(node_tag)
            # the new_node will return node if it wasn't executed
            if new_node is not None:
                for node in nodes:
                    if new_node.id in node.successors:
                        new_node.predecessors.append(node)
                new_node.parent_workflow = workflow_tag_name
                nodes.append(new_node)
                workflow.nodes.append(new_node)
        workflows_list.append(workflow)


def main(xml_path, tasks_csv_path, workflows_csv_path):
    workflow = xml_to_tree(xml_path)
    nodes_lst = []
    workflows_lst = []
    if workflow:
        root = workflow.getroot()
        user = 'UnKnown'
        for userName in root.iter('user.name'):
            user = userName.text
        parse_workflow(root, nodes_lst, workflows_lst, user, None)
    else:
        print("The workflow summary provided was not in XML format or was corrupted. "
              "Make sure the path is correct and provide a valid file.")
        return False

    tasks_df = pd.DataFrame.from_records([node.to_ml_ready_dict() for node in nodes_lst]).fillna(0)
    workflow_df = pd.DataFrame.from_records([workflow.to_ml_ready_dict() for workflow in workflows_lst]).fillna(0)
    tasks_df.to_csv(tasks_csv_path, index=False)
    workflow_df.to_csv(workflows_csv_path, index=False)
    for workflow in workflows_lst:
        if workflow.name == "Collect Information":
            for node in workflow.nodes:
                print(node.name)
                print(node.has_failed)
    return True
