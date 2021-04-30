import json
from pandas import json_normalize
import sys


# taken from https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
def flatten_json(nested_json):
    # both the workflow and the nodes have the change of being represented by nested lists and dictionaries;
    # they need to be "flattened" before being stored.
    out = {}

    def flatten(json_item, name=''):
        if type(json_item) is dict:
            for key in json_item:
                flatten(json_item[key], name + key + '.')
        elif type(json_item) is list:
            i = 0
            for item in json_item:
                flatten(item, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = json_item

    flatten(nested_json)
    return out


def parse_workflow(environment, workflow, workflows_json=None, nodes_json=None):
    nodes_key = 'nodes'
    name_key = 'name'
    subworkflow_key = 'subWorkflow'
    nodes = []
    workflow_name = 'UnKnown'
    if workflows_json is None:
        workflows_json = []
    if nodes_json is None:
        nodes_json = []
    # check if subworkflow
    if subworkflow_key in workflow.keys():
        workflow = workflow[subworkflow_key]
    # look for nodes
    if nodes_key in workflow.keys():
        nodes = workflow[nodes_key]
        # we copy to avoid mutating the original dictionary
        workflow = workflow.copy()
        workflow.pop(nodes_key)
    # check if subworkflow has a name
    if name_key in workflow.keys():
        workflow_name = workflow[name_key]

    # all workflows share the same environment information
    # so we append that info to each workflow
    flattened_workflow = flatten_json(workflow)
    if len(environment) > 0:
        flattened_workflow.update(environment)
    workflows_json.append(flattened_workflow)
    for node in nodes:
        if subworkflow_key in node.keys():
            parse_workflow(environment, node, workflows_json, nodes_json)
            node = node.copy()
            node.pop(subworkflow_key)
        node_json = flatten_json(flatten_json(node))
        node_json.update({'workflow_name': workflow_name})
        nodes_json.append(node_json)
    return workflows_json, nodes_json


def json_to_dataframe(summary):
    environment = {}
    workflow_key = 'workflow'
    # we want to get all the execution information except the workflow information
    # which is stored in its own variable
    for key, value in summary.items():
        if key != workflow_key:
            environment[key] = value
    environment = flatten_json(environment)
    try:
        workflow = summary[workflow_key]
        workflow_json, nodes_json = parse_workflow(environment, workflow)
        workflows = json_normalize(workflow_json)
        nodes = json_normalize(nodes_json)
    except KeyError as e:
        print("Could not find mandatory key. Make sure the JSON file is correctly formatted.")
        sys.stderr.write(str(e) + "\n")
        raise KeyError

    return workflows, nodes


def get_nodes_and_workflows(filepath):
    data = open(filepath, encoding="utf8").read()
    summary = json.loads(data)
    return json_to_dataframe(summary)
