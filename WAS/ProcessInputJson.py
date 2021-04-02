import json
from pandas import json_normalize
import sys


# taken from https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
def flatten_json(nested_json):
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


def parse_workflow(environment, workflow, workflows_json, nodes_json):
    if 'subWorkflow' in workflow.keys():
        workflow = workflow['subWorkflow']
    nodes_key = 'nodes'
    try:
        nodes = workflow['nodes']
    except KeyError as e:
        print("Could not find mandatory key {}. Make sure the JSON file is correctly formatted.".format(nodes_key))
        sys.stderr.write(str(e) + "\n")
        return KeyError
    workflow.pop('nodes')
    flattened_workflow = flatten_json(workflow)
    # all workflows share the same environment information
    # so we append that info to each workflow
    flattened_workflow.update(environment)
    workflows_json.append(flattened_workflow)
    for node in nodes:
        if "subWorkflow" in node.keys():
            parse_workflow(environment, node, workflows_json, nodes_json)
        else:
            nodes_json.append(flatten_json(node))


def main(filepath):
    data = open(filepath).read()
    execution_summary = json.loads(data)
    workflow_json = []
    nodes_json = []
    environment = {}
    workflow_key = 'workflow'
    # we want to get all the execution information except the workflow information
    # which are stored in their own variable
    for key, value in execution_summary.items():
        if key != workflow_key:
            environment[key] = value
    flatten_json(environment)
    try:
        workflow = execution_summary[workflow_key]
    except KeyError as e:
        print("Could not find mandatory key {}. Make sure the JSON file is correctly formatted.".format(workflow_key))
        sys.stderr.write(str(e) + "\n")
        raise KeyError
    parse_workflow(environment, workflow, workflow_json, nodes_json)
    workflows = json_normalize(workflow_json)
    nodes = json_normalize(nodes_json)

    return nodes, workflows
