import numpy as np
from data_processing.ProcessInputJson import flatten_json
from data_processing.ProcessInputJson import parse_workflow
from data_processing.ProcessInputJson import json_to_dataframe

nodes_json = [{'name': 'node1', 'id': 1},
              {'name': 'node1', 'id': 2, 'execution':
                  {
                      'duration': 5.5
                  }},
              {'name': 'node2', 'id': 3, 'subWorkflow': {
                  'name': 'sub-workflow1',
                  'nodes': [{'name': 'node3.1', 'id': '3:1'}]
              }}]

workflows_json = {'name': 'workflow1', 'author': 'john',
                  'nodes': nodes_json}

environment_json = {'version': 0.3, 'creation_time': '11:30'}


def test_flatten_json():
    assert flatten_json(nodes_json[0]) == {'id': 1, 'name': 'node1'}
    assert flatten_json(nodes_json[1]) == {'execution.duration': 5.5, 'id': 2, 'name': 'node1'}

    # edge cases
    assert flatten_json({}) == {}


def test_parse_workflow():
    workflow_json_result, node_json_result = parse_workflow(environment_json, workflows_json)
    assert workflow_json_result == [
        {'author': 'john', 'creation_time': '11:30', 'name': 'workflow1', 'version': 0.3},
        {'creation_time': '11:30', 'name': 'sub-workflow1', 'version': 0.3}]
    assert node_json_result == [
        {'id': 1, 'name': 'node1', 'workflow_name': 'workflow1'},
        {'execution.duration': 5.5, 'id': 2, 'name': 'node1', 'workflow_name': 'workflow1'},
        {'id': '3:1', 'name': 'node3.1', 'workflow_name': 'sub-workflow1'},
        {'id': 3, 'name': 'node2', 'workflow_name': 'workflow1'}]

    # edge cases
    workflow_json_result, node_json_result = parse_workflow({}, {})
    assert workflow_json_result == [{}]
    assert node_json_result == []


def test_json_to_dataframe():
    environment_json['workflow'] = workflows_json
    result_workflows, result_nodes = json_to_dataframe(environment_json)
    assert list(result_workflows.columns) == ['name', 'author', 'version', 'creation_time']
    assert list(result_nodes.columns) == ['name', 'id', 'workflow_name', 'execution.duration']

    assert list(result_workflows['name']) == ['workflow1', 'sub-workflow1']
    assert list(result_workflows['author']) == ['john', np.NAN]
    assert list(result_workflows['version']) == [0.3, 0.3]
    assert list(result_workflows['creation_time']) == ['11:30', '11:30']

    assert list(result_nodes['name']) == ['node1', 'node1', 'node3.1', 'node2']
    assert list(result_nodes['id']) == [1, 2, '3:1', 3]
    assert list(result_nodes['workflow_name']) == ['workflow1', 'workflow1', 'sub-workflow1', 'workflow1']
    # the nans don't match for numerical columns, so fill the nans with a value we can check
    assert list(result_nodes['execution.duration'].fillna(-1)) == [-1, 5.5, -1, -1]
