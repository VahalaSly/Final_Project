from data_processing.ProcessInputJson import flatten_json
from data_processing.ProcessInputJson import parse_workflow

nodes_json = [{'name': 'node1', 'id': 1},
              {'name': 'node1', 'id': 2, 'execution':
                  {
                      'duration': 5.5
                  }},
              {'name': 'node2', 'id': 3, 'subWorkflow': {
                  'name': 'sub-workflow1',
                  'nodes': [{'name': 'node3.1', 'id': '3:1'}]
              }}]

workflow_json = {'name': 'workflow1', 'author': 'john',
                 'nodes': nodes_json}

environment_json = {'version': 0.3, 'creation_time': '11:30'}


def test_flatten_json():
    assert flatten_json(nodes_json[0]) == {'id': 1, 'name': 'node1'}
    assert flatten_json(nodes_json[1]) == {'execution.duration': 5.5, 'id': 2, 'name': 'node1'}


def test_parse_workflow():
    workflows, nodes = parse_workflow(environment_json, workflow_json)
    assert workflows == [
        {'author': 'john', 'creation_time': '11:30', 'name': 'workflow1', 'version': 0.3},
        {'creation_time': '11:30', 'name': 'sub-workflow1', 'version': 0.3}]
    assert nodes == [
        {'id': 1, 'name': 'node1', 'workflow_name': 'workflow1'},
        {'execution.duration': 5.5, 'id': 2, 'name': 'node1', 'workflow_name': 'workflow1'},
        {'id': '3:1', 'name': 'node3.1', 'workflow_name': 'sub-workflow1'},
        {'id': 3, 'name': 'node2', 'workflow_name': 'workflow1'}]
