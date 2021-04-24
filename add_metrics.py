import json
import os

files = os.listdir('raw_json_files/')


def get_inferred_tasks_and_state(nodes, number_of_nodes=0, total_duration=0, failed=0):
    for node in nodes:
        if 'subWorkflow' in node.keys():
            n_nodes, t_duration, failed = get_inferred_tasks_and_state(node['subWorkflow']['nodes'],
                                                                       number_of_nodes, total_duration, failed)
            number_of_nodes = n_nodes
            total_duration = t_duration
        else:
            number_of_nodes += 1
            if node['state'] != 'EXECUTED':
                failed = 1
            if 'executionStatistics' in node.keys():
                total_duration += node['executionStatistics']['executionDurationSinceStart']
    return number_of_nodes, total_duration, failed


for file in files:
    data = open("raw_json_files/{}".format(file), encoding="utf8")
    json_data = json.load(data)
    num_nodes, duration, failed = get_inferred_tasks_and_state(json_data['workflow']['nodes'])
    json_data['workflow']['tasksPerSecond'] = round(duration/num_nodes, 2)
    json_data['workflow']['failure'] = failed

    with open('json_files/{}'.format(file), 'w') as outfile:
        json.dump(json_data, outfile)
