import json
import os


def tasks_per_sec(tasks, duration):
    if duration > 0:
        return round(tasks / (duration * 0.001), 2)
    else:
        return 0


def infer_tasks_and_state(workflow, number_of_nodes=0, total_duration=0, executed=1):
    for node in workflow['nodes']:
        if 'subWorkflow' in node.keys():
            nodes, duration, executed = infer_tasks_and_state(node['subWorkflow'])
            number_of_nodes += nodes
            total_duration += total_duration
        else:
            number_of_nodes += 1
            if node['state'] != 'EXECUTED':
                executed = 0
            if 'executionStatistics' in node.keys():
                total_duration += node['executionStatistics']['executionDurationSinceStart']
    workflow['tasksPerSecond'] = tasks_per_sec(number_of_nodes, total_duration)
    workflow['executed'] = executed
    return number_of_nodes, total_duration, executed


files = os.listdir('raw_json_files/')
for file in files:
    data = open("raw_json_files/{}".format(file), encoding="utf8")
    json_data = json.load(data)
    infer_tasks_and_state(json_data['workflow'])

    with open('../json_files/{}'.format(file), 'w') as outfile:
        json.dump(json_data, outfile)
