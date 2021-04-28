import numpy as np
import pandas
from analysis.TopologicalAnalysis import count_ratios_and_means
from analysis.TopologicalAnalysis import get_branch_statistics
from analysis.TopologicalAnalysis import get_tasks_statistics
from analysis.TopologicalAnalysis import get_workflow_branches

sample_dataframe = pandas.DataFrame({
    'id': ['0', '1', '2', '3', '4'],
    'name': ['name0', 'name1', 'name2', 'name3', 'name3'],
    'successors_0_id': ['1', '4', '3', '4', np.NAN],
    'successors_1_id': ['2', np.NAN, np.NAN, np.NAN, np.NAN],
    'workflow_name': ['workflow1', 'workflow1', 'workflow1', 'workflow1', 'workflow1'],
    'state': ['executed', 'executed', 'executed', 'failed', 'failed'],
    'duration': [3, 2, 3, 3, 4]
})

sample_larger_dataframe = pandas.DataFrame({
    'id': ['1', '1', '3', '4', '4', '4', '4'],
    'name': ['name1', 'name2', 'name3', 'name3', 'name3', 'name3', 'name3'],
    'workflow_name': ['workflow1', 'workflow2', 'workflow1', 'workflow1', 'workflow1', 'workflow2', 'workflow1'],
    'state': ['failed', 'executed', 'failed', 'failed', 'failed', 'executed', 'executed'],
    'duration': [2, 3, 1, 1, 2, 3, 3]
})


def test_count_ratios_and_means():
    assert count_ratios_and_means(sample_dataframe, 'state') == ['executed, ratio: 0.6 | ',
                                                                 'failed, ratio: 0.4 | ']
    assert count_ratios_and_means(sample_dataframe, 'duration') == 3.0


def test_get_workflow_branches():
    assert get_workflow_branches(sample_dataframe) == [['0', '1', '4'], ['0', '2', '3', '4']]
    assert get_workflow_branches(sample_larger_dataframe) == []


def test_get_branch_statistics():
    # calculated ratios and means manually
    assert get_branch_statistics(sample_dataframe, sample_larger_dataframe, ['state', 'duration']) == [
        {'branch_ids': ['0', '1', '4'],
         'branch_names': ['name0', 'name1', 'name3'],
         'branch_statistics: duration': 2.0,
         'branch_statistics: state': ['failed, ratio: 0.75 | ',
                                      'executed, ratio: 0.25 | ']},
        {'branch_ids': ['0', '2', '3', '4'],
         'branch_names': ['name0', 'name2', 'name3', 'name3'],
         'branch_statistics: duration': 1.75,
         'branch_statistics: state': ['failed, ratio: 0.75 | ',
                                      'executed, ratio: 0.25 | ']}]


def test_get_tasks_statistics():
    # ratios and means calculated manually
    assert get_tasks_statistics(sample_dataframe, sample_larger_dataframe, ['state', 'duration'])[1:] == [
        {'task': 'name1',
         'task_statistics: duration': 2.0,
         'task_statistics: state': ['failed, ratio: 1.0 | ']},
        {'task': 'name2',
         'task_statistics: duration': 3.0,
         'task_statistics: state': ['executed, ratio: 1.0 | ']},
        {'task': 'name3',
         'task_statistics: duration': 2.0,
         'task_statistics: state': ['failed, ratio: 0.6 | ',
                                    'executed, ratio: 0.4 | ']}]

    # we assert the first value separately because it contains a NAN which does not accept the equality
    first_row = get_tasks_statistics(sample_dataframe, sample_larger_dataframe, ['state', 'duration'])[:1][0]
    assert first_row['task'] == 'name0'
    assert first_row['task_statistics: state'] == []
