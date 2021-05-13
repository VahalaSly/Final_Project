import numpy as np
import pandas
from WorkflowAnalysisSystem import add_latest_exec_to_historical_df
from WorkflowAnalysisSystem import get_filtered_dfs

sample_dataframe = pandas.DataFrame({
    'id': ['0', '1', '2', '3', 4],
    'state': ['executed', 'executed', 'executed', 'failed', 'failed'],
    'duration': [3, 2, 3, 3, 4]
})

sample_larger_dataframe = pandas.DataFrame({
    'id': ['1', '1', '3', 4, 4, 4, 4],
    'name': ['name1', 'name2', 'name3', 'name3', 'name3', 'name3', 'name3'],
    'duration': [2, 3, 1, 1, 2, 3, 3]
})


def test_add_latest_exec_to_historical_df():
    new_df_result = add_latest_exec_to_historical_df(sample_dataframe, sample_larger_dataframe)
    assert list(new_df_result['id']) == ['0', '1', '2', '3', 4, '1', '1', '3', 4, 4, 4, 4]
    assert list(new_df_result['name']) == [np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, 'name1', 'name2', 'name3', 'name3',
                                           'name3', 'name3', 'name3']
    assert list(new_df_result['state']) == ['executed', 'executed', 'executed', 'failed', 'failed', np.NAN, np.NAN,
                                            np.NAN, np.NAN,
                                            np.NAN, np.NAN, np.NAN]
    assert list(new_df_result['duration']) == [3, 2, 3, 3, 4, 2, 3, 1, 1, 2, 3, 3]


def test_get_filtered_dfs():
    features_list = ['id']
    labels_map = {'classifier': ['state'], 'regressor': []}
    filtered_sample_df, filtered_sample_larger_df = get_filtered_dfs(features_list, labels_map,
                                                                     sample_dataframe, sample_larger_dataframe)
    # using set since the list ordering can be mixed
    assert set(filtered_sample_df.columns) == {'id', 'state'}
    assert list(filtered_sample_larger_df.columns) == ['id']

    assert list(filtered_sample_df['state']) == list(sample_dataframe['state'])
    assert list(filtered_sample_df['id']) == list(sample_dataframe['id'])
    assert list(filtered_sample_larger_df['id']) == list(sample_larger_dataframe['id'])

    unfiltered_sample_df, unfiltered_sample_larger_df = get_filtered_dfs([], labels_map,
                                                                         sample_dataframe, sample_larger_dataframe)

    assert set(unfiltered_sample_df.columns) == {'id', 'state', 'duration'}
    assert set(unfiltered_sample_larger_df.columns) == {'id', 'name', 'duration'}

    assert list(unfiltered_sample_df['state']) == list(sample_dataframe['state'])
    assert list(unfiltered_sample_df['duration']) == list(sample_dataframe['duration'])
    assert list(unfiltered_sample_df['id']) == list(sample_dataframe['id'])
    assert list(unfiltered_sample_larger_df['id']) == list(sample_larger_dataframe['id'])
    assert list(unfiltered_sample_larger_df['name']) == list(sample_larger_dataframe['name'])
    assert list(unfiltered_sample_larger_df['duration']) == list(sample_larger_dataframe['duration'])
