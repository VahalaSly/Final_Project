# Final Project

## Workflow Analysis System (WAS)

The WAS takes the summary of execution of a workflow and uses a random forest algorithm to analyse the workflow
execution and suggest improvements to the workflow. The WAS handles both classification and regression when predicting
labels.

### The input

The WAS takes a JSON file as input. The accepted format of the file is quite flexible, however the following keys are
expected:

A `workflow` key, which in turn contains a `name` key, respectively providing the workflow's information, and the
workflow's name.

`workflow` also needs to contain a `nodes` key. `nodes` contains a flexible number of workflow nodes/tasks.

Each node is expected to have an `id` and a `name`.

If any node has successor nodes, the successors are expected to be stored by `id` under a `successor-id` key.
The `successor-id` key can be flexible in nomenclature as long as the words `successor` and `id` are contained in the
key.

The input JSON is iterated through each workflow and each node. If any of the nodes is a subworkflow, it should contain
the key `subWorkflow`.

If the nodes contain nested dictionaries, they will be flattened. This means that if a node value is given as:

`"nodes": [{"executionStatistics": {"executionDurationSinceStart": 811} } ]`

the value will be saved as:

`executionStatistics.executionDurationSinceStart = 811`.

### The command-line arguments

The WAS expects and accepts a number of arguments when called.

`-f` or `--file` passing the file-path to the JSON input file. Required.

`-u` or `--user` passing the username of the user executing the WAS. Required

`-tc` or `--task_classifier` passing one or multiple nodes/tasks labels for the classifier. The labels should be
categorical values. The labels must exists in the JSON file. Optional.

`-tr` or `--task_regressor` passing one or multiple nodes/tasks labels for the regressor. The labels must be continuous
numerical values. The labels must exists in the JSON file. Optional.

`-wc` or `--workflow_classifier` passing one or multiple workflow labels for the classifier. The labels should be
categorical values. The labels must exists in the JSON file. Optional.

`-wr` or `--workflow_regressor` passing one or multiple workflow labels for the regressor. The labels must be continuous
numerical values. The labels must exists in the JSON file. Optional.

`-tf` or `--task_features` passing names of keys (remember to account for the flattening of the input) which are to be
considered as features by the machine learning when attempting to predict the task labels. If none are given, then all keys given by
the JSON file are used. Optional.

`-wf` or `--workflow_features` passing names of keys (remember to account for the flattening of the input) which are to
be considered as features by the machine learning when attempting to predict the workflow labels. If none are given, then all keys
given by the JSON file are used. Optional.

### The profiles

The above command line arguments can be provided in the form of a profile. Profiles are text files containing the
desired parameters. The file can be passed when calling the WAS using the `@` symbol, for example:

`python WorkflowAnalysisSystem.py @../profiles/user_2`

Example profiles are provided in the 'profiles' directory.

### The machine learning analysis

The WAS uses the random forest predictions to analyse the labels and features of the workflow execution. For each of the
workflows and subworkflows provided as input, as well as for each task, the ML attempts to predict the labels and
returns both the predictions, and the features it deemed to be most important to infer the label value.

The ML predicts using any historical data available which matches the current workflows and tasks.

If the prediction error of the ML for a given label is too high (over 20%), no important features are returned as the ML
inference cannot be trusted.

### The topological analysis

To aid the user in improving the workflow, a topological analysis is also performed on the execution data.

The topological analysis first analyses any paths that branch out of a single node/task and reunite on a secondary
node/task. For each label and important feature, it calculates the ratios (if a categorical values), or the mean (if a
continous numerical values) of the branch's tasks using the historical data available.

Then, for each task provided by the JSON input it analysis the historical performance, and returns the calculated
ratios (if a categorical values) or mean (if a continous numerical values).

### The report
The WAS outputs a report containing all the above-detailed analyses in xlsx format. 


## Packages needed
To use the WAS, the following must be installed:

pandas:
`pip install pandas`

matplotlib:
`pip install matplotlib`

scikit-learn:
`pip install scikit-learn`

Jinja2:
`pip install Jinja2`

xlsxwriter:
`pip install xlsxwriter`

Python 3.6 or newer must also be installed.

### The tests
To run the tests available in the tests dir you need to install pytest using:

`pip install pytest`

Then, from the main WAS dir, run:

`python -m pytest`