def get_feedback(results, input_data, labels):
    feedback = {}
    for rf_type, label in labels.items():
        features = []
        predictions = []
        label_result = results[rf_type]
        error = label_result['error']
        if error < 0.2:
            for feature, importance in label_result['features_importance']:
                if importance > 0:
                    features.append((feature, importance))
            for i in range(0, len(label_result['predictions'])):
                predictions.append((input_data.loc[i, 'name'], label_result['predictions'][i]))
            feedback[label] = (predictions, features)
    return feedback


def produce_report(task_feedback, workflow_feedback):
    pass
