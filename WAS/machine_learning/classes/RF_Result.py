class RFResult:
    """ Class RFResult represents an instance of random forest results for a given label """

    def __init__(self, label_name, predictions, features_importance, error):
        self.label_name = label_name
        self.predictions = predictions
        self.features_importance = features_importance
        self.error = error
