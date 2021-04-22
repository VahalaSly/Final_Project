class Label:
    """ Class Label represents an instance of Label for Random Forest training and testing """

    def __init__(self, name, rf_type, train_values, test_values, encoder):
        self.name = name
        self.rf_type = rf_type
        self.train_values = train_values
        self.test_values = test_values
        self.encoder = encoder

    @property
    def all_values(self):
        all_values = list(self.train_values) + list(self.test_values)
        return all_values

    def encoded_train_test(self):
        train = self.train_values
        test = self.test_values
        if self.encoder is not None:
            self.encoder.fit(self.all_values)
            train = self.encoder.transform(self.train_values)
            test = self.encoder.transform(self.test_values)
        return train, test

    def decode(self, values):
        if self.encoder is not None:
            values = self.encoder.inverse_transform(values)
        return values
