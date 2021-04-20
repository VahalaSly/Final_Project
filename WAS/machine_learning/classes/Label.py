import numpy as np


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
        return list(self.train_values) + list(self.test_values)

    @property
    def train_values_array(self):
        encoded_values = self.encode(self.train_values)
        return np.array(encoded_values)

    @property
    def test_values_array(self):
        encoded_values = self.encode(self.test_values)
        return np.array(encoded_values)

    def encode(self, values):
        if self.encoder is not None:
            self.encoder.fit(self.all_values)
            values = self.encoder.transform(values)
        return values

    def decode(self, values):
        if self.encoder is not None:
            values = self.encoder.inverse_transform(values)
        return values
