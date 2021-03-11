import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('csvs/summary.csv')

#Creating the dependent variable class - change text to a numeric ID
factor = pd.factorize(dataset['State'])
print(factor[0])
definitions = factor[1]
# print(dataset.state.head())
# print(definitions)