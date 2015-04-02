import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import StandardScaler

def trainAndPredict(clf, submissionFileName):
	global X_total, X_train, y_train
	X_train = X_total[:X_train.shape[0]]
	clf.fit(X_train, y_train)
	X_test = X_total[X_train.shape[0]:]
	y_pred = clf.predict(X_test)
	submission = y_pred.reshape(y_pred.shape[0], 1)
	submit_df = pd.DataFrame(submission)
	submit_df.to_csv(submissionFileName)

# Preprocessing data
df_train = pd.read_csv('train.csv', sep=',')
X_train = df_train[list(df_train.columns)[:-1]]
y_train = df_train['revenue']

df_test = pd.read_csv('test.csv', sep=',')
df_total = df_train.append(df_test, ignore_index=True)

# First calculate naive predication using all numerical data
X_total = df_total
X_total = X_total.drop('Type', 1)
X_total = X_total.drop('City', 1)
X_total = X_total.drop('City Group', 1)
X_total = X_total.drop('Id', 1)
X_total = X_total.drop('Open Date', 1)
X_total = X_total.drop('revenue', 1)
regressor = LinearRegression()
trainAndPredict(regressor, 'submission-001.csv')

# Convert and add back 'Type' feature
feature = df_total['Type']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)
trainAndPredict(regressor, 'submission-002.csv')

# Convert and add back 'City Group' feature
feature = df_total['City Group']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)
trainAndPredict(regressor, 'submission-003.csv')

# Convert "Open Date" feature to # of days feature
from datetime import date
from datetime import datetime
today = date.today()
today_time = datetime.combine(today, datetime.min.time())
date_format = "%m/%d/%Y"

feature = df_total['Open Date']
for i, val in enumerate(feature):
	delta = today_time - datetime.strptime(val, date_format)
	feature[i] = delta.days
feature = feature.to_frame(name='Days Opened')
X_total = feature.join(X_total)
trainAndPredict(regressor, 'submission-004.csv')

# With the same feature vectors, try Ridge CV regressor
from sklearn.linear_model import RidgeCV
regressor = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 4135.0, 4136.0, 4137.0])
trainAndPredict(regressor, 'submission-005.csv')

# Scale features (is this optional?)
'''
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
regressor = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 4135.0, 4136.0, 4137.0])
trainAndPredict(regressor, 'submission-006.csv')
'''

# Convert and add back 'City' feature (results are actually worse)
'''
feature = df_total['City']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)
regressor = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 4135.0, 4136.0, 4137.0])
trainAndPredict(regressor, 'submission-006.csv')
'''
