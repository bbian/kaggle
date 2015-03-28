import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# First calculate naive predication using all numerical data
df_train = pd.read_csv('train.csv', sep=',')
X_train = df_train[list(df_train.columns)[5:-1]]
y_train = df_train['revenue']

regressor = LinearRegression()
regressor.fit(X_train, y_train)

df_test = pd.read_csv('test.csv', sep=',')
X_test = df_test[list(df_test.columns)[5:]]

y_pred = regressor.predict(X_test)
submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv('firstSubmission.csv')

# Re-trim X and convert/add categorical features
df_total = df_train.append(df_test, ignore_index=True)
X_total = df_total
X_total = X_total.drop('Type', 1)
X_total = X_total.drop('City', 1)
X_total = X_total.drop('City Group', 1)
X_total = X_total.drop('Id', 1)
X_total = X_total.drop('Open Date', 1)
X_total = X_total.drop('revenue', 1)

# Convert and add back 'Type' feature
feature = df_total['Type']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)

# Regenerate train and test set and run prediction
X_train = X_total[:X_train.shape[0]]
regressor.fit(X_train, y_train)
X_test = X_total[X_train.shape[0]:]
y_pred = regressor.predict(X_test)
submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv('secondSubmission.csv')

# Convert and add back 'City Group' feature
feature = df_total['City Group']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)

# Regenerate train and test set and run prediction
X_train = X_total[:X_train.shape[0]]
regressor.fit(X_train, y_train)
X_test = X_total[X_train.shape[0]:]
y_pred = regressor.predict(X_test)
submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv('thirdSubmission.csv')

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

# Regenerate train and test set and run prediction
X_train = X_total[:X_train.shape[0]]
regressor.fit(X_train, y_train)
X_test = X_total[X_train.shape[0]:]
y_pred = regressor.predict(X_test)
submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv('fourthSubmission.csv')


'''
# Convert and add back 'City' feature
feature = df_total['City']
feature_list = feature.T.to_dict().values()
feature = pd.get_dummies(feature_list)
X_total = feature.join(X_total)

# Regenerate train and test set and run prediction
X_train = X_total[:X_train.shape[0]]
regressor.fit(X_train, y_train)
X_test = X_total[X_train.shape[0]:]
y_pred = regressor.predict(X_test)
submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv('fourthSubmission.csv')
'''
