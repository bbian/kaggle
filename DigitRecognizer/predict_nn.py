import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sys

def distance(p0, p1):
	return np.sum((p0 - p1) ** 2)

def nn_classify(training_set, training_labels, new_example):
	dists = np.array([distance(t, new_example) for t in training_set])
	nearest = dists.argmin()
	return training_labels[nearest]

df_train = pd.read_csv('train.csv', header=0, delimiter=',')
X_train = df_train[list(df_train.columns)[1:]].values
y_train = df_train['label'].values

df_test = pd.read_csv('test.csv', header=0, delimiter=',')
X_test = df_test[:].values

predictions = []
for item in X_test:
	sys.stdout.write('.')
	sys.stdout.flush()
	predictions.append(nn_classify(X_train, y_train, item))

print "\n"

y_pred = np.array(predictions)

submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.index += 1
submit_df.to_csv("results.csv", header=['Label'], index_label='ImageId')
