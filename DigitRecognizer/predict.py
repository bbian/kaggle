import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

df_train = pd.read_csv('train.csv', header=0, delimiter=',')
X_train = df_train[list(df_train.columns)[1:]]
y_train = df_train['label']

df_test = pd.read_csv('test.csv', header=0, delimiter=',')
X_test = df_test[:]
y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

submission = y_pred.reshape(y_pred.shape[0], 1)
submit_df = pd.DataFrame(submission)
submit_df.to_csv("prediction.csv")
