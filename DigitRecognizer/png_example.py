import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sys
import mahotas as mh
import scipy as sp

K = 1

def distance(p0, p1):
	return np.sum((p0 - p1) ** 2)

def nn_classify(training_set, training_labels, new_example):
	dists = np.array([distance(t, new_example) for t in training_set])
	nearest_indicies = np.argsort(dists)
	nearest_indicies = nearest_indicies[:K]
#	print ("Nearest K indicies", nearest_indicies)

	# Find the corresponding K number of labels
	nearest_k_labels = training_labels[nearest_indicies]
#	print ("Nearest K labels", nearest_k_labels)
	voted_label = np.bincount(nearest_k_labels).argmax()
#	print ("Voted label:", voted_label)

	# Get the actual count of the voted label
	count = np.bincount(nearest_k_labels)[voted_label]

	# Now we want to locate the first index in the 
	# nearest_k_labels[] array that has the same count.
	# If it happens to be the voted_label, that's fine.
	# If there are ties, we will ensure that the first
	# such index is the nearest neighbor to our new_example,
	# and that's the one we should use.

	final_label = np.argmax(np.bincount(nearest_k_labels)==count)
	return final_label

df_train = pd.read_csv('train.csv', header=0, delimiter=',')
X_train = df_train[list(df_train.columns)[1:]].values
y_train = df_train['label'].values

df_test = pd.read_csv('test.csv', header=0, delimiter=',')
X_test = df_test[:].values

# Add code just for dad to illustrate how to save images
X_train = X_train[1234:1236]
y_train = y_train[1234:1236]
X_test = X_test[4321:4323]

print "Convert training set to binary vectors"
for idx, image in enumerate(X_train):
	# Reshape image to 2D array
	image = np.reshape(image, (-1, 28))
	image = image.astype(np.uint8)
	#thresh = mh.thresholding.otsu(image)
	#image = (image > thresh)
	imageFileName = 'train' + str(idx) + '.png'
	mh.imsave(imageFileName, image)
	X_train[idx] = image.ravel().astype('int64')

print "Convert test set to binary vectors"
for idx, image in enumerate(X_test):
	# Reshape image to 2D array
	image = np.reshape(image, (-1, 28))
	image = image.astype(np.uint8)
	#thresh = mh.thresholding.otsu(image)
	#image = (image > thresh)
	imageFileName = 'test' + str(idx) + '.png'
	mh.imsave(imageFileName, image)
	X_test[idx] = image.ravel().astype('int64')

'''
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
'''
