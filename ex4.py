# Amnon Ophir 302445804, Ross Bolotin 310918610

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64')
y = mnist['target']
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]  # The next line flattens the vector into 1D array of size 784
X = X.reshape((X.shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler() # the next lines standardize the images
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# next lines add column of ones to train set for bias
numOfImagesTrainSet = len(X_train)
numOfPixels = len(X_train[0])
onesX = np.ones((numOfImagesTrainSet, numOfPixels + 1))
onesX[:,:-1] = X_train
X_train = onesX

print("done")