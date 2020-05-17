# Amnon Ophir 302445804, Ross Bolotin 310918610

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np


def e_func(t, w, x):
    ln_y_mat = ln_y(np.transpose(w), x)
    return - np.matmul(np.transpose(t), np.transpose(ln_y_mat)).sum()


def ln_y(w, x):
    numerator = np.matmul(w, x)
    numerator = np.exp(numerator)
    denominator = numerator.sum(axis=0)
    denominator = np.expand_dims(denominator, axis=0)
    denominator = np.repeat(a=denominator, repeats=10, axis=0)
    return np.log(numerator) - np.log(denominator)


def classify(w, x):
    ln_y_matrix = ln_y(np.transpose(w), np.transpose(x))
    ln_y_matrix = np.transpose(ln_y_matrix)
    classifications = np.zeros_like(ln_y_matrix)
    classifications[np.arange(len(ln_y_matrix)), ln_y_matrix.argmax(1)] = 1
    return classifications


def main():
    mnist = fetch_openml('mnist_784')
    X = mnist['data'].astype('float64')
    y = mnist['target']
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]  # The next line flattens the vector into 1D array of size 784
    X = X.reshape((X.shape[0], -1))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler() # the next lines standardize the images
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # next lines turn y_train to one-hot
    y_train = y_train.astype(int)
    y_one_hot = np.zeros((y_train.size, 10))
    y_one_hot[np.arange(y_train.size), y_train] = 1
    y_train = y_one_hot.astype(object)

    # next lines add column of ones to train set for bias
    num_of_images_train_set = len(x_train)
    num_of_pixels = len(x_train[0])
    ones_x = np.ones((num_of_images_train_set, num_of_pixels + 1))
    ones_x[:,:-1] = x_train
    x_train = ones_x

    w = np.random.rand(num_of_pixels + 1, 10)  # random weights matrix

    e = e_func(y_train, w, np.transpose(x_train))

    test_set_classifications = classify(w, x_train)

    print("done")


if __name__ == "__main__":
    # execute only if run as a script
    main()