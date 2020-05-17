# Amnon Ophir 302445804, Ross Bolotin 310918610

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np
from scipy.special import logsumexp

np.random.seed(777)
eta = 0.01


def e_func(t, w, x):
    ln_y_mat = ln_y(np.transpose(w), x)
    return - np.matmul(np.transpose(t), np.transpose(ln_y_mat)).sum()


def ln_y(w, x):
    numerator = np.matmul(w, x)
    numerator = numerator #np.exp(numerator)
    denominator = logsumexp(numerator, axis=0)#numerator.sum(axis=0)
    denominator = np.expand_dims(denominator, axis=0)
    denominator = np.repeat(a=denominator, repeats=10, axis=0)
    return numerator - denominator #enp.log(numerator) - np.log(denominator)


def classify(w, x):
    ln_y_matrix = ln_y(np.transpose(w), np.transpose(x))
    ln_y_matrix = np.transpose(ln_y_matrix)
    classifications = np.zeros_like(ln_y_matrix)
    classifications[np.arange(len(ln_y_matrix)), ln_y_matrix.argmax(1)] = 1
    return np.transpose(classifications)


def accuracy_calc(classifications, y):
    accuracy = classifications + y
    return (accuracy == 2).sum() / len(classifications)


def bias_set(x):
    num_of_images = len(x)
    num_of_pixels = len(x[0])
    ones_x = np.ones((num_of_images, num_of_pixels + 1))
    ones_x[:, :-1] = x
    return ones_x


def one_hot(y):
    y = y.astype(int)
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot.astype(object)


def grad_e(x, y, t):
    return np.matmul(np.transpose(x), np.transpose(y) - t)


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

    # next lines turn y_train and y_test to one-hot
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    # next lines add column of ones to train set and test set for bias
    x_train = bias_set(x_train)
    x_test = bias_set(x_test)

    w = np.random.rand(len(x_test[0]), 10)  # random weights matrix

    e = e_func(y_train, w, np.transpose(x_train))
    print("Initial error is ", e)
    test_set_classifications = classify(w, x_test)
    accuracy = accuracy_calc(np.transpose(test_set_classifications), y_test)
    print("Initial accuracy is ", accuracy)
    prev_accuracy = 0
    i = 0
    while accuracy - prev_accuracy > 0.001:
        i = i + 1
        print("Loop number ", i)
        prev_accuracy = accuracy
        train_set_classifications = classify(w, x_train)
        gradient = grad_e(x_train, train_set_classifications, y_train)
        gradient = gradient.astype(float)
        w = w - eta * gradient
        test_set_classifications = classify(w, x_test)
        accuracy = accuracy_calc(np.transpose(test_set_classifications), y_test)
        print("Accuracy is ", accuracy)

    train_set_classifications = classify(w, x_train)
    train_accuracy = accuracy_calc(np.transpose(train_set_classifications), y_train)
    e = e_func(y_train, w, np.transpose(x_train))
    print("Done, final accuracy on test set is ", accuracy)
    print("Final accuracy on train set is ", train_accuracy)
    print("Final error is ", e)
    print("Bye!")


if __name__ == "__main__":
    # execute only if run as a script
    main()