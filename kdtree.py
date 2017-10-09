import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import io as spio


def build_tree(x_train, x_test, depth):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf.fit(x_train, x_test)
    return clf

if __name__ == '__main__':
    data = spio.loadmat('hw1data.mat', struct_as_record=True)
    xs = data['X']
    ys = data['Y']
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.8, random_state=1)
    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)
    depth = range(1, 40)
    err_list1 = []
    err_list2 = []
    for d in depth:
        clf = build_tree(x_train, y_train, d)
        y_test_hat = clf.predict(x_test)
        y_train_hat = clf.predict(x_train)
        result1 = (y_test_hat == y_test)
        result2 = (y_train_hat == y_train)
        err1 = 1 - np.mean(result1)
        err_list1.append(err1)
        err2 = 1 - np.mean(result2)
        err_list2.append(err2)

    plt.figure(facecolor='w')
    plt.plot(depth, err_list1, 'ro-', lw=2, label='test_error')
    plt.plot(depth, err_list2, 'bo-', lw=2, label='trainin_error')
    plt.xlabel("DEPTH K", fontsize=15)
    plt.ylabel("ERROR RATE", fontsize=15)
    plt.title("Decision Tree(different training size and test size)", fontsize=17)
    plt.legend()
    plt.grid(True)
    plt.show()