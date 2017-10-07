import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import multivariate_normal

mat = scipy.io.loadmat('hw1data.mat')
def train(X, Y):
    global mu, cov
    mu = []
    
    Z = np.concatenate((X, Y), axis=1)
    m = 784
    for l in range(10):
        idx = Z[:, m] == l
        zero = Z[idx]
        zero = zero[:, :m]
        mu.append(np.mean(zero, axis=0))
    mu = np.array(mu)
    mux = np.mean(X, axis=0)
    cov = X-mux
    cov = (1.0/X.shape[0])*np.dot(cov.T, cov) + 0.95*np.eye(m)
    # print(np.linalg.det(cov))

def predict(x):
    res = np.zeros((x.shape[0], 10))
    for i in range(10):
        mn = multivariate_normal(mean=mu[i], cov=cov)
        p = mn.pdf(x)
        res[:,i]=p
    return res.T.argmax(0)
    #return ([.pdf(x, ) for i in range(10)])
    

X = (1.0/255)*mat['X']
Y = mat['Y']
accuracy = []
trainingSize = []
for i in range(1,10):
	split = 1000*i
	trainingSize.append(split)
	train_X = X[:split,:]
	train_Y = Y[:split]
	test_X = X[split:,:]
	test_Y = Y[split:]

	train(train_X, train_Y)
	z = predict(test_X)
	err = 0
	for i in range(z.shape[0]):
	    if z[i] != test_Y[i]:
	        err += 1
	# print (1.0 - err * 1.0 / z.shape[0])
	accuracy.append(1.0 - err * 1.0 / z.shape[0])
#print(err)
print(accuracy)
plt.plot(trainingSize, accuracy, 'r-')
plt.xlabel('Training Sample Size')
plt.ylabel('Accuracy')
plt.show()