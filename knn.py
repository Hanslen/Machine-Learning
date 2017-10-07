import numpy as np
import scipy.io
import math
import operator

mat = scipy.io.loadmat('hw1data.mat')

def euclideanDistance(p1, p2, length):
	distance = 0
	for x in range(length):
		distance += pow(p1[x]-p2[x],2)
	return math.sqrt(distance)

def getNeighbor(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response]+=1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

X = (1.0/255)*mat['X']
Y = mat['Y']
split = 9000
train_X = X[:split,:]
train_Y = Y[:split]
test_X = X[split:,:]
test_Y = Y[split:]

train_X = np.concatenate((train_X, train_Y), axis=1)
test_X = np.concatenate((test_X, test_Y), axis=1)
predictions = []
k = 5
for x in range(len(test_X)):
	neighbors = getNeighbor(train_X, test_X[x],k)
	result = getResponse(neighbors)
	predictions.append(result)
print(predictions)
print(getAccuracy(test_X,predictions))
