# coding uft-8
import numpy as np
import math
import operator


class Logistic(object):
    def __init__(self, filename, maxCycle):
        self.filename = filename
        self.maxCycle = maxCycle

    def loadDateSet(self):
        data = np.loadtxt(self.filename)
        dataMat = data[:, 0: -1]
        classLabels = data[:, -1]
        # dataMat = np.insert(dataMat, 0, 1, axis=1)
        return dataMat, classLabels

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradDescent(self, dataMat, classLabels):
        dataMatrix = np.mat(dataMat)  
        labelMat = np.mat(classLabels).transpose()
        m, n = np.shape(dataMatrix)
        weights = np.ones((n, 1))  
        alpha = 0.001    

        for i in range(self.maxCycle):
            h = self.sigmoid(dataMatrix * weights)  
            error = labelMat - h
            weights = weights + alpha * dataMatrix.transpose() * error 
        return np.round(weights, 4)

    def classifyVector(self, inX, weights):
        prob = self.sigmoid(sum(np.dot(inX, weights)))
        return prob


class KNN(object):
    def __init__(self, filename):
        self.filename = filename

    def file2matrix(self):
        fr = open(self.filename)
        arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)
        returnMat = np.zeros((numberOfLines, 320))

        classLabelVector = []
        index = 0  
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split(' ')
            returnMat[index, :] = listFromLine[0:320] 

            if listFromLine[-1] == '0':
                classLabelVector.append(0)
            elif listFromLine[-1] == '1':
                classLabelVector.append(1)     

            index += 1  
        return returnMat, classLabelVector

    def classifyKNN(self, inputData, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]  

        diffMat = np.tile(inputData, (dataSetSize, 1)) - dataSet  
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5

        sortedDistIndices = distances.argsort()

        classCount = {}
        for i in range(k):
            voteLabel = labels[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1        

            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)       
            return sortedClassCount[0][0]

    def classifyVector(self, inputData):
        datingDataMat, datingLabels = self.file2matrix() 
        classifierResult = self.classifyKNN(inputData, datingDataMat, datingLabels, 3)

        return classifierResult


if __name__ == '__main__':
    with open("data_3.txt") as f:
        inputData = f.readline().split(' ')
        inputData = np.array(inputData)
        inputData = inputData.astype('float64')

    bayes = Bayes("data_2.txt")

    result = bayes.classifyVector(inputData)
    print(result)
