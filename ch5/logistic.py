from math import exp
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

def loadDataset():
    filename = "./data/testset.txt"
    dataMat = []
    labelmat = []
    with open(filename) as lines:
        for line in lines:
            lineArr = line.strip().split()
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelmat.append(float(lineArr[2]))
    return dataMat,labelmat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatrix,classLabels,numIter=150):
    #dataMatrix = np.mat(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 +j +i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            #print(dataMatrix[randIndex].shape)
            #print(weights.shape)
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

    # dataMatrix = np.mat(dataMatIn)
    # labelMat = np.mat(classLabels).transpose()
    # m,n = np.shape(dataMatrix)
    # alpha = 0.001
    # maxCycles = 500
    # weights = np.ones((n,1))
    # for k in range(maxCycles):
    #     h = sigmoid(dataMatrix*weights)
    #     print("type type")
    #     print(type(h))
    #     print(h.shape)
    #     error = (labelMat - h)
    #     #dataMatrix.transpose()
    #     change = alpha * dataMatrix.transpose()*error#(np.multiply(error,np.multiply(h,(1-h))))
    #     weights = weights + change
    # return weights

def plotBestFit(weights):
    #weights = weights.getA()
    print("###")
    print(type(weights))

    dataMat,labelMat = loadDataset()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c="red",marker='s')
    ax.scatter(xcord2,ycord2,s=30,c="green")
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel("X2")
    plt.show()


dataMat,labelmat = loadDataset()
weights = gradAscent(np.array(dataMat),labelmat)
print(type(weights))
print(weights.shape)
plotBestFit(weights)






