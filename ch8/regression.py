from numpy import *

import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights*xMat)
    if linalg.det(xTx)==0.0:
        print("this matrix is singular, cannot do inverse")
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

abX,abY = loadDataSet('./data/abalone.txt')
# yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
# yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
# yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    xMeans = mean(xMat,0)

def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat




# yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
# yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
# yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
# print(rssError(abY[100:199],yHat01.T))
# print(rssError(abY[100:199],yHat1.T))
# print(rssError(abY[100:199],yHat10.T))

#
# xArr,yArr = loadDataSet('./data/ex0.txt')
# yHat = lwlrTest(xArr,xArr,yArr,k=0.003)
# xMat = mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2)
# plt.show()
'''
'''
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy*ws
# ax.plot(xCopy[:,1],yHat)
# plt.show()
abX,abY = loadDataSet("./data/abalone.txt")
ridgeWeights = ridgeTest(abX,abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()