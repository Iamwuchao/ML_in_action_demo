from numpy import *
import numpy as np
# import matplotlib.pyplot as plt
class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    print(type(dataSet))
    print(dataSet.shape)
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;bestIndex=0;bestValue=0
    for featIndex in range(n-1):
        #print("&*&*&*&*&*&* " + str(featIndex))
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):

            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #print("^^^^^^^^^^^")
            if (shape(mat0)[0]<tolN) or(shape(mat1)[0] < tolN):continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
        #print("!!!!!!!!!!!!")
    if(S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0]<tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue



def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    print("feat "+str(feat))
    print("val "+str(val))
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

# fig = plt.figure()
# ax = fig.add_subplot(111)

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['right']+tree['left'])/2.0

def prune(tree,testData):
    if shape(testData)[0]==0: return getMean(tree)
    if(isTree(tree['right'])) or isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): tree['right'] = prune(tree['left'],lSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2))+\
            sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m,n = shape(dataSet)
    X= mat(ones((m,n)));Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1];Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('this matrix is singular,cannot do inverse, try '+\
                        'try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat



# myMat2 = mat(loadDataSet('./data/exp2.txt'))
# tree = createTree(myMat2,modelLeaf,modelErr,(1,10))
#
# print(tree)
# myDat = loadDataSet("./data/ex2.txt")
# myMat = mat(myDat)
# print("&&&&&&")
# print(myMat.shape)
# retTree = createTree(myMat,ops=(0,1))
# print(retTree)
# print("***********************************")
# myDatTest = loadDataSet('./data/ex2test.txt')
# myMatTest = mat(myDatTest)
# tree = prune(retTree,myMatTest)
# print(tree)

trainMat = mat(loadDataSet("./data/bikeSpeedVsIq_train.txt"))
testMat = mat(loadDataSet("./data/bikeSpeedVsIq_test.txt"))
# myTree = createTree(trainMat,ops=(1,20))
# yHat = createForeCast(myTree,testMat[:,0])
# result = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
# print(result)

mytree = createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat = createForeCast(mytree,testMat[:,0],modelTreeEval)
result = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print(result)