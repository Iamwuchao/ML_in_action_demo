import numpy.random as random
import numpy as np

def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b=0;m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*\
                        (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                        ((labelMat[i]*Ei > toler) and \
                                 (alphas[i]>0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T))+ b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i]-C)
                    H = min(C,alphas[j] + alphas[i])
                if L==H : print("L==H");continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - \
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta >= 0 ");continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] =  clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i]*\
                             (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if ( 0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print(" iter：%d i :%d ,pairs changed %d "%\
                      (iter,i,alphaPairsChanged))
                if alphaPairsChanged == 0:iter+=1
                else: iter = 0
                print("iteration number : %d"%iter)
            return b,alphas

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))

    def calcEK(oS,k):
        fXk = float(np.multiply(oS.alphas,))

    def selectJ(i,oS,Ei):
        maxK = -1;maxDeltaE = 0;Ej=0
        oS.eCache[i] = [1,Ei]
        valiEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
        if(len(valiEcacheList))>1:
            for k in valiEcacheList:
                if k == i: continue
                Ek = optStruct.calcEK(oS,k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k;maxDeltaE = deltaE;Ej = Ek
            return maxK,Ej
        else:
            j = selectJrand(i,oS.m)
            Ej = optStruct.calcEK(oS,j)
        return j,Ej

    def updateEk(oS,k):
        Ek = optStruct.calcEK(oS,k)
        oS.eCache[k] = [1,Ek]


def innerL(i,oS):
    Ei = optStruct.calcEk(oS,i)
    if((oS.labelMat[i]*Ei < oS.tol) and (oS.alphas[i] < oS.C) or \
               (oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJrand(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j]+oS.alphas[i])






dataArr,labelArr = loadDataSet("./data/testSet.txt")
b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
print(b)
print(alphas)