from numpy import zeros
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import kdtree

class Item(object):
    def __init__(self, coords, data):
        self.coords = coords
        self.data = data

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        item_str = ""
        for i in self.coords:
            item_str+=(" "+str(i))
        item_str+=" "+str(self.data)
        return item_str
        #return 'Item({}, {}, {})'.format(self.coords[0], self.coords[1], self.data)

def file2matrix(filename):
    '''

    :param filename:
    :return: 向量和标签
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def normalization(datingDatamat):
   max_arr = datingDatamat.max(axis=0)
   min_arr = datingDatamat.min(axis=0)
   ranges = max_arr - min_arr
   norDataSet = zeros(datingDatamat.shape)
   m = datingDatamat.shape[0]
   norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
   norDataSet = norDataSet/np.tile(ranges,(m,1))
   return norDataSet

def knn():
    datingDatamat, datinglables = file2matrix("./data/datingTestSet2.txt")
    norDataSet = normalization(datingDatamat)
    print(norDataSet.shape)
    norDataSet =  np.insert(norDataSet,3,datinglables,axis=1)
    np.random.shuffle(norDataSet)
    mylist = norDataSet.tolist()
    sampleCount = len(mylist)
    dividing = int(sampleCount*0.2)
    print(dividing)
    trainSamples = mylist[dividing:]
    testSamples = mylist[0:dividing]

    nodeList = []
    for sample in trainSamples:
        vec = sample[0:3]
        point = Item(coords=vec,data=sample[-1])
        nodeList.append(point)

    myKDTree = kdtree.create(nodeList)
    myKDTree.rebalance()
    return myKDTree,testSamples

def test(myKDtree,testSamples):
    right = 0
    for sample in testSamples:
        point = sample[0:3]
        results = myKDTree.search_knn(point=point, k=6)
        classdict = {}
        for r in results:
            keyNode = r[0]
            key = keyNode.data.data
            if key in classdict:
                classdict[key] += 1
            else:
                classdict[key]=1
        classType = -1
        max = -1
        for (key,val) in classdict.items():
            if max < val:
                max = val
                classType = key
        if classType == sample[-1]:
            right+=1
    print("right")
    print(right)




myKDTree,testSamples = knn()
test(myKDTree,testSamples)




