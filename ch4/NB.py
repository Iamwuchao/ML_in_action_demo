import os

#在0类别中，每个词出现的概率
pw0 = {}

#在1类别中每个词出现的概率
pw1 = {}

#0类别出现的概率
pc0 = 0

#1类别出现的概率
pc1 = 0

word2index = []

testdata_list =[]
spam_dir = "./data/email/spam/"
ham_dir = "./data/email/ham/"

def trainNB():
    '''
    :param dirt: 文件目录
    :return: 处理之后的数据
      0 垃圾邮件 spam
    '''
    global  pc1,pc0,word2index,pw1,pw0,testdata_list

    spam_dir = "./data/email/spam/"
    ham_dir = "./data/email/ham/"
    fileNames0 = os.listdir(spam_dir)
    fileNames1 = os.listdir(ham_dir)

    dividing_ham = int(0.2 * len(fileNames1))

    dividing_spam = int(0.2 * len(fileNames0))

    #获取测试数据
    ham_test_list = fileNames1[0:dividing_ham]
    spam_test_list = fileNames0[0:dividing_spam]
    for filename in ham_test_list:
        with open(ham_dir+filename) as lines:
            words = []
            for line in lines:
                line = line.strip()
                words = words + line.split(" ")
            testdata_list.append((words,1))

    for filename in spam_test_list:
        with open(spam_dir+filename) as lines:
            words = []
            for line in lines:
                line = line.strip()
                words = words + line.split(" ")
            testdata_list.append((words,0))


    #训练数据
    fileNames0 = fileNames0[dividing_ham:]
    fileNames1 = fileNames1[dividing_spam:]


    doc_count = len(fileNames0) + len(fileNames1)

    pc0 = len(fileNames0)/doc_count
    pc1 = len(fileNames1)/doc_count
    wordset = set()

    #统计类别0
    word_dict0,word_sum_0 = statistics(spam_dir,fileNames0)
    word_dict1,word_sum_1 = statistics(ham_dir,fileNames1)

    for (k,v) in word_dict0.items():
        wordset.add(k)
        pw0[k] = v*1.0/word_sum_0

    for (k,v) in word_dict1.items():
        wordset.add(k)
        pw1[k] = v*1.0/word_sum_1
    word2index = list(wordset)




def statistics(path,fileNames):
    word_dict = {}
    word_sum=0
    for filename in fileNames:
        print(path+" "+filename)
        with open(path+filename) as lines:
            for line in lines:
                line = line.strip()
                if line:
                    wordarray = line.split(" ")
                    for word in wordarray:
                        if word in word_dict:
                            word_dict[word]+=1
                        else:
                            word_dict[word]=1
                        word_sum+=1
    return word_dict,word_sum



def doctoworddict(filename):
    global word2index
    worddict = {}
    with open(filename,encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if line:
                wordarray = line.split(" ")
                for word in wordarray:
                    if word in worddict:
                        worddict[word]+=1
                    else:
                        worddict[word] = 1

    return worddict



def classfyNB(docdict):
    p0 = pc0
    p1=  pc1
    for (k,v) in docdict.items():
        if k in pw0:
            p0 = p0 + v*pw0[k]
        if k in pw1:
            p1 = p1 + v*pw1[k]
    if p1 >= p0:
        return 1
    else:
        return 0



def test():
    right = 0
    count = len(testdata_list)
    samplelist = []
    for sample in testdata_list:
        words = sample[0]
        docdict = {}
        for word in words:
            if word in docdict:
                docdict[word] += 1
            else:
                docdict[word] = 1
        samplelist.append((docdict,sample[1]))
    for sample in samplelist:
        tag = classfyNB(sample[0])
        if tag == sample[1]:
            right += 1
    print("count : "+str(count))
    print("right : "+str(right))


trainNB()
test()

