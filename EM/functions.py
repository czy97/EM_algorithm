import numpy as np
import math

#extract data from .txt to list format
def extractTraindata(filename):
    data = []
    label = []
    with open(filename) as txtData:
        lines = txtData.readlines()
        for line in lines:
            tmpData = []
            lineData = line.strip().split(' ')
            tmpData.append(float(lineData[0]))
            tmpData.append(float(lineData[1]))
            data.append(tmpData)
            label.append(int(lineData[2]))
    return data, label
#extract data from .csv to numpy array format
def extractTestData(filename):
    import pandas as pd
    df = pd.read_csv(filename)

    npData = np.array(df)
    data = npData[:,1:]

    return data

#plot the data with label
def plotdata(data,label):
    np_data = np.array(data)
    np_label = np.array(label)
    import matplotlib.pyplot as plt

    x1 = np_data[np_label == 1, 0]
    y1 = np_data[np_label == 1, 1]
    x2 = np_data[np_label == 2, 0]
    y2 = np_data[np_label == 2, 1]

    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    # set the range of the axis
    plt.xlim((-2, 4))
    plt.ylim((-2, 3))

    plt.show()

#plot the test data
def plotTestData(data):

    import matplotlib.pyplot as plt

    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y)
    plt.xlim((-2, 4))
    plt.ylim((-2, 3))

    plt.show()

def gaussian(mean,variance,x):
    return np.exp(-(x - mean) ** 2 /(2 * variance))/(np.sqrt(2*variance*math.pi))
def MultiGaussian(mean,CoVariance,X,dim = 2):
    denominator =  np.power(2*math.pi,1.0*dim/2)*np.sqrt(np.linalg.det(CoVariance))
    numerator = np.exp(-np.dot(np.dot((X-mean),np.linalg.inv(CoVariance)),np.transpose(X-mean))/2)

    return numerator/denominator

def Kmeans(inputData,K):

    dataNum,featureNum = inputData.shape

    tmp = np.arange(8)
    np.random.shuffle(tmp)

    centers = inputData[0:K,:]
    distance = np.zeros((dataNum,K))

    ######
    # centers[0] = np.array([0.5,0.5])
    # centers[1] = np.array([1.0, -1.0])
    # centers[2] = np.array([2.0, 2.0])
    # centers[3] = np.array([3.0, 0.0])


    ######

    dis = 5.0

    while(dis > 0.0001):
        tmpCenters = centers

        for i in range(K):
            distance[:,i] = np.sqrt(np.sum((inputData - centers[i,:].reshape(1,featureNum)) ** 2, 1))

        mindistance = distance <= np.min(distance, 1).reshape(dataNum, 1)
        label = (np.argwhere( mindistance == True))[:,1]

        for i in range(K):
            centers[i,:] = np.mean(inputData[label==i,:],0).reshape(1,featureNum)

        dis = (tmpCenters - centers) ** 2
        dis = sum(sum(dis))

    return label


from pandas.core.frame import DataFrame
def storeCsv(filename,input):

    num = input.shape[0]

    idList = range(num)
    resList = input.tolist()
    #判断训练集和测试集的输入是否加了bias，如果加了，则不把bias写入csv



    res = {'id':idList,'classes':resList}
    data = DataFrame(res,columns=['id','classes'])
    data.to_csv(filename, encoding='gbk',index=False)


def calAccuracy(inputaData,label,model1,model2):
    inputaData = np.array(inputaData)
    label = np.array(label)
    res = np.zeros(inputaData.shape)
    res[:, 0] = model1.predict(inputaData).reshape(inputaData.shape[0])
    res[:, 1] = model2.predict(inputaData).reshape(inputaData.shape[0])

    max_value = res >= np.max(res, 1).reshape(inputaData.shape[0], 1)
    ResLabel = (np.argwhere(max_value == True))[:, 1]

    ResLabel += 1
    count = ((list(ResLabel==label)).count(True))

    return 1.0*count/label.shape[0],ResLabel

def trainWithValid(inputaData,label,model1,model2,initIter = 15,postIter = 50):
    for i in range(initIter):
        model1.update()
        model2.update()

    currentAccuracy,label =  calAccuracy(inputaData,label,model1,model2)
    maxAccuracy = currentAccuracy
    bestLabel = label

    for k in range(postIter):
        currentAccuracy, label = calAccuracy(inputaData, label, model1, model2)

        if(currentAccuracy>maxAccuracy):
            maxAccuracy = currentAccuracy
            bestLabel = label

    return maxAccuracy,bestLabel

def generateTestLabel(inputaData,model1,model2):
    inputaData = np.array(inputaData)

    res = np.zeros(inputaData.shape)
    res[:, 0] = model1.predict(inputaData).reshape(inputaData.shape[0])
    res[:, 1] = model2.predict(inputaData).reshape(inputaData.shape[0])

    max_value = res >= np.max(res, 1).reshape(inputaData.shape[0], 1)
    ResLabel = (np.argwhere(max_value == True))[:, 1]

    ResLabel += 1


    return ResLabel