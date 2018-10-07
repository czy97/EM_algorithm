import numpy as np
import math
from functions import *

class EM_model(object):
    def __init__(self,modelNum = 4,X = None,error = 0.001):
        self.X = X
        self.modelNum = modelNum
        self.dataNum,self.featureNum = self.X.shape

        self.error = error

        self.initParam()

    def setError(self,error):
        self.error = error
    def initParam(self):
        self.Mean = np.zeros((self.modelNum,self.featureNum))
        self.Covariance = np.zeros((self.modelNum,self.featureNum,self.featureNum))



        label = Kmeans(self.X, 4)

        for i in range(self.modelNum):
            self.Mean[i] = np.mean(self.X[label==i,:],0)
            tmpItem = self.X[label==i,:] - self.Mean[i].reshape(1,self.featureNum)
            self.Covariance[i] = np.dot(np.transpose(tmpItem),tmpItem)
            # self.Covariance[i] = np.array([[1.0,0],[0,1.0]])

        #Random init----------------------------------
        # num = int(self.dataNum/4)
        # np.random.shuffle(self.X)
        # for i in range(self.modelNum):
        #     self.Mean[i] = np.mean(self.X[i*num:(i+1)*num-1,:],0)
        #     tmpItem = self.X[i*num:(i+1)*num-1,:] - self.Mean[i].reshape(1,self.featureNum)
        #     self.Covariance[i] = np.dot(np.transpose(tmpItem),tmpItem)

        #Random init end------------------------------


        self.C = 1.0 *np.ones((1,self.modelNum))/self.modelNum

        #----------use data to init C-------------------
        # self.C = np.ones((1,self.modelNum))
        # tmpLabel = label.tolist()
        # labelLen = len(tmpLabel)
        # for i in range(self.modelNum):
        #     self.C[0][i] = 1.0 * tmpLabel.count(i)/labelLen


    #calcualte for a single input
    def MultiGaussian(self,mean, CoVariance, X, dim=2):
        # print(X.shape)
        # print(mean.shape)
        # print(CoVariance.shape)

        # print(CoVariance)
        # print(np.linalg.det(CoVariance))

        denominator = np.power(2 * math.pi, 1.0 * dim / 2) * np.sqrt(np.linalg.det(CoVariance))
        numerator = np.exp(-np.dot(np.dot((X - mean), np.linalg.inv(CoVariance)), np.transpose(X - mean)) / 2)
        return numerator / denominator

    #caculate when not only one input
    #the input X should be N by featureDim
    def Gaussian(self,mean, CoVariance, X, dim=2):
        res = []
        # print(X.shape)
        dataNum = X.shape[0]
        for i in range(dataNum):
            tmp = []
            for j in range(self.modelNum):

                # print('mean :{}'.format(mean[j]))
                # print('CoVariance :{}'.format(CoVariance[j]))
                # print('X :{}'.format(X[i]))
                # print('Res :{}'.format(self.MultiGaussian(mean[j], CoVariance[j], X[i])))
                tmp.append(self.MultiGaussian(mean[j], CoVariance[j], X[i]))
            res.append(tmp)

        np_res = np.array(res)
        # print('Gaussian res :{}'.format(np_res))
        return np_res #self.dataNum by self.modelNum(N by M)

    #X:n by 2   C:1 by m
    def calGamma_M_N(self,X,C):
        if(X.shape[1] != self.featureNum):
            X = np.transpose(X)
        if (C.shape != (1,self.modelNum)):
            C = C.reshape((1,self.modelNum))
        # print(X.shape)
        # print(C.shape)
        # print(self.Gaussian(self.Mean, self.Covariance, X))
        numerator = self.Gaussian(self.Mean, self.Covariance, X)*C #n by m
        # print(numerator)
        denominator = np.dot(self.Gaussian(self.Mean, self.Covariance, X),np.transpose(C)) #n by 1

        self.Gamma_M_N = numerator/denominator ##n by m
        return self.Gamma_M_N

    def calGamma_M(self):
        self.Gamma_M = np.sum(self.Gamma_M_N, 0) #(m,)
        return self.Gamma_M

    def calMean(self):

        #self.Gamma_M_N n by m
        #self.X         n by 2
        #self.Gamma_M   m


        #m by 2
        self.Mean = np.dot(np.transpose(self.Gamma_M_N),self.X)/(self.Gamma_M.reshape(self.modelNum,1))
        return self.Mean
    def calCovariance(self):
        # self.X         n by 2
        # self.Gamma_M_N n by m
        # self.Gamma_M   m
        # self.Mean      m by 2

        for i in range(self.modelNum):
            tmp = self.Gamma_M_N[:,i].reshape(self.dataNum,1) #n by 1
            tmp_1 = self.X - self.Mean[i].reshape(1,self.featureNum) # n by 2
            tmp_2 = tmp*tmp_1 #n by 2
            # print(tmp)
            self.Covariance[i] = np.dot(np.transpose(tmp_1),tmp_2)/self.Gamma_M[i]#2 by 2
            # print(self.Covariance[i])
        return self.Covariance

    def cal_C(self):
        self.C = self.Gamma_M/(np.sum(self.Gamma_M))

    def train(self):
        count = 0
        lastLog = 0.0
        currentLog = 1.0
        # while(tmpError > self.error):
        while(np.abs((currentLog-lastLog))>self.error):
            count += 1

            tmpMean = self.Mean
            # print('Mean:{}'.format(tmpMean))
            tmpCovariance = self.Covariance

            lastLog = currentLog

            self.calGamma_M_N(self.X, self.C)
            self.calGamma_M()
            self.cal_C()

            self.calMean()
            self.calCovariance()

            # tmpError_mean = (tmpMean - self.Mean)**2
            # tmpError_mean = sum(sum(tmpError_mean))
            #
            # tmpError_Covariance = (tmpCovariance - self.Covariance)**2
            # tmpError_Covariance = sum(sum(sum(tmpError_Covariance)))
            #
            # tmpError = tmpError_mean + tmpError_Covariance
            # print('iter : {},error : {}'.format(count,tmpError))


            currentLog = np.sum(np.log(np.dot(self.Gaussian(self.Mean, self.Covariance, self.X),self.C.reshape(self.modelNum,1))))

            print('iter : {},currentLog : {}'.format(count,currentLog))

    def predict(self,input):
        if (input.shape[1] != self.featureNum):
            input = np.transpose(input)
        return np.dot(self.Gaussian(self.Mean, self.Covariance, input),self.C.reshape(self.modelNum,1)) #n by 1


    def update(self):
        self.calGamma_M_N(self.X, self.C)
        self.calGamma_M()
        self.cal_C()
        self.calMean()
        self.calCovariance()









