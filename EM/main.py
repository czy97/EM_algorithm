from functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import *




#############train process###############

data, label = extractTraindata('data\\train.txt')
devData, devLabel = extractTraindata('data\\dev.txt')
testData = extractTestData('data\\test.csv')
X = np.array(data)
label = np.array(label)
devData = np.array(devData)
devLabel = np.array(devLabel)




#
X_train1 = X[label == 1,:]
X_train2 = X[label == 2,:]
X_dev1 = devData[devLabel == 1,:]
X_dev2 = devData[devLabel == 2,:]


model1 = EM_model(4,X_train1)#-3325.345775383047 -3261.3392486817984
model2 = EM_model(4,X_train2)#0.9775

# model1 = EM_model(4,np.vstack((X_train1,X_dev1))) #-3950.823254322962 -3810.891278677009
# model2 = EM_model(4,np.vstack((X_train2,X_dev2))) #0.97875

model1.train()
model2.train()

#####plot the result##########################

# x1 = X[label == 1, 0]
# y1 = X[label == 1, 1]
# x2 = X[label == 2, 0]
# y2 = X[label == 2, 1]
#
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
# plt.scatter(model1.Mean[:,0], model1.Mean[:,1])
# plt.scatter(model2.Mean[:,0], model2.Mean[:,1])
# # set the range of the axis
# plt.xlim((-2, 4))
# plt.ylim((-2, 3))
#
# plt.show()






acc,labelTmp = calAccuracy(devData,devLabel,model1,model2)
print(acc)

ResLabel = generateTestLabel(testData,model1,model2)
