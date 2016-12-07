from numpy import *
import operator
import numpy as np
#Create data set function;
from numpy import *
from numpy import array
import operator

#Overflown file2matrix to limit read entries from testdata
def file2matrix(filename, limit=None):
    fr = open(filename)
    if limit==None:
        numberOfLines = len(fr.readlines())
    else:
        numberOfLines = limit
    returnMat = zeros((numberOfLines,11))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[0:11]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        if index==limit:
            break
    return returnMat, classLabelVector

#AdaBoost Classifier

# def adaBoostTrainDS(dataArr,classLabels,numIt=40):
#     weakClassArr = []
#     m = shape(dataArr)[0]
#     D = mat(ones((m,1))/m) 
#     aggClassEst = mat(zeros((m,1)))
#     for i in range(numIt):
#     bestStump,error,classEst = buildStump(dataArr,classLabels,D)
#     print "D:",D.T
#     alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
#         bestStump['alpha'] = alpha 
#         weakClassArr.append(bestStump) 
#         print "classEst: ",classEst.T
#         expon = multiply(-1*alpha*mat(classLabels).T,classEst)
#         D = multiply(D,exp(expon)) 
#         D = D/D.sum() 
#         aggClassEst += alpha*classEst 
#         print "aggClassEst: ",aggClassEst.T 
#         aggErrors = multiply(sign(aggClassEst) != 
#         mat(classLabels).T,ones((m,1))) 
#         errorRate = aggErrors.sum()/m 
#         print "total error: ",errorRate,"\n"
#         if errorRate == 0.0: break
#     return weakClassArr
