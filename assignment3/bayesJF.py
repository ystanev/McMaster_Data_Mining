'''
Created on Nov 1, 2016

@author: Jeff Fortuna
'''
from numpy import *
from collections import namedtuple
import numpy as np

BayesTrainResults = namedtuple('BayesTrainingResults', 'probCatC0 probCatC1 statsC0 statsC1 PC0 PC1')




def trainNB0(trainDataCategorical, numCategories, trainDataNumeric, trainClass):   
    trainClass = np.array(trainClass)
    
    # class 0 categorical data
    class0trainCat = trainDataCategorical[nonzero(trainClass == 0)]
    # class 1 categorical data
    class1trainCat = trainDataCategorical[nonzero(trainClass == 1)]
    
    
    # class 0 numeric data
    class0trainNum = trainDataNumeric[nonzero(trainClass == 0)]
    # class 1 numeric data
    class1trainNum = trainDataNumeric[nonzero(trainClass == 1)]
   
    probCategoryClass0 = zeros((class0trainCat.shape[1], max(numCategories)), dtype = int) 
    probCategoryClass1 = zeros((class1trainCat.shape[1], max(numCategories)), dtype = int)

    # count up the number of examples of each category for class 0
    for i in range(class0trainCat.shape[0]):
        for j in range(class0trainCat.shape[1]):
            for k in range(numCategories[j]):
                if class0trainCat[i,j] == k:
                    probCategoryClass0[j,k] += 1;

    # calculate categorical class 0 probabilities
    probCategoryClass0 =  divide(probCategoryClass0, float(class0trainCat.shape[0]))

    # count up the number of examples of each category for class 1
    for i in range(class1trainCat.shape[0]):
        for j in range(class1trainCat.shape[1]):
            for k in range(numCategories[j]):
                if class1trainCat[i,j] == k:
                    probCategoryClass1[j,k] += 1;

    # calculate categorical class 1 probabilities                    
    probCategoryClass1 =  divide(probCategoryClass1, float(class1trainCat.shape[0]))

    # calculate mean and standard deviation for both classes for numeric data 
    meanClass0 = mean(class0trainNum, axis = 0)
    meanClass1 = mean(class1trainNum, axis = 0)
    stdClass0 = std(class0trainNum, axis = 0, ddof = 1)
    stdClass1 = std(class1trainNum, axis = 0, ddof = 1)

    statsC0 = vstack((meanClass0, stdClass0))
    statsC1 = vstack((meanClass1, stdClass1))

    # calculate the probability of class 0 and class 1
    numDataPoints = trainClass.shape[0] ##Change to length op since it is a vector and not a matrix.
    numC0DataPoints = sum(trainClass == 0)
    numC1DataPoints = sum(trainClass == 1)

    PC0 = numC0DataPoints / float(numDataPoints)
    PC1 = numC1DataPoints / float(numDataPoints)

    result = BayesTrainResults(probCategoryClass0, probCategoryClass1, statsC0, statsC1, PC0, PC1)
    
    return result


def classifyNB(inXCat, inXNum, trainResult):

    # if the categorical input is a scalar it is not an array so shape will not work
    # therefore turn the input into an array
    if inXCat.shape == ():
        inXCat = array([inXCat])

    # if the numeric input is a scalar it is not an array so shape will not work
    # therefore turn the input into an array
    if inXNum.shape == ():
        inXNum = array([inXNum])


    PAttCatC0 = empty(inXCat.shape[0])
    PAttCatC1 = empty(inXCat.shape[0])

    # look up the probability of categorical attributes
    for i in range(inXCat.shape[0]):
        PAttCatC0[i] = trainResult.probCatC0[i, inXCat[i]]
        PAttCatC1[i] = trainResult.probCatC1[i, inXCat[i]]

    C0LogCat = sum(log(PAttCatC0))
    C1LogCat = sum(log(PAttCatC1))

   

    PAttNumC0 = empty(inXNum.shape[0])
    PAttNumC1 = empty(inXNum.shape[0])

    # calculate the probability of numeric attributes
    for i in range(inXNum.shape[0]):
        PAttNumC0[i] = (1.0/(sqrt(2.0*pi)*trainResult.statsC0[1,i]))*exp( -((inXNum[i] - trainResult.statsC0[0,i])**2)/(2.0*(trainResult.statsC0[1,i])**2.0) ) 
        PAttNumC1[i] = (1.0/(sqrt(2.0*pi)*trainResult.statsC1[1,i]))*exp( -((inXNum[i] - trainResult.statsC1[0,i])**2.0)/(2.0*(trainResult.statsC1[1,i])**2.0) ) 

    C0LogNum = sum(log(PAttNumC0))
    C1LogNum = sum(log(PAttNumC1))

    # calculate the overall probability of each class
    resultC0 = C0LogCat + C0LogNum + log(trainResult.PC0)
    resultC1 = C1LogCat + C1LogNum + log(trainResult.PC1)
    #print "%r > %r" %(resultC0, resultC1)
    # return the resulting class or -1 if there is a tie
    if resultC0 == resultC1:
        return -1
    elif resultC0 > resultC1:
        return 0
    else:
        return 1

def read_file(filename, linesize, a, b, limit=None):
    fr = open(filename)
    if limit==None:
        numberOfLines = len(fr.readlines())
    else:
        numberOfLines = limit
    returnMat = zeros((numberOfLines,linesize))
    class_name = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[a:b]
        class_name.append(int(listFromLine[-1]))
        index += 1
        if index==limit:
            break
    return returnMat, class_name

def titanic_test(): 
     test_nominal_matrix, tVect1 = read_file("Titanic Test.txt", 2, 1, 3)
     test_numeric_matrix, tVect2 = read_file("Titanic Test.txt", 1, 0, 1)
     m = test_nominal_matrix.shape[0]
     errorCount = 0.0
     for i in range (m):
         classifierResult =  classifyNB(test_nominal_matrix[i], test_numeric_matrix[i], trainResult)
         print "Test returned: %d, actual value: %d" % (classifierResult, tVect1[i])
         if (classifierResult != tVect1[i]): errorCount += 1.0
     print "Error Rate: %f" % (errorCount/float(m) * 100) 
     print "Accuracy Rate: %f" % (100 - (errorCount/float(m) * 100))
     
##Start Main Loop
numCategories = [2,3] #2 gender cases 3 accomadation cases
catMatrix, catVector = read_file("Titanic Training.txt" ,2 , 1, 3)   
numMatrix, numVector = read_file("Titanic Training.txt",1,0,1) 
classMatrix, classVector = read_file("Titanic Training.txt", 1, 3, 4)

trainResult = trainNB0(catMatrix, numCategories, numMatrix, numVector)




i=0
titanic_test()