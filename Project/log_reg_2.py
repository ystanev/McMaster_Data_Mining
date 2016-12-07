#Import Modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import time
from sklearn.metrics import roc_curve, auc #Used to plop a ROC

start = time.time() #Starts timer

# load dataset
dta = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)

#Prepare Data for Logistic Regression
# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  dta, return_type="dataframe")
#print X.columns

# fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# flatten y into a 1-D array
y = np.ravel(y)

#Logistic Regression
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
print 'Accuracy of training dataset is: %r' %(model.score(X, y)*100) + ' %'

# what percentage had affairs?
print 'Percentage of women who had an affair: %r' %(y.mean()*100) + ' %'

#Model Evaluation Using a Validation Set
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# predict class labels for the test set
predicted = model2.predict(X_test)
print predicted

# generate class probabilities
# probs = model2.predict_proba(X_test)
# print "Probability of each class is: %r" %probs

# generate evaluation metrics
#print metrics.accuracy_score(y_test, predicted)
#print metrics.roc_auc_score(y_test, probs[:, 1])

#Confusion matrix
print "Confusion Matrix: \n" 
print metrics.confusion_matrix(y_test, predicted)
#print metrics.classification_report(y_test, predicted)

#Model Evaluation Using Cross-Validation
# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)#Does cross-validatin of training and test set
print "Cross-Validation Scores: \n"
print scores
print "Cross-Validation Mean: %r" %(scores.mean()*100) +" %"
#print scores.mean()


#ROC Curve
# x = # false_positive_rate
# y = # true_positive_rate 
# 
# # This is the ROC curve
# plt.plot(x,y)
# plt.show() 
# 
# # This is the AUC
# auc = np.trapz(y,x)

end = time.time() - start #Calcuates total run time
print 'The program run time is: %r seconds' %end