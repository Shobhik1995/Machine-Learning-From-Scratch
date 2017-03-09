#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:14:38 2017

@author: shobhik
"""
import math
import numpy as np
import random


# Function for generating a random line that takes in two arguments,
#and for generating a random dataset
def generate_data():
    np.random.seed(7)
    # 'weight' has the coefficients of the line y=ax1+bx2+c
    weight=np.random.rand(3,1)
    x_main0=np.ones((100,1))
    x_main1=np.transpose(np.matrix([random.randint(-100,100) for r in range(100)]))
    x_main2=np.transpose(np.matrix([random.randint(-100,100) for r in range(100)]))
    
    x=np.hstack((x_main0,x_main1))
    x=np.hstack((x,x_main2))
    y=x*weight
    return x[0:59],x[60:79],x[80:99],y[0:59],y[60:79],y[80:99],weight


#creating a linear regression class
class linear_reg():
    def __init__(self,max_iter):
        self.max_iter=max_iter
    #function for training the linear regression object    
    def fit(self,x_train,y_train,eta):
        self.dr=eta
        self.x_train=x_train
        self.y_train=y_train
        np.random.seed(10)
        self.weight=np.random.rand(3,1)
        #print("fitting to the training data\n")
        for i in range(self.max_iter):
            loss=self.x_train*self.weight-self.y_train
            grad=np.transpose(self.x_train)*loss
            n=x_train.shape[0]
            self.weight=self.weight-(self.dr*grad)/n
        return self.weight
    #function for predicting the outcome for a given set of datapoints.   
    def predict(self,x):
        pred=x*self.weight
        return pred
        
        
        
#generating the training, validation and test datasets            
x_train,x_val,x_test,y_train,y_val,y_test,w=generate_data()

lr=linear_reg(1000)


flag=True
eta=10;
prev_eta=eta
prev_error=np.inf

# the loop iterates over various possible values of
#the hyperparameter (the learning rate in gradient descent, 
    #starting from a value of 10 and decreasing in each iteration)
#and figures out the optimum value for the parameter
#based on the total error calculated over the validation set
while(flag):
    
    weights=lr.fit(x_train,y_train,eta)
    pred=lr.predict(x_val)
    curr_error=np.sum(np.multiply(pred-y_val,pred-y_val))
    if(curr_error<=prev_error or math.isnan(curr_error) ):
        flag=True
        if(not math.isnan(curr_error)):
            prev_eta=eta
            prev_error=curr_error
        eta=eta/10
        
    else:
        flag=False
        
print("the optimum value for the learning parameter is : "+str(prev_eta))    
print()

#training with the optimum valu of the learning rate
weights=lr.fit(x_train,y_train,prev_eta)
print('value of the weights obtained after fitting to training data\n')
print(lr.weight)
print()
#predicting y labels for the test data 
pred=lr.predict(x_test)
print('the column on the left shows the predicted values and the column on the right shows the actual y labels for the test data\n')
print(np.hstack((pred,y_test)))
#calculating the error for the predicted value 
#of the y labels in the test data
error=np.multiply(pred-y_test,pred-y_test)
print()
print('Error for this prediction')
print()
print(np.sum(error))

