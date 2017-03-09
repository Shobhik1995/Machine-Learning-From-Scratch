
# coding: utf-8

# In[15]:

import numpy as np 
import random
#problem being solved: Three input XOR problem
class Perceptron:

    def __init__(self):
        self.alpha = 5
        np.random.seed(42)
        self.w1 = np.random.rand(4,3)
        self.w2=np.random.rand(4,1)

    def sig(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid(self, x, deriv = False):
        if deriv == True:
            return np.multiply(self.sig(x),(1-self.sig(x)))            
        return self.sig(x)

    def train(self, X, Y, iterations):
        print("Training the neural net")
        for iter in range(iterations):
            # Forward propagation
            xw1= X*self.w1
            a1 = self.sigmoid(xw1)
            a1_one=np.ones((a1.shape[0],1))
            a1_bias=np.hstack((a1_one,a1))
    
            xw2=a1_bias*self.w2
            a2 = self.sigmoid(xw2)

            # Backward propagation                        
            small_delta_w2=np.multiply(Y-a2,self.sigmoid(xw2,True)) 
            delta_w2=self.alpha*np.transpose(a1_bias)*small_delta_w2 
            self.w2=self.w2+delta_w2
            
            small_delta_w1=np.multiply(small_delta_w2*np.transpose(self.w2[1:,:]),self.sigmoid(xw1,True))
            delta_w1=self.alpha*np.transpose(X)*small_delta_w1
            self.w1 =self.w1+delta_w1  
            
    def predict(self, X):
        xw1= X*self.w1
        a1 = self.sigmoid(xw1)
        a1_one=np.ones((a1.shape[0],1))
        a1_bias=np.hstack((a1_one,a1))
        xw2=a1_bias*self.w2
        a2 = self.sigmoid(xw2)
        a2=np.piecewise(a2, [a2< 0.5,a2>= 0.5], [0, 1])
        return a2
        

if __name__ =='__main__':
    p = Perceptron()
    X=np.matrix('1,0,0,1;1,0,1,0;1,0,1,1;1,1,0,0;1,1,0,1;1,1,1,0;1,1,1,1')
    
    y=np.matrix('0;0;0;1;1;1;1')

    p.train(X, y, 10000)

    
    for i in X:	
        print ("Predicting XOR values for the input :")
        print(i)
        print("Prediction :")
        print(p.predict(i))






