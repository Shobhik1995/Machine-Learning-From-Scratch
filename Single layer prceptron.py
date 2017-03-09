import numpy as np
import random

#single layer perceptron with two input neurons and a bias in the input layer
#and one output neuron in the output layer
#problem being solved in this code is that of boolean AND function for two inputs 
class perceptron:
    #radom initialization of the weights of the slp
    def __init__(self):
        self.W=np.random.rand(3,1)
    #function for training    
    def train(self,X,Y,iter):
        print("training the net")
        for i in range(0,iter):
            out=self.predict(X)
            err=Y-out
            self.W=self.W + 1*np.transpose(X)*err    
        print("final weights of the net")
        print(self.W)
        return self.W
    
    #function for predicting
    def predict(self,x):
        out=x*self.W
        out=np.piecewise(out, [out< 0,out>= 0], [0, 1])
        return out




X=np.matrix('1,0,0;1,1,0;1,0,1;1,1,1')
Y=np.matrix('0;0;0;1')   
stuff=perceptron()

stuff.train(X,Y,100)
print("predicting output for the input")
print(X[:,1:3])
print("predictions :" )
print(stuff.predict(X))
