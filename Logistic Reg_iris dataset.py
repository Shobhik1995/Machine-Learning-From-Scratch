
# coding: utf-8

# In[985]:

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# In[986]:

def generate_data():
    iris= load_iris()
    data=iris.data
    data=np.matrix(data)
    target=iris.target
    target=np.matrix(target)
    target=np.transpose(target)
    data=np.hstack((data,target))
    data=shuffle(data)
    label=data[:,data.shape[1]-1]
    features=data[:,0:data.shape[1]-1]
    features=preprocessing.normalize(features)
    return features[0:90],label[0:90],features[90:120],label[90:120],features[120:150],label[120:150]    


# In[987]:

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[988]:

def costFunction(weight,x,y,lamb):
    m=y.shape[0]
    one=np.matrix(np.ones(y.shape))
    J=(-1/m)*((np.transpose(y)*np.log(sigmoid(x*weight)))+(np.transpose((one-y))*np.log(one-sigmoid(x*weight))))+((lamb/(2*m))*(np.transpose(weight[1:,:])*weight[1:,:]))
    sig=sigmoid(x*weight)
    grad=np.transpose(x)*(sig-y)
    grad=grad/m
    grad[1:,:]=grad[1:,:]+(lamb/m)*weight[1:,:]
    return J,grad


# In[989]:

def fit(x,y,iters,eta,lamb):
    one=np.matrix(np.ones((x.shape[0],1)))
    x=np.hstack((one,x))
    weight=np.matrix(np.ones((x.shape[1],1)))
    for i in range(iters):
        J,grad=costFunction(weight,x,y,lamb)
        weight=weight-eta*grad
    return weight    


# In[990]:

def predict(x,weight):
    one=np.matrix(np.ones((x.shape[0],1)))
    x=np.hstack((one,x))
    return sigmoid(x*weight)


# In[991]:

x_train,y_train,x_val,y_val,x_test,y_test=generate_data()


# In[992]:

# adopting a one-vs-rest classification technique thus training three different logistic regressors 
# making the y training labels for the three different regressors

y_tr0=[]
y_tr1=[]
y_tr2=[]

for i in range(y_train.shape[0]):
    if(y_train[i] == 0):
        y_tr0.append(1)
        y_tr1.append(0)
        y_tr2.append(0)
    elif(y_train[i] == 1):
        y_tr0.append(0)
        y_tr1.append(1)
        y_tr2.append(0)
    else:
        y_tr0.append(0)
        y_tr1.append(0)
        y_tr2.append(1)
        
y_tr0=np.transpose(np.matrix(y_tr0))
y_tr1=np.transpose(np.matrix(y_tr1))
y_tr2=np.transpose(np.matrix(y_tr2)) 
        
        


# In[993]:

#making y vaidation labels for the three clasifiers
#the values of the hyperparameters was chose by repeatedly improving the performance of the models on the validation set

y_val0=[]
y_val1=[]
y_val2=[]

for i in range(y_val.shape[0]):
    if(y_val[i] == 0):
        y_val0.append(1)
        y_val1.append(0)
        y_val2.append(0)
    elif(y_val[i] == 1):
        y_val0.append(0)
        y_val1.append(1)
        y_val2.append(0)
    else:
        y_val0.append(0)
        y_val1.append(0)
        y_val2.append(1)

y_val0=np.transpose(np.matrix(y_val0))
y_val1=np.transpose(np.matrix(y_val1))
y_val2=np.transpose(np.matrix(y_val2))



# In[994]:

#training for class 0 vs all
eta=0.01
iters=10000
lamb=0.2
w0=fit(x_train,y_tr0,iters,eta,lamb)
pred0=predict(x_val,w0)

for i in range(pred0.shape[0]):
    if (pred0[i] >= 0.5):
        pred0[i] = 1
    else:
        pred0[i] =0 
print("accuracy score for class 0 vs all regressor : " +str(100*accuracy_score(y_val0,pred0))+str('%'))       



# In[995]:

#training for class 1 vs all
eta=0.01
iters=10000
lamb=0.2

w1=fit(x_train,y_tr1,iters,eta,lamb)
pred1=predict(x_val,w1)

for i in range(pred1.shape[0]):
    if (pred1[i] >= 0.5):
        pred1[i] = 1
    else:
        pred1[i] =0 
print("accuracy score for class 1 vs all regressor : " +str(100*accuracy_score(y_val1,pred1))+str('%'))


# In[996]:

#training for class 2 vs all
eta=0.01
iters=10000
lamb=0


w2=fit(x_train,y_tr2,iters,eta,lamb)
pred2=predict(x_val,w2)

for i in range(pred2.shape[0]):
    if (pred2[i] >= 0.5):
        pred2[i] = 1
    else:
        pred2[i] =0 
print("accuracy score for class 2 vs all regressor : " +str(100*accuracy_score(y_val2,pred2))+str('%'))


# In[997]:

#the test set was used only after the value of the hyperparameters was fixed upon and a model was formed
#because of the initial shuffling of the dataset the value of the accuracy for the test set can vary
#averge value of test accuracy has been around 80% while the maximum recorded has been around 90% 

test_pred0=predict(x_test,w0)
test_pred1=predict(x_test,w1)
test_pred2=predict(x_test,w2)


# In[998]:

test_pred=np.hstack((test_pred1,test_pred2))
test_pred=np.hstack((test_pred0,test_pred))


for i in range(test_pred.shape[0]):
    s=np.sum(test_pred[i])
    test_pred[i]=test_pred[i]/s

y_pred=[]
for j in range(test_pred.shape[0]):
    max=test_pred[j,0]
    index=0
    for i in range(test_pred.shape[1]):
        if(test_pred[j,i] > max):
            max=test_pred[j,i]
            index=i;
    if(index == 0):
        y_pred.append(0)
    elif(index == 1):
        y_pred.append(1)
    else:
        y_pred.append(2)
y_pred=np.transpose(np.matrix(y_pred))

print('Final Weights for the model' )
print('Weights for linear regressor to clssify class 0 vs All')
print(w0)
print('Weights for linear regressor to clssify class 1 vs All')
print(w1)
print('Weights for linear regressor to clssify class 2 vs All')
print(w2)

print('The column on the left shows the actual label for the dataset and the column on the right shows the predicted values for the test set ')
print(np.hstack((y_test,y_pred)))
print("Final test accuracy score : " +str(100*accuracy_score(y_pred,y_test))+str('%'))         


# In[ ]:



