import numpy as np


#Assume y = wx+b where w = slope and b = intercept 
#Assume loss function can be represented with a mean squared error: loss = 1/n∑(Ytrue-Ypred)^2 to tell us how wrong we are
#Adjust w and b to minimize loss with gradient descent, an algorith to find the minimum of a function

#Training data
X=np.array([1, 2, 3, 4])
y=np.array([2, 4, 6, 8])

#parameters
w = 0.0
b = 0.0
lr = 0.01

#Training loop
for _ in range(1000):
    y_pred = w * (X+b)

    #gradients
    dw = (-2 * np.mean(X*(y-y_pred)))
    db = (-2 * np.mean(y-y_pred))

    #Update
    w -= lr *dw
    b -= lr *db

print(w,b)


