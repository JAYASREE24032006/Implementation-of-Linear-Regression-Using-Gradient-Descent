# EX-3 : IMPLEMENTATION OF LINEAR REGRESSION  MODEL USING GRADIENT DESCENT
### Name : R.Jayasree
### R.No : 212223040074

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the required library and read the dataframe.


2.Write a function computeCost to generate the cost function.


3.Perform iterations og gradient steps with learning rate.


4.Plot the Cost function using Gradient Descent and generate the required graph.

## PROGRAM:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  X = np.c_[np.ones(len(X1)),X1]
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1,1)
    errors=(predictions - y ).reshape(-1,1)
    theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head)
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## OUTPUT:

![image](https://github.com/user-attachments/assets/2cdf686e-19ab-4710-92c2-e0c01de38947)
![image](https://github.com/user-attachments/assets/775ca13a-79d7-4952-a395-8748524c2329)
![image](https://github.com/user-attachments/assets/71b4a2a8-27b8-4e79-ac3a-0d76ce5c3a23)
![image](https://github.com/user-attachments/assets/02914ce9-812b-46b6-bbb5-eb2d1d6fa5e2)
![image](https://github.com/user-attachments/assets/629aff89-7c55-49e0-9451-3d5545987927)
![image](https://github.com/user-attachments/assets/f207a2c2-d24c-4a15-b012-a4e3667d5fc2)
![image](https://github.com/user-attachments/assets/90d8b9b8-fcd3-44f2-ae99-964a45e68ac3)









## RESULT:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
