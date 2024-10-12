#Implement a linear regression learner to solve this best fit problem for 1 dimensional data
#The data is in the file P2input2024.txt. The first column is the x value and the second column is the y value.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Extract data from txt file and fit the model
with open('P2input2024.txt') as f:
    data = f.read().splitlines()
x = []
y = []
for i in data:
    x.append(float(i.split()[0]))
    y.append(float(i.split()[1]))

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

#y(t) = a0 – (a1*x+ a2*x**2+a3*x**3 ) – b*sin(c*x) where c= π/8

#calculate squares of x
x2 = x**2
#calculate cubes of x
x3 = x**3
#calculate sin of x
xsin = np.sin(np.pi/8*x)
#combine all features in one array
X = np.column_stack((x, x2, x3, xsin))  

model = LinearRegression()
model.fit(X, y)
a0 = model.intercept_
a1, a2, a3, b = model.coef_[0]
print(f'y(t) = {a0} - ({a1}*x + {a2}*x**2 + {a3}*x**3) - ({b})*sin(pi/8*x)')
y_pred = model.predict(X)
mse = np.mean((y - y_pred)**2)
print(f'Mean Squared Error: {mse}')


plt.scatter(x, y, label='actual data')
plt.plot(x, y_pred, color='red', label='predicted data')
plt.legend()
plt.show()
