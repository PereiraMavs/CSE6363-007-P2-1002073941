#Implement linear regression from scratch 

import numpy as np
import matplotlib.pyplot as plt

def extract_data(file):
    with open(file) as f:
        data = f.read().splitlines()
    x = []
    y = []
    for i in data:
        x.append(float(i.split()[0]))
        y.append(float(i.split()[1]))
    return x, y

#linear regression
def linear_regression(X, y):
    #calculate the coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(f'Coefficients: {beta}')
    return beta

#predict the output
def predict(X, beta, y, x):
    y_pred = X @ beta
    mse = np.mean((y - y_pred)**2)
    print(f'Mean Squared Error: {mse}')

    plt.scatter(x, y, label='actual data')
    plt.plot(x, y_pred, color='red', label='predicted data')
    plt.xlabel('time')
    plt.ylabel('Available parking spaces')
    plt.legend()
    plt.show()

#Write main function to implement linear regression from scratch
def main():
    x, y = extract_data('P2input2024.txt')
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
    X = np.column_stack((np.ones_like(x), x, x2, x3, xsin))  
    coef = linear_regression(X, y)
    predict(X, coef, y, x)
    
if __name__ == '__main__':
    main()