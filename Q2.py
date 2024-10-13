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

    return y_pred

#find when parking spaces are zero
def find_zero(beta, x, y_pred):
    #y(t) = a0 – (a1*x+ a2*x**2+a3*x**3 ) – b*sin(c*x) where c= π/8
    #a0 = beta[0], a1 = beta[1], a2 = beta[2], a3 = beta[3], b = beta[4]
    x_val = 12.25
    step = 0.25
    y_val = parking_spaces(x_val, beta)
    while y_val > 0:
        x_val += step
        y_val = parking_spaces(x_val, beta)
        x = np.append(x, x_val)
        y_pred = np.append(y_pred, y_val)
    return x_val, x, y_pred

def parking_spaces(x, beta):
    return beta[0] - (beta[1]*x + beta[2]*x**2 + beta[3]*x**3) - beta[4]*np.sin((np.pi/8)*x)

#Write main function to implement linear regression from scratch
def main():
    x, y = extract_data('P2input2024.txt')
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    #y(t) = a0 – (a1*x+ a2*x**2+a3*x**3 ) – b*sin(c*x) where c= π/8
    #calculate squares of x
    x2 = -1*x**2
    #calculate cubes of x
    x3 = -1*x**3
    #calculate sin of x
    xsin = -1*np.sin(np.pi/8*x)
    #combine all features in one array
    X = np.column_stack((np.ones_like(x), -x, x2, x3, xsin))

    #part a  
    coef = linear_regression(X, y)
    y_pred = predict(X, coef, y, x)

    #part b
    zero_parking_spaces, x1, y_pred1 = find_zero(coef, x, y_pred)
    print(f'Parking spaces will be zero at time: {zero_parking_spaces}')

    plt.scatter(x, y, label='actual data')
    plt.plot(x1, y_pred1, color='red', label='predicted data')
    plt.xlabel('time')
    plt.ylabel('Available parking spaces')
    plt.legend()
    plt.show()  
    
if __name__ == '__main__':
    main()