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

def predict(x, params):
    #print(f'Params: {params}')
    return params[1] - (params[2]*x + params[3]*x**2 + params[4]*x**3) - params[5]*np.sin((np.pi/8)*x)

def calculate_mse_q3a(a0, a1, a2, a3, b, x, y):
    all_mse = []
    for i in range(len(a1)):
        for j in range(len(a2)):
            for k in range(len(a3)):
                for l in range(len(b)):
                    y_pred = a0 - (a1[i]*x + a2[j]*x**2 + a3[k]*x**3) - b[l]*np.sin((np.pi/8)*x)
                    mse = np.mean((y - y_pred)**2)
                    #add value of mse a0, a1, a2, a3, b to the list
                    all_mse.append([mse, a0, a1[i], a2[j], a3[k], b[l]])
                    #print(f'Mean Squared Error: {mse}')
    print(f'Minimum Mean Squared Error: {min(all_mse)}')
    return all_mse

def find_zero(beta):
    a = []
    b = []
    x_val = 0.00
    step = 0.01
    y_val = predict(x_val, beta)
    a.append(x_val)
    b.append(y_val)
    while y_val > 0:
        x_val += step
        y_val = predict(x_val, beta)
        a.append(x_val)
        b.append(y_val)
    return a, b, x_val

def main():
    x, y = extract_data('P2input2024.txt')
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    a0 = 100
    a1 = [0, 1]
    a2 = [0.2, 0.4]
    a3 = [0, 4]
    b = [20, 40]

    #Question 3a: Calculate all Mean Squared Errors
    all_mse = calculate_mse_q3a(a0, a1, a2, a3, b, x, y)
    all_mse.sort()

    #Question 3b: all mse values with a0, a1, a2, a3, b
    print('Question 3b: All Mean Squared Errors sorted in ascending order')
    for i in all_mse:
        print(i.__str__())

    #Question 3c: plot predicted data with 4 least mse values
    y1 = predict(x, all_mse[0])
    y2 = predict(x, all_mse[1])
    y3 = predict(x, all_mse[2])
    y4 = predict(x, all_mse[3])

    #print(y1)
    #print(y2)
    #print(y3)
    #print(y4)

    #Question 3d: if a0 = 200

    #for least mse parking spot at 0

    print('Question 3d: Parking spot will be fully occupied at x for least mse value')
    a, c, zero = find_zero([all_mse[0][0], 200, all_mse[0][2], all_mse[0][3], all_mse[0][4], all_mse[0][5]])
    print(f'Parking spot will be fully occupied at x for least mse value = {zero}')
    for i in range(1, 5):
        aa, bb, zero_ = find_zero([all_mse[i][0], 200, all_mse[i][2], all_mse[i][3], all_mse[i][4], all_mse[i][5]])
        print(f'Parking spot will be fully occupied at x for {i+1}th least mse value = {zero_}')

    plt.scatter(x, y, label='actual data')
    plt.plot(x, y1, color='red', label='predicted data with least mse')
    plt.plot(x, y2, color='green', label='predicted data with 2nd least mse')
    plt.plot(x, y3, color='blue', label='predicted data with 3rd least mse')
    plt.plot(x, y4, color='purple', label='predicted data with 4th least mse')
    plt.legend()
    plt.show()
    plt.close()

    #Question 3e: plot for Qd
    plt.plot(a,c , color='red', label='predicted data for a0 = 200')
    plt.xlabel('time')
    plt.ylabel('Available parking spaces')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()