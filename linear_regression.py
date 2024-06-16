import numpy as np


t1 = 1.4
t2 = 1.9
t3 = 3.2

# the goal of grad descent is to find these two variables
# demonstration will hard code the w to find the intercept
intercept = 0
w = 0.64


s1 = 0.5
s2 = 2.3
s3 = 2.9


loss = (t1 - (intercept + w * s1)) ** 2 + (t2 - (intercept + w * s2))**2 + (t3 - (intercept + w * s3)) ** 2

dintercept = -2 * (t1 - (intercept + w * s1)) - 2 * (t2 - (intercept + w * s2)) - 2 * (t3 - (intercept + w * s3))


def find_loss(intercept, w):
    return (t1 - (intercept + w * s1)) ** 2 + (t2 - (intercept + w * s2))**2 + (t3 - (intercept + w * s3)) ** 2

def find_slope_intercept(intercept):
    return -2 * (t1 - (intercept + w * s1)) - 2 * (t2 - (intercept + w * s2)) - 2 * (t3 - (intercept + w * s3))


def find_slope_weight(w):
    return -2 * s1 * (t1 - (intercept + w * s1)) - 2 * s2 * (t2 - (intercept + w * s2)) - 2 * s3 *  (t3 - (intercept + w * s3))






def grad():
    # init intercept to some value
    intercept = 0
    # init weight to some value
    weight = 1


    # init learning rate
    learning_rate = 0.01
    
    converged = 0.001
    step_size_intercept = 0
    step_size_weight = 0
    count = 0

    while True:

        if step_size_intercept >= converged and step_size_weight >= converged:
            break
        print(find_loss(intercept=intercept, w=weight))

        slope_intercept = find_slope_intercept(intercept=intercept)
        print(f'Slope intercept: {slope_intercept}')

        step_size_intercept = slope_intercept * learning_rate
        print(f'Step size intercept: {step_size_intercept}')

        intercept = intercept - step_size_intercept
        print(f'New intercept: {intercept}')

        slope_weight = find_slope_weight(w=weight)
        print(f'Slope weight: {slope_intercept}')

        step_size_weight = slope_weight * learning_rate
        print(f'Step size weight: {slope_intercept}')

        weight = weight - step_size_weight
        print(f'New weight: {slope_intercept}')


        
        count += 1
        if count > 200:
            break



        print('\n')


grad()

'''
    Linear regression()

    def fit(X, y):
        X is a matrix
        y is a vector

        The goal is to find the weights vector (w) and bias (b) 

'''

class LinearRegression:
    def __init__(self) -> None:
        pass

    def fit(X, y):
        learning_rate = 0.01
        intercept = 0
        # W will be a vector
        w = np.random.rand(len(X[0]))


        def find_loss(intercept, w):
            s = 0
            for i in range(len(y)):
                pred = 0
                for a in range(len(X[i])):
                    pred += w[a] * X[i][a]

                pred += intercept
                loss_1_row = (y[i] - pred) ** 2
                
                s += loss_1_row
            return s
        

        def find_slope_intercept(intercept):
            s = 0

            for i in range(len(y)):
                temp = 0
                for a in range(len(X[i])):
                    temp += w[a] * X[i][a]
                temp += intercept
                s += -2 * (y[i] - temp)
            return s
            

        def find_slope_weight(weight):
            s = 0

            for i in range(len(y)):
                temp = 0
                for a in range(len(X[i])):
                    temp += w[a] * X[i][a]
                temp += intercept
                s += -2 * (y[i] - temp) * X[i][a]
            return s



        def grad():
            step_size_intercept = 0
            step_size_weight = np.zeros(w.shape, 0.0)
            converged = 0.001
            count = 0

            while True:
                count += 1
                if count > 1000:
                    break
                # check convergence
                if step_size_intercept <= converged:
                    conv = True
                    for i in range(len(step_size_weight)):
                        if step_size_weight > converged:
                            conv = False
                            break
                
                
                slope_intercept = find_slope_intercept(intercept)
                step_size_intercept = slope_intercept * learning_rate
                intercept = intercept - step_size_intercept


                for i in range(len(w)):
                    slope_weight = find_slope_weight(w[i])
                    step_size_weight = slope_weight * learning_rate
                    w[i] = w[i] - step_size_weight

            return {'intercept': intercept, 'weights': w}


















