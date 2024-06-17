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



# (t1 - (intercept + w1 * x1 + w2 * x2 + w3 * x3)) ** 2 + 
# (t2 - (intercept + w1 * s1 + w2 * s2 + w3 * s3)) ** 2 + 
# (t3 - (intercept + w1 * r1 + w2 * r2 + w3 * r3)) ** 2 + 


# 2udu

# 2 * (t1 - (intercept + w1 * x1 + w2 * x2 + w3 * x3)) * -x1






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


# grad()

'''
    Linear regression()

    def fit(X, y):
        X is a matrix
        y is a vector

        The goal is to find the weights vector (w) and bias (b) 

'''
# TODO:
# Write this with np, might make this faster


def fit(X, y):
    learning_rate = 0.001
    intercept = 0
    print(y)
    print(y.iloc[0])
    print(type(y))

    w = np.random.rand(len(X.iloc[0]))


    def find_loss(intercept, w):
        s = 0
        for i in range(len(y)):
            pred = 0
            for a in range(len(X.iloc[i])):
                pred += w[a] * X.iloc[i][a]

            pred += intercept
            loss_1_row = (y[i] - pred) ** 2
            
            s += loss_1_row
        return s
    

    def find_slope_intercept(intercept):
        s = 0

        for i in range(len(y)):
            temp = 0
            temp += np.dot(w, X.iloc[i])

            temp += intercept
            s += -2 * (y.iloc[i] - temp)
        return s
        

    def find_slope_weight(weight_ind):
        s = 0
        w_copy = w.copy()
        
        for i in range(len(y)):

            row_calculation = np.dot(w_copy, X.iloc[i])
            s += -2 * X.iloc[i][weight_ind] * (y.iloc[i] - (row_calculation + intercept))
        return s



    def grad():
        step_size_intercept = None
        step_size_weight = np.zeros(w.shape)
        print('step size weight array')
        converged = 0.001
        count = 0
        intercept = 0

        while True:
            print('\n')
            print('step size weight')
            print(step_size_weight)
            print('step size intercept')
            print(step_size_intercept)

            count += 1
            if count > 1000:
                break
            # check convergence
            
            if step_size_intercept != None:
                if step_size_intercept < converged:
                    conv = True
                    for i in range(len(step_size_weight)):
                        if step_size_weight[i] >= converged:
                            conv = False
                            break
                    if conv:
                        break
            
            slope_intercept = find_slope_intercept(intercept)
            step_size_intercept = slope_intercept * learning_rate
            intercept = intercept - step_size_intercept

            if intercept == np.NaN:
                break

            print('Begin weight tuning')
            for i in range(len(w)):
                # Find slope of the ith weight
                slope_weight = find_slope_weight(i)
                step_size_weight[i] = slope_weight * learning_rate


                w[i] = w[i] - step_size_weight[i]
                # w.iloc[i] = w.[i] - step_size_weight
                # print(w.iloc[i])
                print( {'intercept': intercept, 'weights': w})

        return {'intercept': intercept, 'weights': w}
    
    return grad()



def predict(X, dictionary):
    y_pred = []
    intercept = dictionary['intercept']
    weights = dictionary['weights']

    for i in range(len(X)):
        y = 0
        for j in range(len(X.iloc[i])):
            y += weights[j] * X.iloc[i][j]
        y += intercept
        y_pred.append(y)

    return y_pred





# model.fit()











