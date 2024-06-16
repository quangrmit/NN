


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


def find_loss(intercept):
    return (t1 - (intercept + w * s1)) ** 2 + (t2 - (intercept + w * s2))**2 + (t3 - (intercept + w * s3)) ** 2

def find_slope(intercept):
    return -2 * (t1 - (intercept + w * s1)) - 2 * (t2 - (intercept + w * s2)) - 2 * (t3 - (intercept + w * s3))



def grad():
    # init intercept to some value
    intercept = 0
    # init learning rate
    learning_rate = 0.01
    
    converged = 0.001
    step_size = 0
    count = 0

    while True:

        if step_size >= 0.1:
            break

        slope = find_slope(intercept=intercept)
        print(f'Slope: {slope}')

        step_size = slope * learning_rate
        print(f'Step size: {step_size}')

        intercept = intercept - step_size
        print(f'New intercept: {intercept}')
        
        count += 1
        if count > 1000:
            break



        print('\n')


grad()








    