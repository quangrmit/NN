from nn import sigmoid



w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = sigmoid(dot) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit



# forward pass
x = 3
y = -4

sigx = sigmoid(x)
sigy = sigmoid(y)

num = x + sigy
s = x + y
sq_sum = (s)**2
denom = sigx + sq_sum

f = num / denom



# backprop
dnum = 1 / denom

ddenom = num * (-1 / denom**2)

dsigx = 1 * ddenom
dsq_sum = 1 * ddenom

ds = 2*s * dsq_sum

dx = 1







