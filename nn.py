import numpy as np
import math

# TODO:
# Create a basic feed-forward


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Layer:

    def __init__(self, n_nodes, activation=None, inp=None) -> None:
        self.n_nodes = n_nodes
        self.nodes = np.full((n_nodes), 0.0)
        self.activation = activation

        if type(inp) != type(None):
            self.n_nodes = len(inp)
            self.nodes = inp
    
    def __repr__(self) -> str:
        return str(self.nodes)

class Network:

    def __init__(self, layers, inp) -> None:
        self.layers = layers
        self.layers[0] = Layer(inp=inp, n_nodes=None)
        
        # Init weights
        # The weights in between each two layers is stored in an arrays
        self.weights = []
        for i in range(1, len(self.layers)):
            n = (self.layers[i].n_nodes * self.layers[i - 1].n_nodes)
            self.weights.append(np.random.rand(n))


    def forward(self):

        # Starting from the second layer
        for i in range(1, len(self.layers)):

            # Starting from the first node of the behind layer
            for j in range(len(self.layers[i].nodes)):

                sum_product = 0
                # Calculate the sum of products for this node\
                # Loop through the nodes of the before layer
                for a in range(len(self.layers[i - 1].nodes)):

                    # Each node as a weight

                    weight_ind = a + j * len(self.layers[i - 1].nodes)
                    # print(weight_ind)
                    w = self.weights[i - 1][weight_ind]

                    print([self.layers[i - 1].nodes[a], w])

                    n = self.layers[i - 1].nodes[a] * w
                    sum_product += n
                
                # Update the number in the node
                self.layers[i].nodes[j] = sigmoid(sum_product)
                print(sum_product)
                        

    def __repr__(self) -> str:
        return str(self.layers)


l1 = Layer(5)
l2 = Layer(4)
l3 = Layer(5)
all_layers = [l1, l2, l3]

np.random.seed(0)
inp = np.random.rand(5)

nn = Network(layers=all_layers, inp=inp)
nn.forward()
print(nn.layers)
print(nn.layers[1].nodes[0])







