"""

network.py

A module to implement stochastic gradient descent learning algorithms
for digit recognition.
"""

import random

import numpy as np

#network
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weigths = [np.random.randn(y, x)
                        for x, y in zip(sizes[:,-1], sizes[1:])]

    def feedforward(self, a):
        #return the output of the network, if a is the input
        
        for b, w in zip(self.biases, self.weigths):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SDG(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        #sdg method that implements stochastic gradient descent

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {}".format(j, self.evaluate(test_data), n_test)

            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        #update the networks weights and biases by applying gradient descent

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shaep) for w in self.weigths]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla,w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weigths = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weigths, nabla_w)]

    def backprop(self, x, y):
        #return the gradient of the cost function given the training data as a tuple

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shaep) for w in self.weigths]

        #feedforward
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weigths):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activation[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weigths[-l+1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l+1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        #return the number of test inputs for which the network computes the correct result

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #return the vector of partial derivatives of the cost function

        return(output_activations-y)


#functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
