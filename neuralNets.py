from math import exp
from random import random

class Weight:
    """weight class for units"""

    def __init__(self, step_size=0.1):
        self.value = 2*random() - 1
        self.step_size = step_size

    def __mul__(self, num):
        return self.value * num

    def __rmul__(self, num):
        return self.value * num

    def __add__(self, num):
        return self.value + num

    def __radd__(self, num):
        return self.value + num

    def __repr__(self):
        return repr(self.value)

    def update(self, deriv):
        self.value += self.step_size * deriv

class MomentumWeight(Weight):

    momentum_constant=0.5

    def __init__(self, step_size = 0.1, ):
        super(MomentumWeight, self).__init__(step_size)
        self.prev_update = 0

    def update(self, deriv):
        val = self.step_size * deriv + self.momentum_constant * self.prev_update
        self.prev_update = val
        self.value += val

def sigmoid(x):
    return 1 / (1 + exp( -1 * x))

class Unit:
    """The superclass for sigmoid units, implemented in case of future extensions."""

    def __init__(self, size, weight_type = Weight, step_size = 0.1):
        self.weights = [weight_type(step_size=step_size) for i in range(size)]
        self.bias = weight_type(step_size=step_size)
        self.inputVector = None
        self.out = None
        self.delta = 0
        self.size = size

    def activate(self, inputVector):
        self.inputVector = inputVector
        result = 0
        result += self.bias
        for i in range(len(inputVector)):
            result += self.weights[i] * inputVector[i]
        self.out = result
        return result

    def get_delta(self, error):
        self.delta = error

    def update(self):
        for i in range(self.size):
            self.weights[i].update(self.delta * self.inputVector[i])

    def __mul__(self, inputVector):
        return self.activate(inputVector)

    def __rmul__(self, inputVector):
        return self.activate(inputVector)

    def __repr__(self):
        s = "Bias: " + repr(self.bias)
        s += "\nWeights:"
        for weight in self.weights:
            s += "\n" + repr(weight)
        return s + "\n"

class SigmoidUnit(Unit):

    def __init__(self, size, weightType = Weight, step_size = 0.1):
        super(SigmoidUnit, self).__init__(size, weightType, step_size=step_size)

    def activate(self, inputVector):
        self.inputVector = inputVector
        self.out = sigmoid(super().activate(inputVector))
        return self.out

    def get_delta(self, error):
        self.delta = error * self.out * (1-self.out)

    def update(self):
        for i in range(self.size):
            self.weights[i].update(self.delta * self.inputVector[i])


class UnitLayer:

    def __init__(self, insize, size, unitType = SigmoidUnit, weightType = MomentumWeight, step_size = 0.1):
        self.units = [unitType(insize, weightType, step_size) for i in range(size)]
        self.insize = insize
        self.size = size

    def __repr__(self):
        res = ""
        for unit in self.units:
            res += repr(unit) + "\n"
        return res

    def activate(self, inputVector):
        return [unit.activate(inputVector) for unit in self.units]

    def update(self, errorVec):
        #need to return vals of wij deltai
        for i in range(self.size):
            self.units[i].get_delta(errorVec[i])

        out = [
            #Vector for each unit in previous layer
            sum([self.units[i].weights[j].value * self.units[i].delta for i in range(self.size) ]) for j in range(self.insize)
        ]
        for i in range(self.size):
            self.units[i].update()

        return out

def MSE(o, t):
    return 0.5*sum([(t[i] - o[i]) ** 2 for i in range(len(o))])

def MSE_deriv(o,t):
    return [t[i] - o[i] for i in range(len(o))]


class UnitNetwork:

    def __init__(self, topology, unitType = SigmoidUnit, weightType = MomentumWeight, step_size=0.1):
        self.layers = [
            UnitLayer(topology[i], topology[i+1], unitType, weightType, step_size=step_size) for i in range(len(topology) - 1)
        ]

    def train(self, dataset, epochs, quiet=False):
        #dataset = iterable of instance-classification pair tuples
        for i in range(epochs):
            loss = 0

            for invec, target in dataset:

                outvec = self.activate(invec)

                loss += MSE(outvec, target)
                errorVec = MSE_deriv(outvec, target)

                for layer in self.layers[::-1]:
                    errorVec = layer.update(errorVec)

            if (i+1)%50 == 0 and not quiet:
                print(loss)

    def activate(self, invec):
        for layer in self.layers:
            invec = layer.activate(invec)

        return invec

    def evaluate(self, dataset):
        #Assumes task is classification, so just checks for highest confidence to give result
        correct_count = 0

        for invec, target in dataset:
            outvec = self.activate(invec)

            if outvec.index(max(outvec)) == target.index(max(target)):
                correct_count+=1

        return correct_count/len(dataset)
