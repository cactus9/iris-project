from math import log2

class Function:
    def __init__(self, value, func):
        self.func = func
        self.value = value
    def __repr__(self):
        return repr(self.value)
    def __call__(self, x):
        return self.func(x)

class LTFunction(Function):

    def __init__(self, value):
        super(LTFunction, self).__init__(value, lambda x: x < value)
    def __repr__(self):
        return f"< {self.value}"

class GEQFunction(Function):

    def __init__(self, value):
        super(GEQFunction, self).__init__(value, lambda x: x >= value)
    def __repr__(self):
        return f">= {self.value}"

class Node:
    att_names = None
    def __init__(self, attr_index):
        self.attr_index = attr_index
        self.children = {}

    def addChild(self, classifier, subtree):
        self.children[classifier] = subtree

    def __repr__(self):
        if Node.att_names:
            return Node.att_names[self.attr_index] + " is " + repr(self.children)
        return str(self.attr_index) + " is " + repr(self.children)

class TerminalNode:
    def __init__(self, label):
        self.label = label
    def __repr__(self):
        return "Classification: " + repr(self.label)

def makeClass(i):
    #wrapper function for making the anonymous ones, as otherwise they interact strangely
    return Function(i, lambda x : x == i)

def entropy(data):
    #for each class, we have to find it's proportion
    classValues = {}
    for instance in data:
        if instance[-1] not in classValues:
            classValues[instance[-1]] = 1
        else:
            classValues[instance[-1]] += 1

    e = 0
    for c in classValues:
        c = classValues[c]/len(data)
        e += c * log2(c)

    return -1 * e

def getClassifiers(data):
    #data is in the form (attribute, classification)
    #We return a set of boolean functions such that a function returns true iff the input matches its class
    if type(data[0][0]) in [int, float]:
        #we need to find the optimal splitting point
        sorted_data = sorted(data)
        bestEnt = entropy(data)
        bestIndex = 0
        for i in range(1,len(data)):
            if sorted_data[i-1][1] != sorted_data[i][1]:
                #this is a boundary point, and so may be a candidate
                lt_entropy = entropy(sorted_data[:i]) * i / len(data)
                geq_entropy = entropy(sorted_data[i:]) * (len(data) - i) / len(data)
                if bestEnt >= lt_entropy + geq_entropy:
                    bestIndex = i
                    bestEnt = lt_entropy + geq_entropy
        splitVal = sorted_data[bestIndex][0]
        return [GEQFunction(splitVal), LTFunction(splitVal)]

    #Our data is in discrete classes
    classes = {x[0] for x in data}
    classifiers = [makeClass(c) for c in classes]
    return classifiers

def infogain(data, attr_index):
    info_gain = entropy(data)
    classifiers = getClassifiers([(x[attr_index], x[-1]) for x in data])
    for classifier in classifiers:
        subset = [x for x in data if classifier(x[attr_index])]
        info_gain -= entropy(subset) * len(subset) / len(data)

    return (info_gain, classifiers)

def ID3(data, attr_indices):
    if entropy(data) == 0.0:
        return TerminalNode(data[0][-1])

    if len(attr_indices) == 0:
        #Take the most popular classification
        classes = {x[-1] for x in data}
        classCounts = [
            (len([0 for x in data if x[-1] == c]), c) for c in classes
        ]
        bestClass = max(classCounts)[1]
        return TerminalNode(bestClass)

    bestGain = -1
    bestClassifiers = None
    bestIndex = None
    for i in attr_indices:
        info_gain, classifiers = infogain(data, i)
        if info_gain > bestGain:
            bestGain = info_gain
            bestClassifiers = classifiers
            bestIndex = i

    newNode = Node(bestIndex)
    newAtts = [i for i in attr_indices if i != bestIndex]

    for classifier in bestClassifiers:
        subset = [x for x in data if classifier(x[bestIndex])]
        newNode.addChild(classifier, ID3(subset, newAtts))

    return newNode

def fit(data):
    attr_indices = [x for x in range(len(data[0]) - 1)]
    return ID3(data, attr_indices)

def classify(tree, instance):
    if type(tree) == TerminalNode:
        return tree.label
    else:
        for c in tree.children:
            if c(instance[tree.attr_index]):
                return classify(tree.children[c], instance)

def evaluate(tree, data):
    count = 0
    for instance in data:
        if instance[-1] == classify(tree, instance):
            count+=1

    return count / len(data)
