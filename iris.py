#General structure:
# load iris data
# import yer classifiers
# evaluate them, nothing too spicy

import random
import ID3
import KNN
from neuralNets import UnitNetwork as NN

random.seed(0)

data = []
with open("iris.data", "r") as f:
    data = f.readlines()

#convert data to proper types
data = [ x.split(",") for x in data ][:-1]

data = [
    [float(x[0]),
    float(x[1]),
    float(x[2]),
    float(x[3]),
    x[4][:-1]] for x in data
]

random.shuffle(data)

train_data = data[::2]
test_data = data[1::2]
#First up, ID3
ID3.Node.att_names = ("Sepal length","Sepal width","Petal length", "Petal width")
tree = ID3.fit(train_data)
print(tree) # Not always the cleanest tree (some pruning of extraneous nodes possible)
print("The accuracy of the ID3 tree:", ID3.evaluate(tree, test_data))

#Next, KNN with a search over the number of neighbours
knn = KNN.KNN(train_data)
for i in range(1,6):
    print(f"KNN with {i} neighbours considered: {knn.evaluate(test_data, i)}")


#Finally, the neural network
#First we need to reformat the data
label_list = list({x[-1] for x in data})

def toOneHot(x):
    return [1 if x == label else 0 for label in label_list]

def reformat(dataset):
    return [(x[:-1], toOneHot(x[-1])) for x in dataset]

#We will search over the size of the hidden layer
for i in range(1,6):
    net = NN([4,i,3])
    net.train(reformat(train_data), 500, quiet=True)
    print(f"Accuracy of a neural network with a hidden layer of size {i}: {net.evaluate(reformat(test_data))}")
