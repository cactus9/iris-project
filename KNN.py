def mean(data):
    return sum(data)/len(data)

def std(data):
    return mean([x ** 2 for x in data]) - mean(data) ** 2

def distance_ignore_final(v1, v2):
    return sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)])

class KNN(object):

    def __init__(self, data):
        self.data = data
        self.normalised_data = []

        self.normalizing_params = []
        for i in range(len(data[0])-1):
            #gives us the index of every non-label attribute
            att_data = [x[i] for x in self.data]
            self.normalizing_params += [(mean(att_data), std(att_data))]

        for x in self.data:
            #Normalize it
            self.normalised_data.append([
                (x[i] - self.normalizing_params[i][0]) / self.normalizing_params[i][1] for i in range(len(x) - 1)
            ] + [x[-1]])

    def __repr__(self):
        return repr(self.normalizing_params)

    def classify(self, x, k):
        #First up, normalise the instance using learned params
        norm_instance = [(x[i] - self.normalizing_params[i][0]) / self.normalizing_params[i][1] for i in range(len(x))]

        #Now we start to rank our neighbours - O(n)
        ranked_neighbours = [(distance_ignore_final(nbour, norm_instance), nbour) for nbour in self.normalised_data]

        #Get the top k - O(nk)
        #No sorting so that we are still linear
        good_nbours = []
        for i in range(k):
            next_max = min(ranked_neighbours)
            ranked_neighbours.remove(next_max)
            good_nbours.append(next_max[1])

        classValues = {}
        for instance in good_nbours:
            if instance[-1] not in classValues:
                classValues[instance[-1]] = 1
            else:
                classValues[instance[-1]] += 1

        #get the most common one
        bestCount = 0
        bestVal = None
        for val in classValues:
            if classValues[val] >= bestCount:
                bestVal = val

        return val

    def evaluate(self, data, k):
        count = 0
        for instance in data:
            if instance[-1] == self.classify(instance[:-1], k):
                count+=1

        return count / len(data)
