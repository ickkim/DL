
# coding: utf-8

# In[1]:

from random import randrange
from random import random
from math import exp
from csv import reader


# ## Preprocess the Data

# In[2]:

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# load and prepare data
filename = './data/seeds_dataset.csv'
dataset = load_csv(filename)


# In[3]:

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)


# ## Normalization

# In[4]:

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


# In[5]:

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# ## Initialize Network

# In[6]:

# Initialize a network
def initialize_network(n_inputs: int,
                       n_hidden: int,
                       n_outputs: int) -> list:
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# ## Forward Propagation

# In[7]:

# Calculate neuron activation for an input
def apply_linear(weights: list, inputs: list) -> float:
    activation = weights[-1]
    
    '''
    IMPLEMENT THE REST PART OF THE FUNCTION
    '''

    return activation
 
# Transfer neuron activation
def apply_activation(value: float) -> float:
    '''
    IMPLEMENT THE REST PART OF THE FUNCTION
    '''
    return None
 
# Forward propagate input to a network output
def forward_propagate(network: list, row: list) -> list:
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = apply_linear(neuron['weights'], inputs)
            neuron['output'] = apply_activation(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# ## Backward Propagation

# In[23]:

# Calculate the derivative of an neuron output
# Derivative of Sigmoid Function
def transfer_derivative(output: float) -> float:
    
    '''
    IMPLEMENT THE REST PART OF THE FUNCTION
    '''
    
    return None


def compute_error_for_layer(network: list, layer: list, index: int) -> float:
    error = 0.0
                
    '''
    IMPLEMENT THE REST PART OF THE FUNCTION
    '''
    
    return error

def compute_delta(neuron: list, index: int, errors: list) -> float:
    
    '''
    IMPLEMENT THE REST PART OF THE FUNCTION
    '''
    
    return None

# Backpropagate error and store in neurons
def backward_propagate_error(network: list,
                             expected: list):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for idx in range(len(layer)):
                errors.append(compute_error_for_layer(network, network[i + 1], idx))
        else:
            for idx in range(len(layer)):
                neuron = layer[idx]
                errors.append(expected[idx] - neuron['output'])
        for idx in range(len(layer)):
            neuron = layer[idx]
            neuron['delta'] = compute_delta(neuron, idx, errors)


# ## Train a Network

# In[24]:

# Update network weights with error
def update_weights(network: list,
                   row: list,
                   l_rate:float):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network:list,
                  train: list,
                  l_rate: float,
                  n_epoch: int,
                  n_outputs: int):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# ## Evaluate a Model

# In[25]:

# Make a prediction with a network
def predict(network: list, row: list):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def train_and_test(train: list,
                     test: list,
                     l_rate: float,
                     n_epoch: int,
                     n_hidden: int) -> list:
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)


# In[26]:

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = train_and_test(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# ## Main Function

# In[27]:

# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))



