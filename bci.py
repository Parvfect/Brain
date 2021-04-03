
"""
Following https://colab.research.google.com/drive/1ZQt8RCkkmTEYXRDmFbj1kpcXpUXD2Da0#scrollTo=vm296sSPBjbG
BCI - classifier
"""


from collections import OrderedDict
from pylab import rcParams
import torch
import torch.nn as nn
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler

# Randomizer seed for consistency
torch.manual_seed(100)

# Initialize parameters
eeg_sample_count = 240
learning_rate = 1e-3 
eeg_sample_length = 226 
number_of_classes = 1 # We want to answer the "is this a P300?" question
hidden1 = 500 
hidden2 = 1000 
hidden3 = 100 
output = 10 

# Creating a sample dataset to check if network is working

sample_positives = [None, None] # Element [0] is the sample, Element [1] is the class

# Creating random samples of x = sample_count/2, y = sample_length
sample_positives[0] = torch.rand(int(eeg_sample_count / 2), eeg_sample_length) * 0.50 + 0.25
sample_positives[1] = torch.ones([int(eeg_sample_count / 2), 1], dtype=torch.float32)

sample_negatives = [None, None] # Element [0] is the sample, Element [1] is the class
sample_negatives_low = torch.rand(int(eeg_sample_count / 4), eeg_sample_length) * 0.25
sample_negatives_high = torch.rand(int(eeg_sample_count / 4), eeg_sample_length) * 0.25 + 0.75
sample_negatives[0] = torch.cat([sample_negatives_low, sample_negatives_high], dim = 0)
sample_negatives[1] = torch.zeros([int(eeg_sample_count / 2), 1], dtype=torch.float32)

# Mix the two and shuffle em
samples = [None, None] # Combine the two
samples[0] = torch.cat([sample_positives[0], sample_negatives[0]], dim = 0)  # concat
samples[1] = torch.cat([sample_positives[1], sample_negatives[1]], dim = 0)

## Create test data that isn't trained on
test_positives = torch.rand(10, eeg_sample_length) * 0.50 + 0.25 # Test 10 good samples
test_negatives_low = torch.rand(5, eeg_sample_length) * 0.25 # Test 5 bad low samples
test_negatives_high = torch.rand(5, eeg_sample_length) * 0.25 + 0.75 # Test 5 bad high samples
test_negatives = torch.cat([test_negatives_low, test_negatives_high], dim = 0)





rcParams['figure.figsize'] = 15, 5

plt.title("Sample Data Set")
plt.plot(list(range(0, eeg_sample_length)), sample_positives[0][0], color = "#bbbbbb", label = "Samples")
plt.plot(list(range(0, eeg_sample_length)), sample_positives[0].mean(dim = 0), color = "g", label = "Mean Positive")
plt.plot(list(range(0, eeg_sample_length)), sample_negatives_high[0], color = "#bbbbbb")
plt.plot(list(range(0, eeg_sample_length)), sample_negatives_high.mean(dim = 0), color = "r", label = "Mean Negative")
plt.plot(list(range(0, eeg_sample_length)), sample_negatives_low[0], color = "#bbbbbb")
plt.plot(list(range(0, eeg_sample_length)), sample_negatives_low.mean(dim = 0), color = "r")
plt.plot(list(range(0, eeg_sample_length)), [0.75] * eeg_sample_length, color = "k")
plt.plot(list(range(0, eeg_sample_length)), [0.25] * eeg_sample_length, color = "k")
plt.legend()
plt.show()


""" Creating the network """

# Define the network
model = nn.Sequential()

# Input Layer (Size 226 -> 500)
model.add_module('Input Linear', nn.Linear(eeg_sample_length, hidden1))
model.add_module('Input Activation', nn.CELU()) 

# Hidden Layer (Size 500 -> 1000)
model.add_module('Hidden Linear', nn.Linear(hidden1, hidden2))
model.add_module('Hidden Activation', nn.ReLU())

# Hidden Layer (Size 1000 -> 100)
model.add_module('Hidden Linear2', nn.Linear(hidden2, hidden3))
model.add_module('Hidden Activation2', nn.ReLU())

# Hidden Layer (Size 100 -> 10)
model.add_module('Hidden Linear3', nn.Linear(hidden3, 10))
model.add_module('Hidden Activation3', nn.ReLU())

# Output Layer (Size 10 -> 1)
model.add_module('Output Linear', nn.Linear(10, number_of_classes))
model.add_module('Output Activation', nn.Sigmoid())

# Define a loss function 
loss_function = torch.nn.MSELoss()

# Define a training procedure 
def train_network(train_data, actual_class, iterations):

    #Keep track of loss 
    loss_data = []

    #Begin training for certain iterations
    for i in range(iterations):

        classification = model(train_data)

        #Find out how wrong the network was
        loss = loss_function(classification, actual_class)
        loss_data.append(loss)

        # Zero out the optimizer gradient every iteration
        optimizer.zero_grad()

        # Backpropogate
        loss.backward()
        optimizer.step()
    
    # Plot a nice loss graph at the end of training
    #rcParams['figure.figsize'] = 10, 5
    
    print(loss_data.numpy())
    #plt.title("Loss vs Iterations")
    #plt.plot(list(range(0, len(loss_data))), loss_data)
    #plt.show()




# Save the network's default state so we can retrain from the default weights
torch.save(model, "/home/parv/Documents/model_default_state")

#Verify network

# Make sure we're starting from untrained every time
tutorial_model = torch.load("/home/parv/Documents/model_default_state")

# Define a learning function, needs to be reinitialized every load
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the network using our training procedure with the sample data
print("Below is the loss graph for our training session")
train_network(samples[0], samples[1], iterations = 100)