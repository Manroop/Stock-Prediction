"""
Author: Manroop Turna
Advisor: Dr. Young Kim
Deep neural network is used to train Economic data provided by Dr. Kim by
applying two Hidden layrs of sigmoid activation function.
To output layer is linear as it is a regression problem.
30 preceptrons are used in each hidden layer.
"""

from __future__ import division, print_function, absolute_import
import tflearn
import tensorflow
import numpy as np

# Import Data
train_data = np.loadtxt('CCJ_regression.csv', delimiter=',', dtype=np.float32)
trainX = train_data[0:2300, 0:-1]
trainY = train_data[0:2300, [-1]]

# Validation data
valid_X = train_data[2500:2900, 0:-1]
valid_Y = train_data[2500:2900, [-1]]

#test data
test_data = np.loadtxt('CC_test.csv', delimiter=',', dtype=np.float32)
testX = test_data[0:563, 0:-1]
testY = test_data[0:563:, [-1]]

# Building Neural Network
input_layer = tflearn.input_data(shape = [None, 46])
#       Normalizing Input Batch
input_norm = tflearn.layers.normalization.batch_normalization(input_layer,
            beta = 0.0, gamma = 1.0, epsilon = 1e-05, decay = 0.9, stddev = 0.002
            , trainable = True, restore = True, reuse = False, scope = None,
            name = 'BatchNormalization' )
# Hidden Layer 1
dense1 = tflearn.fully_connected(input_norm, 10, activation = 'sigmoid')
dropout1  =tflearn.dropout(dense1, 0.1)

# Hidden Layer 2
dense2 = tflearn.fully_connected(dropout1, 10, activation = 'sigmoid',)
dropout2 = tflearn.dropout(dense2, 0.1)

# Output Layer
softmax = tflearn.fully_connected(dropout2, 1, activation = 'ReLU')

# Loss function SGD = Stochastic Gradient Descent
sgd = tflearn.SGD(learning_rate = 0.3)

# Standard Error
top_k  =tflearn.metrics.R2()

# Initilizing Neral Network
net = tflearn.regression(softmax, optimizer = sgd, metric = top_k, loss = 'mean_square', batch_size = 50)

# Training Neral Network
model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='/home/mst/Documents/Deep/')
model.fit(trainX,trainY, n_epoch = 8, validation_set = (valid_X, valid_Y), show_metric = True, run_id = "dense_model_25")

# Saving Model
model.save('mp3')

# Load Model for Testing
model.load('./mp3')

# Print Predictions
diff = []
for i in range(563):
    diff.append(model.predict([testX[i]]) - testY[i])

S = (sum(diff))**2
RMSE = ((S/563)**(1/2))
print (RMSE)
# Print dataset accuracy
print(model.evaluate(testX, testY))






"""______________END________________"""
