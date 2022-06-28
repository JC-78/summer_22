#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:58:40 2022

@author: yaron
"""

#%% Imports, definitions

import numpy as np
import matplotlib.pyplot as plt
from Spline import SplineModel, fit_linear_model

#PyTorch imports for PyTorch example
import torch
import torch.nn as nn
from torch.optim import Adam

#%% Begin by constructing data for a 1D function and displaying the function

x_vals = np.linspace(1, 5, 100) 
y_vals = np.array([(x**2) + (5*x) + 6 for x in x_vals])
plt.plot(x_vals, y_vals)
plt.show()

#%% Initialize the spline model and do a least squares fit

config = {'xknots' : np.linspace(1, 5, 50), #The position of the control points for the spline
          'equal_knots' : False, #Whether or not to have equal knots in the two halves of a joined spline (disregard)
          'cutoff' : 5, #The cutoff distance for the spline
          'bconds' : 'none',  #The boundary conditions for the spline
          'deg' : 3, #The degree of the spline. Here, we are using cubic (3rd degree) splines
          'max_der' : 2} #The maximum derivative of the spline evaluated. The maximum derivative possible is the spline degree - 1

model = SplineModel(config)
coefs, A, b = fit_linear_model(model, x_vals, y_vals)

#%% Overlay the fitted values and the fitted data

#Generating predictions using a matrix multiply of the form y = Ax + b

predicted_values = (A @ coefs) + b
fig, axs = plt.subplots()
axs.plot(x_vals, predicted_values, label = 'spline predictions')
axs.plot(x_vals, y_vals, label = 'truth values')
axs.set_title("Overlayed predicted values and truth values")
axs.set_ylabel("Value")
axs.set_xlabel("X")
axs.legend()

#%% Try adding noise!

'''
Add some randomized noise to the truth value and then see the quality of the
fit. Try altering the magnitude of the random noise and the number of fitting
points
'''
#Try some intiial noisy data drawn from a uniform distribution over [0, 1)

y_vals = np.array(y_vals)
noise = np.random.rand(y_vals.shape[0])

noisy_data = y_vals + noise
fig, axs = plt.subplots()
axs.plot(x_vals, noisy_data)
axs.set_title("Noisy spline data, uniform noise on interval [0, 1)")
axs.set_ylabel("Value")
axs.set_xlabel("X")

#You can alter the magnitude of the noise by scaling each value by an arbitrary factor
noisier_data = y_vals + (noise * 10) #Scale here by a factor of 10
fig, axs = plt.subplots()
axs.plot(x_vals, noisier_data)
axs.set_title("Noisier spline data, uniform noise on interval [0, 10)")
axs.set_ylabel("Value")
axs.set_xlabel("X")

#Try fitting the spline to each of these scenarios. We will use the same configurtion
#   as before
config = {'xknots' : np.linspace(1, 5, 50), 
          'equal_knots' : False, 
          'cutoff' : 5, 
          'bconds' : 'none',  
          'deg' : 3, 
          'max_der' : 2}

noisy_model = SplineModel(config)
noisier_model = SplineModel(config)

coefs_noisy, A_noisy, b_noisy = fit_linear_model(noisy_model, x_vals, noisy_data)
coefs_noisier, A_noisier, b_noisier = fit_linear_model(noisier_model, x_vals, noisier_data)

#Check the quality of the overlayed fits
fig, axs = plt.subplots()
axs.plot(x_vals, noisy_data, label = 'noisy data truth value')
axs.plot(x_vals, (A_noisy @ coefs_noisy) + b_noisy, label = 'predicted noisy values')
axs.set_title("Noisy spline fit overlay, uniform noise on interval [0, 1)")
axs.set_ylabel("Value")
axs.set_xlabel("X")

fig, axs = plt.subplots()
axs.plot(x_vals, noisier_data, label = 'noisier data truth value')
axs.plot(x_vals, (A_noisier @ coefs_noisier) + b_noisier, label = 'predicted noisier values')
axs.set_title("Noisy spline fit overlay, uniform noise on interval [0, 10)")
axs.set_ylabel("Value")
axs.set_xlabel("X")


#%% Generate a fit using PyTorch

'''
Note: PyTorch might take some time to run since it is a gradient descent
method. Assume that you do not know the form of the matrices A and b but not the 
values of the coefficient vectors

There are a few key parts to a PyTorch training program.

Implementing the model:
    You will need to figure out what the form of the model is. The most generic
    model is a dense fully-connected feed-forward neural network but given PyTorch's
    ability to handle dynamic computational graphs, you can build very 
    novel architectures. We will take advantage of this fact here.

Figuring out your data:
    You need to define what data you are using for training, testing, and 
    validation. This can be partitioned in different ways, such as an 80-20
    train-test split or into k-folds for cross validation

Establishing the training routine:
    Training requires the use of an optimizer, a loss function, and a training
    loop. These parts are all shown in the example below.
'''

#First, generate a fit of the model assuming that the coefficients are wrong

config = {'xknots' : np.linspace(1, 5, 50),
          'equal_knots' : False,
          'cutoff' : 5,
          'bconds' : 'none', 
          'deg' : 3,
          'max_der' : 2}

model = SplineModel(config)
_, A, b = fit_linear_model(model, x_vals, y_vals)

#Because the matrix multiply is Ax + b, we generate a dummy coefficient 
#   vector of ones

coefs = np.ones(A.shape[-1])

#Let's see what the initial fit looks like
initial_predictions = (A @ coefs) + b
fig, axs = plt.subplots()
axs.plot(x_vals, y_vals, label = 'truth value')
axs.plot(x_vals, initial_predictions, label = 'initial predictions')
axs.legend()

#We convert the coefficient vector to a tensor to allow it to become trainable.
#   tensors are the key building blocks of PyTorch as they construct the 
#   computational graph

#requires_grad specifies that the tensor should have its gradients tracked (and is therefore trainable)
#   we also convert A and b to tensors so that everything is of the same type (although PyTorch and Numpy can talk to each other)
#   As a rule of thumb, everything should be a tensor
coefs = torch.tensor(coefs, requires_grad = True)
A, b = torch.tensor(A), torch.tensor(b)
y_vals = torch.tensor(y_vals)

#The model parameters have to be an iterable of tensors before being passed into the 
#   optimizer
model_parameters = [coefs]

#We bind the trainable parameters of the model to the Adam optimizer and use the
#   default values here (e.g. learning rate, alpha/beta parameters)
optimizer = Adam(model_parameters)

#We will use the MSE loss for computing our loss and backpropagating
criterion = nn.MSELoss()

#Specify some of the constants of this training process
nepochs = 75_000 #The number of iterations to do for the training loop
batch_size = 10 #Not used here, but will be relevant when we handle more complex data that requires batching

epoch_loss = []

#Begin the training loop
for i in range(nepochs):
    optimizer.zero_grad() #Zero all the gradients (this is important! Don't forget to do this)
    output = (A @ coefs) + b
    loss = criterion(output, y_vals)
    #Print the loss every 1000 epochs
    if i % 1000 == 999:
        print(f"Loss for epoch {i + 1} is {loss.item()}")
    #Plot the predicitons every 10,000 epochs
    if i % 10_000 == 9_999:
        print(f"Visualizing the predictions for epoch {i + 1}")
        curr_predictions = (A @ coefs) + b
        curr_predictions = curr_predictions.detach().numpy()
        fig, axs = plt.subplots()
        axs.plot(x_vals, y_vals, label = 'truth value')
        axs.plot(x_vals, curr_predictions, label = 'current predictions')
        axs.legend()
    epoch_loss.append(loss.item())
    loss.backward() #This .backward() call performs the backpropagation
    optimizer.step() #This performs the update of the coefficient vector
    
#Let's see how the fit looks like now
final_predictions = (A @ coefs) + b
final_predictions = final_predictions.detach().numpy() #Convert to Numpy
fig, axs = plt.subplots()
axs.plot(x_vals, y_vals, label = 'truth value')
axs.plot(x_vals, final_predictions, label = 'final predictions')
axs.legend()

#What you might notice is that the quality of the fit degrades considerably with
#   fewer training epochs. You want to train for a larger number of epochs to obtain
#   the truth value. Through some experimentation, 75,000 epochs was found to work well.
#   Try lowering the number and seeing what happens


#%% Try fitting an arbitrary function!



