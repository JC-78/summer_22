# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:50:38 2022

@author: fhu14
"""

#%% Imports, definitions
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from Spline import SplineModel, fit_linear_model

Tensor = torch.Tensor

#%% Code behind

def compute_form_loss(A: Tensor, b: Tensor, coefs: Tensor,
                      weight: float = 100) -> Tensor:
    r"""Computes the form penalty loss based on the predictions of the second
        derivative
    
    Arguments: 
        A (Tensor): Spline basis matrix for second derivative
        b (Tensor): Spline basis constant vector for second derivative
        coefs (Tensor): The tensor of coefficients
        weight (float): The value to scale the form penalty loss by. Defaults to
            100
    
    Returns:
        loss (Tensor): The loss computed based on the sign of the second derivative
    
    Notes: 
        The matrix multiply takes the form y = Ax + b for a matrix-vector product.
        Here, it is y'' = (A'' @ coefs) + b''
    """
    predicted_second_der = (A @ coefs) + b
    #Use the rectified linear unit function, refer to the documentation
    #   here: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    m = torch.nn.ReLU()
    #Because we want the second derivative to be positive, we want to 
    #   penalize negative values of the second derivative. For that reason,
    #   we multiply the predictions of the second derivative by -1 so that
    #   the original negative perdictions become positive (and are included by ReLU)
    #   and the original postiive predictions become negative (and are masked by ReLU)
    predicted_second_der  = m(predicted_second_der * -1)
    #This is essentially MSE error
    error = (predicted_second_der @ predicted_second_der) / len(predicted_second_der)
    return weight * error

#Create the data for the function and apply some noise
x_vals = np.linspace(1, 5, 100) 
y_vals = [(x**2) + (5*x) + 6 for x in x_vals]
noise = np.random.rand(len(y_vals))
noisy_data = y_vals + (noise * 10) #This is our new truth value
plt.plot(x_vals, noisy_data)

#Fit the spline to this noisy data
config = {'xknots' : np.linspace(1, 5, 50), #The position of the control points for the spline
          'equal_knots' : False, #Whether or not to have equal knots in the two halves of a joined spline (disregard)
          'cutoff' : 5, #The cutoff distance for the spline
          'bconds' : 'none',  #The boundary conditions for the spline
          'deg' : 3, #The degree of the spline. Here, we are using cubic (3rd degree) splines
          'max_der' : 2} #The maximum derivative of the spline evaluated. The maximum derivative possible is the spline degree - 1

model = SplineModel(config)
coefs, A, b = fit_linear_model(model, x_vals, noisy_data)
predicted_values = (A @ coefs) + b
fig, axs = plt.subplots()
axs.plot(x_vals, noisy_data, label = 'truth value')
axs.plot(x_vals, predicted_values, label = 'predicted value')
axs.legend()
axs.set_xlabel('X')
axs.set_ylabel('Value')
axs.set_title("Overlay of predicted spline values against noisy data")

#We can now optimize this spline using PyTorch against the noisy data as the truth 
#   value
A, b = torch.tensor(A), torch.tensor(b)

#We can also get the higher derivatives using the model.linear_model() method
A_2, b_2 = model.linear_model(x_vals, 2)
A_2, b_2 = torch.tensor(A_2), torch.tensor(b_2)

# coefs = torch.ones(A.shape[-1], requires_grad = True, dtype = A.dtype)
coefs = torch.tensor([1] * A.shape[-1], requires_grad = True, dtype = A.dtype)
noisy_data = torch.tensor(noisy_data)
nepochs = 75_000
optimizer = Adam([coefs])
criterion = nn.MSELoss()

for i in range(nepochs):
    optimizer.zero_grad()
    output = (A @ coefs) + b
    loss = criterion(output, noisy_data)
    #Add another loss criterion
    form_loss = compute_form_loss(A_2, b_2, coefs, weight = 0.1)
    total_loss = loss + form_loss
    #Print the loss every 1000 epochs
    if i % 1000 == 999:
        print(f"Loss for epoch {i + 1} is {total_loss.item()}")
    #Plot the predicitons every 10,000 epochs
    if i % 10_000 == 9_999:
        print(f"Visualizing the predictions for epoch {i + 1}")
        curr_predictions = output.detach().numpy()
        fig, axs = plt.subplots()
        axs.plot(x_vals, noisy_data, label = 'truth value')
        axs.plot(x_vals, curr_predictions, label = 'current predictions')
        axs.legend()
    total_loss.backward() #Performs backpropagation
    optimizer.step()

final_predictions = (A @ coefs) + b
final_predictions = final_predictions.detach().numpy() #Convert to Numpy
fig, axs = plt.subplots()
axs.plot(x_vals, noisy_data, label = 'truth value')
axs.plot(x_vals, final_predictions, label = 'final predictions')
axs.legend()

'''
From the above, we see that the spline wiggles a lot when the data is noisy,
and there is clear overfitting of the data. As a form of regularization, we can
penalize the higher derivatives of the spline. For this example, we can try
applying a sign constraint to the second derivative since the data
is generally concave up. We want to do the following:
    1) Using the .linear_model() function of the model, generate the matrix A and
        constant vector b for the second derivative
    2) Generate predictions for the higher derivatives from those A and b vectors
        using the matrix multiply form y = Ax + b
    3) Compute the penalty of the second derivative (Hint: if we want the second derivative to always be 
        positive, consider applying ReLU to the vector of predicted values and
        calculating a loss from that. ReLU can be obtained from the torch package)
    4) Incorporate the two losses that are calculated (derivative penalty loss and the MSE loss against the 
        target values) and use that to backpropagate and update the model
Try to get a function that is smooth and free of oscillations when fitting to noisy 
data
'''
