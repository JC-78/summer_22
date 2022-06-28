# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:07:14 2022

@author: fhu14
"""

'''
Generally, I like to partition a code file/module into three sections:
    1) Imports, definitions: exactly as it sounds, this is where you 
        import all the packages you need 
    2) Code behind: this is where you write all your functions, classes,
        and anything else you need
    3) Main block: this is where you call your code if this is a top-level
        module. Otherwise, this can be excluded.
'''

#%% Imports, definitions
from Auorg_1_1 import ParDict
from Spline import get_dftb_vals, SplineModel, fit_linear_model, PairwiseLinearModel
from MasterConstants import Model
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
Tensor = torch.Tensor


class DiscontinousModel(PairwiseLinearModel):
    def __init__(self, config, xeval=None):
        # if isinstance(xeval, (int, float)):
        #     xeval = np.array([xeval])
        # self.xknots = config['xknots']
        # if 'deg' in config:
        #     self.deg = config['deg']
        # else:
        #     self.deg = 3
        # if 'bconds' in config:
        #     bconds = config['bconds']
        #     if isinstance(bconds, list):
        #         # list of Bcond or an empty list for no boundary conditions
        #         if all(isinstance(x, Bcond) for x in bconds):
        #             self.bconds = bconds
        #         else:
        #             raise ValueError('Spline bconds is list with unknown types')
        #     elif bconds == 'natural':
        #         # natural at start and end
        #         self.bconds = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)]
        #     elif bconds == 'vanishing':
        #         # natural at start point, zero value and derivative at end point
        #         self.bconds = [Bcond(0, 2, 0.0),
        #                        Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)]
        #     elif bconds == 'none':
        #         self.bconds = []
        #     elif bconds == 'last_only':
        #         #Zero value and derivative at the end point only
        #         self.bconds = [Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)]
        #     elif bconds == 'third_deriv_testing':
        #         self.bconds = [Bcond(-1, 1, 0)]
        # else:
        #     # Natural boundary conditions
        #     self.bconds = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)]
        # if 'max_der' in config:
        #     self.max_der = config['max_der']
        # else:
        #     self.max_der = 0
        # if xeval is not None:
        #     self.spline_dict = spline_linear_model(self.xknots, xeval, None,
        #                                            self.bconds, self.max_der,
        #                                            self.deg)
        # else:
        #     self.spline_dict = None
        self.lowwer_bound = config['xknots'][0]
        self.upper_bound  = config['xknots'][-1]


    def r_range(self):
        # return self.xknots[0], self.xknots[-1]
        self.lower_bound, self.upper_bound

    def nvar(self):
        # if self.spline_dict is None:
        #     # force initialization of spline_dict
        #     _ = self.linear_model(self.xknots)
        # return self.spline_dict['X'][0].shape[1]
        return 2

    def linear_model(self, xeval, ider=0):
        # # TODO: This may not be optimal, especially when derivatives above 0 
        # #       are being requested
        # if (ider > self.max_der):
        #     raise ValueError('linear_model requests derivative above max')
        # if isinstance(xeval, (int, float)):
        #     xeval = np.array([xeval])
        # if self.spline_dict is None:
        #     # dictionary was never initialized
        #     self.spline_dict = spline_linear_model(self.xknots, xeval, None,
        #                                            self.bconds, self.max_der,
        #                                            self.deg)
        # elif (len(self.spline_dict['X']) > ider) and \
        #         (len(self.spline_dict['xvals']) == len(xeval)) and \
        #         all(x == y for x, y in zip(self.spline_dict['xvals'], xeval)):
        #     # dictionary was evaluated at the same points as requested
        #     # and up to the required derivative
        #     pass
        # else:
        #     # need to evaluate at a new set of points
        #     # nder = ider + 1 because of use of range inside
        #     self.spline_dict = spline_new_xvals(self.spline_dict, xeval, nder=ider + 1)

        # return self.spline_dict['X'][ider], self.spline_dict['const'][ider]
        
        # y = m x + c   = x_exal p[0] +1 p[1] = A @ p + const
        #   A(:,0) = x_eval
        #   A(:,1) = 1
        neval = len(xeval)
        A = np.zeros((neval, 2))
        const = np.zeros(neval)
        for i in range(neval):
            A[i,0] = xeval[i]
            A[i,1] = 1
        
        return A, const
        
        

#%% Code behind
#Possible interactions for non-hydrogen atoms
#   ss: interaction between two 2s orbitals
#   sp: interaction between 2s orbital and 2p orbital
#   pp_sigma: pp interaction with sigma mode (along the internuclear axis)
#   pp_pi: pp interaction with pi mode (perpendicular to the atomic plane)

#Specify a model
mod_spec = Model("H", (6, 6), 'pp_sigma')
par_dict = ParDict()
r_grid = np.linspace(1, 5, 100)
y_vals = get_dftb_vals(mod_spec, par_dict, r_grid)

#Start with a visualization of what the SKF function looks like previous to any
#   fitting

fig, axs = plt.subplots()
axs.plot(r_grid, y_vals)
axs.set_xlabel("Angstroms")
axs.set_ylabel("Hartrees")
axs.set_title(str(mod_spec))

# Try fitting a series of different models with different interactions using:
#   linear least squares
#   PyTorch

def fit_lst_sq_function(model: Model, config: Dict, r_bound: Tuple,
                        grid_density: int = 100, par_dict: Dict = ParDict()) -> None:
    """
    Perform a least-squares fit to a skf function specified by the input
        Model object
    
    Arguments:
        model (Model): The tuple specifying the type of model to fit
        config (Dict): The dictionary specifying the configuration of the 
            spline model
        r_bound (Tuple): Format of (r_low, r_high) which is used to generate the 
            r_grid used for fitting
        grid_density (int): The number of points to generate for r_grid based
            on the given values for r_low, r_high. Defaults to 100
        par_dict (Dict): The parameter dictionary for a set of SKFs. Defaults
            to the parameter dictionary for the Auorg SKF set

    Returns:
        None

    Notes: This function does not return anything but does display the results of 
        the function fitting. This is done using the .linear_model() method contained
        in the SplineModel() class.
    """
    #Initialize the model and the data for fitting
    spl_mod = DiscontinousModel(config)
    r_low, r_high = r_bound
    r_grid = np.linspace(r_low, r_high, grid_density)
    y_vals = get_dftb_vals(model, par_dict, r_grid)
    coefs, A, b = fit_linear_model(spl_mod, r_grid, y_vals)
    predicted_values = (A @ coefs) + b
    fig, axs = plt.subplots()
    axs.plot(r_grid, y_vals, label = 'truth value of skf')
    axs.plot(r_grid, predicted_values, label = 'predicted value of skf')
    axs.legend()
    axs.set_xlabel("Angstroms")
    axs.set_ylabel("Hartrees")
    plt.show()
    #Also, it's useful to see the MAE between the predictions and the 
    #   truth value
    MAE = np.mean(np.abs(predicted_values - y_vals))
    print(f"The MAE is {MAE}")

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

def fit_pytorch_function(model: Model, config: Dict, r_bound: Tuple,
                         grid_density: int = 100, par_dict: Dict = ParDict(), 
                         n_epochs: int = 75_000, apply_constraint: bool = False,
                         constraint_weight: float = 0.1) -> None:
    r"""Fits an SKF function using PyTorch and gradient descent. 
    
    Arguments:
        model (Model): The tuple specifying the type of model to fit
        config (Dict): The dictionary specifying the configuration of the 
            spline model
        r_bound (Tuple): Format of (r_low, r_high) which is used to generate the 
            r_grid used for fitting
        grid_density (int): The number of points to generate for r_grid based
            on the given values for r_low, r_high. Defaults to 100
        par_dict (Dict): The parameter dictionary for a set of SKFs. Defaults
            to the parameter dictionary for the Auorg SKF set
        n_epochs (int): The number of epochs to train the model for. Defaults to
            75_000
        apply_constraint (bool): Whether to apply the second derivative sign constraint.
            Defaults ot False. 
        constraint_weight (float): The weight multiplied for the second derivative constraint.
            Defaults to 0.1 
    
    Returns:
        None
    
    Notes: This function does not return a value but does display the results 
        of fitting an SKF function using PyTorch and gradient descent. 
    """
    #Some initialization
    spl_mod = SplineModel(config)
    r_low, r_high = r_bound
    r_grid = np.linspace(r_low, r_high, grid_density)
    #y_vals is our truth value
    y_vals = get_dftb_vals(model, par_dict, r_grid)
    coefs, A, b = fit_linear_model(spl_mod, r_grid, y_vals)
    if apply_constraint:
        #Obtain the values for the second derivative
        A_2, b_2 = spl_mod.linear_model(r_grid, 2)
    #Set the coefs to a dummy vector to demonstrate the effectiveness of the
    #   fitting algorithm
    A, b = torch.tensor(A), torch.tensor(b)
    y_vals = torch.tensor(y_vals)
    coefs = torch.ones(A.shape[-1], requires_grad = True, dtype = A.dtype)
    if apply_constraint:
        A_2, b_2 = torch.tensor(A_2), torch.tensor(b_2)
    #Create an optimizer instance and criterion 
    optimizer = Adam([coefs])
    criterion = nn.MSELoss()
    #Start training loop process
    for i in range(n_epochs):
        optimizer.zero_grad()
        output = (A @ coefs) + b
        loss = criterion(output, y_vals)
        if apply_constraint:
            form_loss = compute_form_loss(A_2, b_2, coefs, weight = constraint_weight)
            loss += form_loss
        #Print the loss every 1000 epochs
        if i % 100 == 99:
            print(f"Loss for epoch {i + 1} is {loss.item()}")
        #Plot the predicitons every 10,000 epochs
        if i % 1000 == 999:
            print(f"Visualizing the predictions for epoch {i + 1}")
            curr_predictions = output.detach().numpy()
            fig, axs = plt.subplots()
            axs.plot(r_grid, y_vals, label = 'truth value')
            axs.plot(r_grid, curr_predictions, label = 'current predictions')
            axs.set_xlabel("Angstroms")
            axs.set_ylabel("Hartrees")
            axs.set_title(f"Spline predictions at epoch {i + 1}")
            axs.legend()
            plt.show()
        loss.backward() #Performs backpropagation
        optimizer.step()
    #Show the final predictions
    
    final_predictions = (A @ coefs) + b
    final_predictions = final_predictions.detach().numpy() #Convert to Numpy
    fig, axs = plt.subplots()
    axs.plot(r_grid, y_vals, label = 'truth value')
    axs.plot(r_grid, y_vals, label = 'final predictions')
    axs.set_xlabel("Angstroms")
    axs.set_ylabel("Hartrees")
    axs.set_title("Final spline predictions")
    axs.legend()
    plt.show()
    
    print("PyTorch fitting completed")

#%% Main block
if __name__ == "__main__":
    #Define the configuration dictionary
    config = {'xknots' : np.linspace(1, 5, 50), #The position of the control points for the spline
              'equal_knots' : False, #Whether or not to have equal knots in the two halves of a joined spline (disregard)
              'cutoff' : 5, #The cutoff distance for the spline
              'bconds' : 'none',  #The boundary conditions for the spline
              'deg' : 3, #The degree of the spline. Here, we are using cubic (3rd degree) splines
              'max_der' : 2} #The maximum derivative of the spline evaluated. The maximum derivative possible is the spline degree - 1
    
    #Define the model and pardict that we want to use along with some
    #   constants
    mod_spec = Model("H", (6, 6), 'pp_sigma')
    par_dict = ParDict()
    r_bound = (1, 5) #r_low, r_high
    grid_density = 100
    nepochs = 3_000
    
    #Start with a least squares fit example
    fit_lst_sq_function(mod_spec, config, r_bound, grid_density = grid_density, 
                        par_dict = par_dict)
    
    #Then do a PyTorchh fit
    fit_pytorch_function(mod_spec, config, r_bound, grid_density, par_dict, nepochs, 
                         apply_constraint = False)