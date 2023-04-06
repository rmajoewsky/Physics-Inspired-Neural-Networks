import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad,Variable
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Korteweg-De Vries Equation, a mathematical model of waves on shallow water surfaces

class PINN_KDV(nn.Module):
    def __init__(self, dt, ub, lb, q, num_dim, hidden, layers):

        super(PINN_KDV, self).__init__()

        #initialize parameters
        self.lambda_1 = Variable(torch.Tensor([0.0]))
        self.lambda_2 = Variable(torch.Tensor([-6.0]))
        self.dt = dt
        self.ub = torch.tensor(ub)
        self.lb = torch.tensor(lb)
        self.q = q
        #cite later

        num_layers = 6 #from paper
        self.num_neurons = 50 #from paper 
        self.num_dim= num_dim
        self.num_out= q
        # add layers
        self.layers=[]
        in_size= self.num_dim
        self.layers= nn.ModuleList()
        self.num_layers=0
        for i in range(0, num_layers-1):
            self.layers.append(nn.Linear(in_size, self.num_neurons))
            torch.nn.init.xavier_uniform(self.layers[self.num_layers].weight)
            self.layers.append(nn.Tanh())
            #self.layers.append(nn.ReLU())
            self.num_layers+=2
            in_size=self.num_neurons

        # last layer
        self.layers.append(nn.Linear(in_size, self.num_out))
        torch.nn.init.xavier_uniform(self.layers[self.num_layers].weight)
        self.layers.append(nn.Tanh())
        self.num_layers+=2

        butcher_text = np.loadtxt('Butcher_IRK50.txt', dtype = np.float32, ndmin = 2)
        weights = np.reshape(butcher_text[0:q**2 + q], (q+1,q))
        self.irk_alpha = torch.from_numpy(weights[0:-1,:])
        self.irk_beta = torch.from_numpy(weights[-1:,:])
        self.irk_time = torch.from_numpy(butcher_text[q**2 + q:])

        print("layers=", self.layers)
        print(" num layers=", self.num_layers)

    def forward_kdv(self, x1, x2):
        
        H1 = (2.0*(x1 - self.lb)/(self.ub - self.lb) - 1.0).float() 
        U1 = self.layers[0](H1)
        for i in range(1,self.num_layers):
           U1 = self.layers[i](U1)
        #print("U.type ", U)

        H2 = (2.0*(x2 - self.lb)/(self.ub - self.lb) - 1.0).float() 
        U2 = self.layers[0](H2)
        for i in range(1,self.num_layers):
           U2 = self.layers[i](U2)
        #print("U.type ", U)
        
        
        #print("U1, U2 ", U1, U2)
        return U1, U2

    def gradients(self, x, U):
        z = torch.ones(U.shape).requires_grad_(True)
        g = grad(outputs=U, inputs=x, grad_outputs=z, create_graph=True)[0]

        #print("gradient check ", grad(outputs=g, inputs=z, grad_outputs=torch.ones(g.shape), create_graph=True, allow_unused=True)[0])

        return grad(outputs=g, inputs=z, grad_outputs=torch.ones(g.shape), create_graph=True)[0]



    def loss_kdv(self, x1, x2):
        x_var1=Variable(x1, requires_grad=True)
        x_var2=Variable(x2, requires_grad=True)
        U1, U2 = self.forward_kdv(x_var1, x_var2)
        U = U1 
        U_X = self.gradients(x_var1, U)
        #print("U_X1 ", U_X)
        U_XX = self.gradients(x_var1, U_X)
        #print("U_XX1 ", U_XX)
        U_XXX = self.gradients(x_var1, U_XX)
        #print("U_XXX1 ", U_XXX)
        lambda2 = torch.exp(self.lambda_2)
        N = -self.lambda_1*U*U_X - lambda2*U_XXX
        U0 = U1 - self.dt * torch.mm(N, self.irk_alpha.T)

        U = U2 
        U_X = self.gradients(x_var2, U)
        #print("U_X2 ", U_X)
        U_XX = self.gradients(x_var2, U_X)
        #print("U_XX2 ", U_XX)
        U_XXX = self.gradients(x_var2, U_XX)
        #print("U_XXX2 ", U_XXX)
        lambda2 = torch.exp(self.lambda_2)
        N = -self.lambda_1*U*U_X - lambda2*U_XXX
        U_x = U2 + self.dt * torch.mm(N, (self.irk_beta - self.irk_alpha).T)
        
        loss = ((U1 - U0)**2).sum() + ((U2 - U_x)**2).sum() 
        return loss

   

