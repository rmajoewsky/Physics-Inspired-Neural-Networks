import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad,Variable




class PINN_AC(nn.Module):
    def __init__(self, dt, ub, lb, q, num_dim, hidden):

        super(PINN_AC, self).__init__()

        #initialize parameters
        self.dt = torch.tensor(dt)
        self.ub = torch.tensor(ub)
        self.lb = torch.tensor(lb)
        self.q = q
       

        num_layers = 6 #from paper
        self.num_neurons = 200 #from paper 
        self.num_dim= num_dim #1
        self.num_out= q+1
        # add layers
        self.layers=[]
        in_size= self.num_dim
        self.layers= nn.ModuleList()
        self.num_layers=0
        for i in range(0, num_layers-1):
            self.layers.append(nn.Linear(in_size, self.num_neurons))
            torch.nn.init.xavier_uniform_(self.layers[self.num_layers].weight)
            self.layers.append(nn.Tanh())
            #self.layers.append(nn.ReLU())
            self.num_layers+=2
            in_size=self.num_neurons

        # last layer
        self.layers.append(nn.Linear(in_size, self.num_out))
        torch.nn.init.xavier_uniform(self.layers[self.num_layers].weight)
        self.layers.append(nn.Tanh())
        self.num_layers+=2

        butcher_text = np.loadtxt('Butcher_IRK100.txt', dtype = np.float32, ndmin = 2)
        self.irk_weights = torch.from_numpy(np.reshape(butcher_text[0:q**2 + q], (q+1,q)))
        self.irk_time = torch.from_numpy(butcher_text[q**2 + q:])
        

        #print("WEIGHTS ", self.irk_weights)

        print("layers=", self.layers)
        print(" num layers=", self.num_layers)


    def forward_ac(self, x):

        #print("x.shape ", x.shape)
        
        H = (2.0*(x - self.lb)/(self.ub - self.lb) - 1.0).float() 
        U = self.layers[0](H)

        for i in range(1,self.num_layers):
           U = self.layers[i](U)
        return U #[:,:-1] 

    
    def gradients(self, x, U):
        z = torch.ones(U.shape).requires_grad_(True)
        g = grad(outputs=U, inputs=x, grad_outputs=z, create_graph=True)[0]
        #print(grad(outputs=g, inputs=z, grad_outputs=torch.ones(g.shape), create_graph=True)[0])
        return grad(outputs=g, inputs=z, grad_outputs=torch.ones(g.shape), create_graph=True)[0]

    def loss_ac(self, x0, x1):
        x_var=Variable(x0, requires_grad=True)
        U1 = self.forward_ac(x_var)
        #print("U1.shape ", U1.shape)
        U = U1[:,:-1] 
        #print("U.shape ", U)
        U_X = self.gradients(x_var, U)
        U_XX = self.gradients(x_var, U_X)
        
        F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_XX
        U0 = U1 - self.dt * torch.mm(F, self.irk_weights.T)
        
        x_var1 = Variable(x1, requires_grad=True)
        U2 = self.forward_ac(x_var1)
        #print("U2.shape ", U2.shape)
        U2_grads = self.gradients(x_var1, U2)
        
        loss = ((U1 - U0)**2).sum() + ((U2[0,:] - U2[1,:])**2).sum() + ((U2_grads[0,:] - U2_grads[1,:])**2).sum()

        return loss

    