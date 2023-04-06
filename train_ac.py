import numpy as np
import torch
import math
from pinn_ac import PINN_AC
import random
import scipy.io
from pyDOE import lhs
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import grad,Variable



def get_ac_data(self, q=100, t0=20, t1=180, N=200, noise=0.0):

    data = scipy.io.loadmat('AC.mat')
    t_star = data['tt'].flatten()[:,None] # T x 1
    x_star = data['x'].flatten()[:,None] # N x 1
    exact = np.real(data['uu']).T # T x N
    
    idx_x = np.random.choice(exact.shape[1], N, replace=False) 
    x0_train = x_star[idx_x,:]
    x0 = torch.from_numpy(np.random.normal(x0_train, noise*np.std(x0_train), (x0_train.shape[0], x0_train.shape[1]))).float() #torch.from_numpy(x_star[idx_x,:]).float()
    u0 = exact[t0:t0+1,idx_x].T
    u0 = torch.from_numpy(u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])).float()
    
    return torch.from_numpy(x_star), t_star, x0, u0, torch.from_numpy(exact).float(), torch.from_numpy(np.vstack((-1., 1.)))


def train_ac(epoch, x0, x1, u0, model, optimizer, criterion):
    collocation = 20000
    batch_size = 32
    num_batch = 10
    batches = collocation // batch_size
    permutation = torch.randperm(x0.size()[0])
    loss = 0
    model.train()
    for i in range(0, x0.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        x0_batch = x0[indices]
        y = u0[indices]
    
        def closure():
            y_model= model.forward_ac(x0_batch)
            loss1 = model.loss_ac(x0_batch, x1)
            loss2= criterion(y_model, y) 
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            return loss
        loss += optimizer.step(closure)
    
    #uncomment to check if weights are updating
    #b = list(model.parameters())[0].clone() #check to see if weights are updating
    #print(torch.equal(a.data, b.data))
    
    
    print(" epoch = ", epoch, "  loss=", loss)
    
def validate_ac(epoch, x_val, u_val, t1, model):
    model.eval()
    x = x_val[:]
    u_pred = model.forward_ac(x)

    #error = np.linalg.norm(u_pred[:,-1].detach().numpy() - u_val[:,-1].detach().numpy(), 2) / np.linalg.norm(u_val[:,-1].detach().numpy(), 2)
    error =np.linalg.norm(u_pred[:,-1].detach().numpy() - u_val[t1,:].detach().numpy(), 2)/np.linalg.norm(u_val[t1,:].detach().numpy(), 2)
    
    print(" epoch= ", epoch, "  Error  = ", error)

    return u_pred[:,-1].detach().numpy()


def main():
    
    #network architecture from paper
    num_dim = 1 
    hidden = 128 #200
    layers = 6
    q = 100
    lb = np.array([-1.0])
    ub = np.array([1.0])
    t0 = 20
    t1 = 180

    #tunable parameters
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)
    num_epochs= 70000 #10000 is not enough epochs to see any sort of result, try upwards of 50000 
    alpha= 0.0001 #2e-03
    noise = 0.0
    
    
    #x_star, t_star, x0_train, x0_val, u0_train, u0_val, exact, x1 = get_ac_data(noise)
    #Raissi's version of splitting data
    x_star, t_star, x0, u0, exact, x1 = get_ac_data(noise) 
    dt = t_star[t1] - t_star[t0]
    
    model = PINN_AC(dt, ub, lb, q, num_dim, hidden)
    
    criterion =  torch.nn.MSELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    #optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    
    for epoch in range(num_epochs):
        #train_ac(epoch, x0_train, x1, u0_train, model, optimizer, criterion)
        train_ac(epoch, x0, x1, u0, model, optimizer, criterion) #for Raissi's version of sampling/splitting data
    #u_pred = validate_ac(epoch, x0_val, u0_val, t1, model)
    #print(u_pred)

    
    u_pred= validate_ac(epoch, x_star, exact, t1, model) #Raissi's version of sampling/splitting data
    print(u_pred)
    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star, exact[t0,:], 'b-', linewidth = 2) 
    #ax.plot(x0_train, u0_train, 'rx', linewidth = 2, label = 'Data') 
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')  #Raissi's version of splitting data  
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t_star[t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star,exact[t1,:], 'b-', linewidth = 2, label = 'Exact') 
    #ax.plot(x0_val, u_pred, 'r--', linewidth = 2, label = 'Prediction') 
    ax.plot(x_star, u_pred, 'r--', linewidth = 2, label = 'Prediction')  #Raissi's version of splitting data  
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t_star[t1]), fontsize = 10)    
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    
    #uncomment to save plot
    plt.savefig('AC.png')
      

if __name__=="__main__":
    main()