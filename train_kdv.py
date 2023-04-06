import numpy as np
import torch
import math
from pinn_kdv import PINN_KDV
import random
import scipy.io
from pyDOE import lhs
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import grad,Variable
from tabulate import tabulate
#import texttable
from pandas.plotting import table


def get_kdv_data(self, q=50, t=40, N0=199, N1=201, skip=120, noise=0.0):

    #print("t + skip ", t + skip)

    data = scipy.io.loadmat('KdV.mat')
    t_star = data['tt'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    exact = np.real(data['uu'])
    
    idx_x0 = np.random.choice(exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x0,:]
    x0 = torch.from_numpy(np.random.normal(x0, noise*np.std(x0), (x0.shape[0], x0.shape[1]))).float() 
    u0 = exact[idx_x0, t][:,None]
    u0 = torch.from_numpy(np.random.normal(u0, noise*np.std(u0), (u0.shape[0], u0.shape[1]))).float()

    idx_x1 = np.random.choice(exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x1,:]
    x1 = torch.from_numpy(np.random.normal(x1, noise*np.std(x1), (x1.shape[0], x1.shape[1]))).float() 
    u1 = exact[idx_x1, int(t+skip)][:,None]
    u1 = torch.from_numpy(np.random.normal(u1, noise*np.std(u1), (u1.shape[0], u1.shape[1]))).float()
    

    return x0, u0, x1, u1, x_star, t_star, exact


def train_kdv(epoch, input0, input1, target0, target1, model, optimizer, criterion):
    model.train()

    x0 = input0[:]
    x1 = input1[:]
    y0 = target0[:]
    y1 = target1[:]
    

    def closure():
        y0_model, y1_model = model.forward_kdv(x0, x1)
        
        loss1= criterion(y0_model, y0)
        loss2 = criterion(y1_model, y1)
        loss3 = model.loss_kdv(x0, x1)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        return loss
    loss= optimizer.step(closure)
    print(" epoch = ", epoch, "  loss = ", loss)



def validate_kdv(epoch, x0, x1, u0, u1, model):
    model.eval()

    x = x0[:]
    x_1 = x1[:]


    u0_pred, u1_pred = model.forward_kdv(x, x_1)
    lambda1 = model.lambda_1
    lambda2 = np.exp(model.lambda_2)

    
    error_lambda1 = np.abs(lambda1 - 1.0)/1.0 *100
    error_lambda2 = np.abs(lambda2 - 0.0025)/0.0025 * 100
    
    

    print(" epoch= ", epoch, "  Error lambda 1 = ", error_lambda1, "Error lambda 2 = ", error_lambda2)

    return error_lambda1, error_lambda2

def run(noise=0.0):
     #network architecture from paper
    num_dim = 1 
    hidden = 50
    layers = 6
    N0 = 199
    N1 = 201

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    num_epochs= 50000
    alpha=2e-03
    epsilon = 6.53319e-23
    delta_t = 0.6
    q = 0.5 * math.log(epsilon) / math.log(delta_t) #from paper
    q = int(round(q))
    noise = 0.0
    t = 40
    skip = 120
    x0, u0, x1, u1, x_star, t_star, exact = get_kdv_data(noise)

    lb = x_star.min(0)
    ub = x_star.max(0)
    dt = np.asscalar(t_star[t+skip] - t_star[t])
    
    
    model = PINN_KDV(dt, ub, lb, q, num_dim, hidden, layers)
    criterion =  torch.nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters())
    #optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    #torch.cat((u0_train, u1_train),1) #CHECK

    for epoch in range(num_epochs):
        train_kdv(epoch, x0, x1, u0, u1, model, optimizer, criterion)
    error_lambda1, error_lambda2= validate_kdv(epoch, x0, x1, u0, u1, model)

    return x0, x1, u0, u1, exact, t_star, x_star, error_lambda1, error_lambda2


def main():

    t = 40
    skip = 120

    x0, x1, u0, u1, exact, t_star, x_star, lambda_1, lambda_2 = run(noise=0.0)
    x0, x1, u0, u1, exact, t_star, x_star, lambda_1n, lambda_2n = run(noise=0.01)
    
    gs1 = gridspec.GridSpec(1, 2)
    #gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star,exact[:,t][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d training data' % (t_star[t], u0.shape[0]), fontsize = 10)
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star,exact[:,int(t + skip)][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d training data' % (t_star[t+skip], u1.shape[0]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    #uncomment for graphs
    plt.savefig('kdv.png')

    
    plt.rcParams.update({'font.size': 36})
    df = pd.DataFrame([["Correct PDE", "u_t + uu_x + 0.0025 u_xxx = 0"], ["Identified PDE (clean data)", "u_t + %.3f uu_x + %.7f u_xxx = 0" % (lambda_1, lambda_2)], ["Identified PDE (1% noise)", "u_t + %.3f uu_x + %.7f u_xxx = 0" % (lambda_1n, lambda_2n)]])
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, df) 

    #uncomment for table
    plt.savefig('kdv_lambda_table.png', bbox_inches="tight")

if __name__=="__main__":
    main()