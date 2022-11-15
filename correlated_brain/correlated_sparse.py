# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:52:53 2022

@author: Sun
"""

import numpy as np
import torch
import hamiltorch
import matplotlib.pyplot as plt
import h5py

filepath = 'C:/Users/laosu/Dropbox (UFL)/Test_Subjects/SBCI_SC_6sub_hcp_1.mat'
sc6 = {}
f = h5py.File(filepath)
for k, v in f.items():
    sc6[k] = np.array(v)
    
filepath = 'C:/Users/laosu/Dropbox (UFL)/Test_Subjects/SBCI_FC_4run_6sub_hcp_1.mat'
fc46 = {}
f = h5py.File(filepath)
for k, v in f.items():
    fc46[k] = np.array(v)
    
sc = sc6['sbci_sc_tensor'][0]
fc = fc46['sbci_fc_tensor'][0][0]
y = torch.from_numpy(fc)
y = y.float()
y[y!=y]=0


S = torch.from_numpy(sc[900:1900,900:1900])
y = y[900:1900,900:1900]
S = S.float()
U, D, V = S.svd()
d=2
n = S.shape[0]

S1 = (U[:,0:d]*10.@(V[:,0:d]).T) + torch.eye(n) 
Sinv =  S1.inverse()
Sinv.det()


from matplotlib.colors import DivergingNorm
norm = DivergingNorm(vmin=y.min(), vcenter=0, vmax=y.max())
col='bwr'
plt.matshow((S1 - torch.eye(n)).abs()*2, norm=norm,cmap=col)
plt.colorbar()

sl = .75

softplus = torch.nn.functional.softplus
def logjac_softplus(x):
    return x - softplus(x)

def proj_l1_ball(beta, r):
    p = beta.shape[0]
    
    abs_w = torch.abs(beta)
    sorted_w, argsort_w = abs_w.sort(descending = True)

    mu = torch.zeros(p)
    K_ = torch.zeros(p)
    
    mu = sorted_w.cumsum(0) -r 
    mu = mu*(mu>0)
    
    scaled_mu = mu/ (torch.arange(p)+1)
    c = (sorted_w > scaled_mu).sum().type(torch.int64) -1 
    
    mu_k = mu[c]
    scaled_mu_k = mu[c]/(c + 1)
    eta = torch.zeros(p)
    s = torch.sign(beta)
#     print(scaled_mu_k)
    
    t = abs_w- scaled_mu_k
    eta = s.t()*torch.max(t.t(), torch.zeros(p))
    return eta



def llh(params):
    beta = params[0: d*n].reshape([d,n])
    #r = softplus(params[d*n:d*n+d])
    mu = beta.abs().quantile(sl, 1)
    llh = 0
    theta = torch.zeros_like(beta)
    for i in range(d):
        llh += - beta[i] @ Sinv @  beta[i] / .1
        #theta[i] = proj_l1_ball(beta[i], r[i])
        theta[i] = (beta[i].abs() - mu[i]) * ((beta[i].abs() - mu[i])>0) * beta[i].sign()
    llh += - ((y - theta.T @theta) ** 2 ).sum() 
    #llh += - torch.log(1.0+ r ** 2).sum() + (logjac_softplus(params[d*n:d*n+d])).sum()
    #llh += -r.sum() + (logjac_softplus(params[d*n:d*n+d])).sum()
    return(llh)


params = torch.normal(torch.zeros(d*n + d)) *.01
#params[d*n:d*n+d] = 50.
params = params.requires_grad_()
lr = .01

optimizer = torch.optim.Adam([params], lr=lr)
for epoch in range(2000):
     loss = -llh(params)
     optimizer.zero_grad()
     loss.backward(retain_graph=True)
     optimizer.step()
     if epoch % 100 ==1:
         optimizer.param_groups[0]['lr'] = lr / epoch ** .9
         print(loss.item())
         
         
beta = params[0: d*n].reshape([d,n])
theta = torch.zeros_like(beta)
r = softplus(params[d*n:d*n+d])
mu = beta.abs().quantile(sl, 1)
for i in range(d):
    #llh += - beta[i] @ Sinv @  beta[i] / 2. / eta2
    #theta[i] = proj_l1_ball(beta[i], r[i])
    theta[i] = (beta[i].abs() - mu[i]) * ((beta[i].abs() - mu[i])>0) * beta[i].sign()

plt.matshow((theta.T@theta).detach().numpy(), norm=norm,cmap=col)
plt.colorbar()
plt.matshow(y, norm=norm,cmap=col)
plt.colorbar()

sl = .75
#========================================================================================
def llh1(params):
    beta = params[0: d*n].reshape([d,n])
    #r = softplus(params[d*n:d*n+d])
    llh = 0
    mu = beta.abs().quantile(sl, 1)
    theta = torch.zeros_like(beta)
    for i in range(d):
        llh += - beta[i] @  beta[i]
        #theta[i] = proj_l1_ball(beta[i], r[i])
        theta[i] = (beta[i].abs() - mu[i]) * ((beta[i].abs() - mu[i])>0) * beta[i].sign()
    llh += - ((y - theta.T @theta) ** 2 ).sum() 
    #llh += - torch.log(1.0+ r ** 2).sum() + (logjac_softplus(params[d*n:d*n+d])).sum()
    return(llh)

params1 = torch.normal(torch.zeros(d*n + d)) *.1
params1[d*n:d*n+d] = 50.
params1 = params1.requires_grad_()
a = .01
lr = .01

optimizer = torch.optim.Adam([params1], lr=lr)
for epoch in range(2000):
     loss = -llh1(params1)
     optimizer.zero_grad()
     loss.backward(retain_graph=True)
     optimizer.step()
     if epoch % 100 ==1:
         optimizer.param_groups[0]['lr'] = lr / epoch ** .9
         print(loss.item())
         
beta1 = params1[0: d*n].reshape([d,n])
theta1 = torch.zeros_like(beta1)
r1 = softplus(params1[d*n:d*n+d])
mu = beta1.abs().quantile(sl, 1)
for i in range(d):
    theta1[i] = (beta1[i].abs() - mu[i]) * ((beta1[i].abs() - mu[i])>0) * beta1[i].sign()

plt.matshow((theta1.T@theta1).detach().numpy(), norm = norm, cmap=col)
plt.colorbar()



inv_mass = (params + 1e-4) ** 2#torch.ones(p+1+1)
params_init=params
step_size = .00001
num_samples = 1000 # For results in plot num_samples = 12000
L = 100
burn = 1 # For results in plot burn = 2000
hamiltorch.set_random_seed(123)
params_hmc_nuts = hamiltorch.sample(log_prob_func=llh,
                                    params_init=params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L,
                                    desired_accept_rate=0.6,
                                    sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn,
                                    inv_mass = inv_mass
                                   )

param_trace = torch.vstack(params_hmc_nuts)
theta_ = []
for _ in param_trace:         
    beta = _[0: d*n].reshape([d,n])
    theta = torch.zeros_like(beta)
    #r = softplus(_[d*n:d*n+d])
    mu = beta.abs().quantile(sl, 1)
    for i in range(d):
        #llh += - beta[i] @ Sinv @  beta[i] / 2. / eta2
        #theta[i] = proj_l1_ball(beta[i], r[i])
        theta[i] = (beta[i].abs() - mu[i]) * ((beta[i].abs() - mu[i])>0) * beta[i].sign()
    theta_.append(theta)
theta_ = torch.stack(theta_)
theta_ = theta_.detach().numpy()
trace_np = param_trace.detach().cpu().numpy()

cov = np.corrcoef(np.transpose(theta_[:,0,:]))
thetaiszero = np.abs(theta_) > .0001
cov = np.corrcoef(np.transpose(thetaiszero[:,0,:]))
plt.matshow(cov)
plt.colorbar()


inv_mass = (params1 + 1e-4) ** 2#torch.ones(p+1+1)
params_init=params1
step_size = .00001
num_samples = 1000 # For results in plot num_samples = 12000
L = 100
burn = 1 # For results in plot burn = 2000
hamiltorch.set_random_seed(123)
params_hmc_nuts1 = hamiltorch.sample(log_prob_func=llh1,
                                    params_init=params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L,
                                    desired_accept_rate=0.6,
                                    sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn,
                                    inv_mass = inv_mass
                                   )

param_trace1 = torch.vstack(params_hmc_nuts)
theta_1 = []
for _ in param_trace:         
    beta = _[0: d*n].reshape([d,n])
    theta = torch.zeros_like(beta)
    mu = beta.abs().quantile(sl, 1)
    for i in range(d):
        #llh += - beta[i] @ Sinv @  beta[i] / 2. / eta2
        #theta[i] = proj_l1_ball(beta[i], r[i])
        theta[i] =  (beta[i].abs() - mu[i]) * ((beta[i].abs() - mu[i])>0) * beta[i].sign()
    theta_1.append(theta)
theta_1 = torch.stack(theta_1)
theta_1 = theta_1.detach().numpy()
theta1iszero = np.abs(theta_1) > .0001
cov1 = np.corrcoef(np.transpose(theta_1[:,0,:]))
plt.matshow(cov1)
plt.colorbar()