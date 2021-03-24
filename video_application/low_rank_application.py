#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:42:48 2021

@author: maoran
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import hamiltorch
import pandas as pd

device = torch.device('cuda:1')
A =  np.array(pd.read_csv('video_matrix_small', header=None))
T=19
H=97
W=124
L_mean = np.mean(A, 1)
A = torch.from_numpy(A).type(torch.float32)
L_mean = torch.from_numpy(L_mean).type(torch.float32)
L_mean = torch.repeat_interleave(L_mean, T).reshape(H*W,T)
lam_S = 30.
lam_L = 5000.
def prox_S(S, lam_S):
    return((torch.abs(S)- lam_S) * ((torch.abs(S)-lam_S)>0)*torch.sign(S))

def prox_L(L, lam_L):
    u, s, v = torch.svd(L)
    s = (torch.abs(s)- lam_L) * ((torch.abs(s)-lam_L)>0)*torch.sign(s)
    pX = u@torch.diag(s)@v.T
    return(pX)

def prior_lam_L(lam, L):
    u,s,v = torch.svd(L)
    r = torch.sum((s-lam)*((s-lam)>0))
    return(-r*100)

def prior_lam_S(lam, S):
    r = torch.sum((S-lam)*((S-lam)>0))
    return(-r*1000)

def prior_S(S):
    return(-torch.norm(S) ** 2)
    
def llh(param, lam_S=lam_S, lam_L=lam_L):
    S = (param[0:H*W*T].reshape((H*W, T)))
    L = (param[H*W*T:H*W*T*2].reshape((H*W,T)))
    #lam_S = param[H*W*T*2+1]
    #lam_L = param[H*W*T*2]
    #print((-torch.norm((A-prox_L(L, lam_L)-prox_S(S,lam_S).reshape(H*W,T))) ** 2, prior_lam_L(lam_L, L), prior_lam_S(lam_S, S)))
    return -torch.norm((A-prox_L(L, lam_L)-prox_S(S,lam_S).reshape(H*W,T))) ** 2  + prior_S(S) + prior_S(L) + prior_lam_L(lam_L, L) + prior_lam_S(lam_S, S)

def mh_SL(params, lamS, lamL, step_size):
    Lnew = lamL + (torch.rand(1)-.5) * step_size
    Snew = lamS + (torch.rand(1)-.5) * step_size * .01
    u = torch.rand(1)
    if  u < min(1, torch.exp(llh(params, Snew, Lnew) - llh(params, lamS, lamL))):
        return((Snew, Lnew, 1))
    else: return((lamS, lamL, 0))



L0 = prox_L(A, lam_L)
S0 = prox_S((A-L0).flatten(), lam_S)
params= torch.hstack((S0,L0.flatten()))
sample_list=[]
ss = 10
ac = 0
sampler = hamiltorch.Sampler.HMC_NUTS
for i in range(500):
    num_samples = 1
    num_param =  H*W*T*2
    step_size = .1
    num_steps_per_sample = 10
    hamiltorch.set_random_seed(123)
    params = hamiltorch.sample(log_prob_func=llh, params_init=params,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, desired_accept_rate=.5)[0][0]
    (lam_S, lam_L, ac_rate) = mh_SL(params, lam_S, lam_L, ss)
    print(lam_S,lam_L)
    sample_list.append((lam_S,lam_L,params))
    ac += ac_rate
    if i % 20 == 0:
        ss = ss * np.exp(ac / 20 -.6)
        ac = 0
        
for i in range(500):
    num_samples = 1
    num_param =  H*W*T*2
    step_size = .1
    num_steps_per_sample = 10
    hamiltorch.set_random_seed(123)
    params = hamiltorch.sample(log_prob_func=llh, params_init=params,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, desired_accept_rate=.5)[0][0]
    (lam_S, lam_L, ac_rate) = mh_SL(params, lam_S, lam_L, ss)
    print(lam_S,lam_L)
    sample_list.append((lam_S,lam_L,params))

lam_S = sample_list[-1][0]
lam_L = sample_list[-1][1]
S = sample_list[-1][2][0:H*W*T]
L = sample_list[-1][2][H*W*T:2*H*W*T]
L = prox_L(L.reshape(H*W,T), lam_L)
S = prox_S(S, lam_S)
plt.matshow(S0.reshape((H*W,T))[:,5].reshape(H,W))