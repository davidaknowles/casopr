import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, init_to_value
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer import Predictive
from pyro.distributions import constraints
from pyro.distributions.util import eye_like

from torch.distributions.transforms import AffineTransform, SigmoidTransform

from scipy.stats import norm
import scipy.stats

import numpy as np
import pandas as pd
from dataclasses import dataclass

import scipy as sp


def model(): 
    x = pyro.sample("x", dist.Normal(1.,1.))
    
def guide():
    
    logit_x_loc = pyro.param("logit_x_loc", torch.tensor(0.))
    logit_x_scale = pyro.param("logit_x_scale", torch.tensor(1.), constraint = constraints.positive)
    
    base_dist = dist.Normal(logit_x_loc, logit_x_scale)
    x = pyro.sample("x", dist.TransformedDistribution(base_dist, SigmoidTransform()))
    
guide = AutoDiagonalNormal(model)

adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO() ) 

import matplotlib.pyplot as plt

pyro.clear_param_store()
losses = []
for j in range(1000):
    # calculate the loss and take a gradient step
    loss = svi.step() 
    losses.append(loss)

plt.plot(losses)

pyro.param('AutoDiagonalNormal.loc')
pyro.param('AutoDiagonalNormal.scale')