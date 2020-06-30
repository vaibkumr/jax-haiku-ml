import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
from jax.experimental import optix


def conv_net(x):
  x = np.asarray(x[0])
  _net = hk.Sequential([
        hk.Conv2D(8, kernel_shape=2, stride=1),
        jax.nn.relu,
        hk.Conv2D(4, kernel_shape=5, stride=1),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(100), 
        jax.nn.relu,
        hk.Linear(10),
    ])
  return _net(x)

def dnn(x):
  x = np.asarray(x[0])
  _net = hk.Sequential([
        hk.Flatten(),
        hk.Linear(100), 
        jax.nn.relu,
        hk.Linear(500),
        jax.nn.relu,
        hk.Linear(100),
        jax.nn.relu,
        hk.Linear(10),        
    ])
  return _net(x)