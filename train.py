import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
from jax.experimental import optix
from model import conv_net
from dataloader import get_loaders
import matplotlib.pyplot as plt
from tqdm import tqdm

KEY = jax.random.PRNGKey(777)

def loss(params, data):
  logits = model.apply(params, data)
  probs = jax.nn.softmax(logits)
  labels = jax.nn.one_hot(jnp.array(data[1].numpy()), 10)
  n = labels.shape[0]
  ce_loss = -jnp.sum(labels*jnp.log(probs))/n
  return ce_loss

def update(params, opt_state, data):
  grads = jax.grad(loss)(params, data)
  updates, opt_state = opt.update(grads, opt_state)
  new_params = optix.apply_updates(params, updates)
  return new_params, opt_state

def accuracy(params, data):
  predictions = model.apply(params, data)
  return jnp.mean(jnp.argmax(predictions, axis=-1) == jnp.array(data[1].numpy()))    


if __name__ == "__main__":
    bs = 64
    epochs = 2
    lr = 3e-4
    dir = "."
    train_loader, val_loader = get_loaders(bs, dir)
    train_iter, val_iter = iter(train_loader), iter(val_loader)

    model = hk.transform(conv_net)
    opt = optix.adam(lr)
    params = model.init(KEY, next(train_iter))
    opt_state = opt.init(params)

    train_accs, val_accs = [], []
    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(tqdm(train_loader)):
            if i % 1 == 0:
                train_accuracy = accuracy(params, data) #This is just accuracy of the current batch
                try:
                    val_accuracy = accuracy(params, next(val_iter)) #This is just accuracy of one batch item
                except:
                    val_iter = iter(val_loader) #A very ugly hack
                    val_accuracy = accuracy(params, next(val_iter)) #This is just accuracy of one batch item
                train_accs.append(train_accuracy)    
                val_accs.append(val_accuracy)    
                # print(f"Train acc: {train_accuracy} | Val acc: {val_accuracy}")
            params, opt_state = update(params, opt_state, data)
    
    plt.plot(train_accs)        
    plt.plot(val_accs)     
    plt.legend(["Train Accuracy", "Valid Accuracy"])   
    plt.show()
    