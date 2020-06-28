import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


def l2(pred, target):
    
    def l2_scalar(n, pred, target):
        residual = pred-target
        return residual**2/n #for numerical limits, running average is better.
    
    loss_l2 = jax.vmap(partial(l2_scalar, len(pred)))   
    return jnp.sqrt(loss_l2(pred, target).sum())


class LR():
    def __init__(self, lr):
        self.lr = lr
        self.w = np.random.rand(1)
        self.c = np.random.rand(1)

    def get_loss(self, x, y, w, c):
        out =  w*x + c
        loss = l2(out, y)
        return loss

    def train(self, X, y, epochs, verbose=False):
        if len(X.shape) > 1:
            self.w = np.random.rand(X.shape[1])
        gl = jax.jit(self.get_loss)
        for i in range(epochs):
            loss = gl(X, y, self.w, self.c)  
            grad_w = jax.grad(gl, argnums=2)(X, y, self.w, self.c)
            grad_c = jax.grad(gl, argnums=3)(X, y, self.w, self.c)
            self.w -= self.lr * grad_w
            self.c -= self.lr * grad_c
            if verbose:
                print(f"Epoch: {i} | Loss: {loss}")
        return self    

    def predict(self, X):
         return self.w * X + self.c           
     
    def __repr__(self):
        return f"w: {self.w} | c: {self.c}"           

if __name__ == "__main__":
    lr = 1e-3
    epochs = 1000
    N_samples = 100
    N_d = 3
    X = 10 * np.random.rand(N_samples, N_d)
    y = [2, 3, -1]*X + 2
    model = LR(lr)
    model.train(X, y, epochs, True)
    print(model)

