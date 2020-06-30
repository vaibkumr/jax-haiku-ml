# Machine Learning using Jax and Haiku
Writing some ML algorithms in jax and neural nets using haiku for learning purposes.

## About
Jax is an autograd library that works directly over numpy and plain python functions. It can also use XLA (jit) to compile code on the first run for faster computations. 

## Haiku
A haiku is a short japanese poem.
```
Intro to Jax-n-Haiku
Learning is an everlasting phase
pyTorch is betrayed
```
Haiku is also an NN library build over the top of jax. Transitioning from tf to haiku is expected to be easy, but idk tf either. I know pyTorch and I love pyTorch, yet, I am excited to learn haiku. Seems interesting especially because jax can work with native python. 

## Files
- `Jax playground.ipynb`: Playing with jax.  
- `Haiku Playground.ipynb`: Playing with haiku. 
- `linear_regression_jax.py`: Linear regression in pure jax.   
- `dataloader.py`: Dataloader for FashionMNIST (autodownloads using pyTorch).
- `model.py`: Haiku neural net models: a) A simple DNN, b) A simple 4 layers deep ConvNet.
- `train.py`: Train over the fashionMNIST data. 


