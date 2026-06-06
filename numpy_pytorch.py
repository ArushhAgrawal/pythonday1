import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#numpy in pytorch
#to change numpy data to tensor datatype we use - torch.from_numpy(nparray)
#pytorch to numpy-  name.numpy()

#np to tensor
x=np.arange(1,21).reshape(4,5)
print(x)
print(x.dtype)
y= torch.from_numpy(x)#converting of np to pytorch value
print(y)
print(y.dtype)
#NOTE - the dtype value of numpy are set to int 64 and float64 by default 

#tensor to np
tensor= torch.arange(1,21).reshape(4,5)
print(tensor)
print(tensor.dtype)
np_array= tensor.numpy()#converting from tensor to np datatype
print(np_array)
print(np_array.dtype)

#reproducibility- trying to take out random stuff out of random stuff
#to reduce the random ness is neural netwrok and pytorch comes with the concept of random seed
#random seed controls the amount of salt added to food 
random_Seed= 1
torch.manual_seed(random_Seed)#manual seed works only for one block of code each randomness need new one
random_a= torch.rand(4,5)
torch.manual_seed(1)
random_b= torch.rand(4,5)
print(random_a==random_b)

