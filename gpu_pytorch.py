import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#running pytorch on gpu (it will perform more fast calculation since gpu does billion of caclution in seconds while cpu does in millions)
#check the gpu access through pytorch
print(torch.mps.is_available())#for mac use mps and for windows use cuda if u have nvdia gpu

#setup device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"
print(device)

#putting tensors and models on the gpu
tensor= torch.tensor([1,2,3])
x=tensor.to("mps")
print(x)#it will show mps:0 here zero is index of gpu many people use multiple gpu so they might have 2,3... so on in place of zero
print(x.device)

#move back NOTE- numpy wont work to gpu
x=tensor.to("cpu") #or tensor.cpu
print(x)
print(x.device)

#numpy and tensor gpu to cpu to gpu 
numpy= x.cpu().numpy()
print(numpy)