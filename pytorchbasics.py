import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#creating a tensor 
#scaler 
scaler = torch.tensor(7)
#print(scaler.type())#prints the type of the tensor
#print(scaler)
f=scaler.ndim #number of dimensions
print(f)#output is 0 because its a scaler
print(scaler.item())#gives them item inside the tensor

#vector
vector= torch.tensor([1,2,3])#to know the dimension of the vector count the number of square brackets
print(vector.ndim)#output is 1  
print(vector.shape)#gives the shape of the vector  number of elemnts inside the vector

#matrix
matrix= torch.tensor([[1,2,3],[4,5,6]])#to know the dimension of the matrix count the number of square brackets
print(matrix.ndim)#output is 2
print(matrix[1])#index 1 vector output
print(matrix.shape)#gives the shape of the matrix number of rows and columns for here 2 rows 3 columns

#tensor
tensor=torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]])#to know the dimension of the tensor count the number of square brackets
print(tensor.ndim)#output is 4 since there are 4 square brackets
print(tensor.shape)#gives the shpe of the tensor first gives 1 seocnd gives 2 because 2 matrices then 3 because 3 row in each matrices and 4 number of elemnts in a vector
print(tensor[0])

#random tensor
#its important because when computer starts to learn it starts with random weights and then it updates the weights based on the loss
random_tensor= torch.rand(3,4)#gives a random tensor with 3 rows and 4 columns (to get the number of elements in the tensor multipley all the numbers in the brackets 3*4=12)
print(random_tensor)
print(random_tensor.ndim)#output is 2 because its a matrix or more simpley it has 2 bar brackets at start and end
print(random_tensor.shape)#gives the shape of the random tensor 3 rows and 4 columns

#image tensor
#images are represented as 3D tensors (height, width, color channels[R,G,B])
image_tensor= torch.rand(224,224,3)#gives a random tensor 
print(image_tensor.ndim)#output is 3 because it has 3 bar brackets at start and end or 3 dimensions h w and color channels
print(image_tensor.shape)#gives the shape of the image tensor 224 height 224 width and 3 color channels

# create tensor of 0 and 1
zero= torch.zeros(2,3)
print(zero)
one=torch.ones(2,3)
print(one.dtype)#prints the type of the tensor

#creating range of tensors and tesnors like
range= torch.arange(2,10,2)#upper limit lower limit then step
print(range)

#creating a tensor like another tensor
like= torch.zeros_like(range)#gives a tensor of zeros with the same shape as the range tensor
print(like)

#tensor datatypes MUST FOCUS ON THIS
float32= torch.tensor([1.0,2.0,3.0]
                    , dtype=None #what datatype is the tensor
                    ,device=None# by default its cpu u can change to cude for gpu, used when one is on gpu and cpu they wont work together so we use this
                    ,requires_grad=False#to track the gradients
                      )#float tensor default datatype is also set to set to float 32 
print(float32.dtype)#prints the type of the tensor

#float 16 tensor
float16= float32.type(torch.float16)#converts int float 16
print(float16.dtype)#prints the type of the tensor
print(float16*float32)#when multiplying two tensors with different datatypes the output is the higher datatype in this case float 32
int32= float32.type(torch.int32)#converts float 32 to int 32
p= int32*float32#when multiplying two tensors with different datatypes the output is the higher datatype in this case float 32
print(p.dtype, int32)#prints the type of the tensor

#getting information from tensors
#to get data type use tensor.dtype
#to get shape use tensor.shape
#to get right device use tensor.device
some_tensor= torch.rand(5,6)
print(f"Data type: {some_tensor.dtype}")
print(f"Shape: {some_tensor.shape}")
print(f"Device: {some_tensor.device}")

