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
random_tensor= torch.rand(5,6)
print(f"Data type: {random_tensor.dtype}")
print(f"Shape: {random_tensor.shape}")
print(f"Device: {random_tensor.device}")

#manipulating tensors or tensor operations

#additon
tensor1= torch.tensor([1,2,3])               
print(tensor1+10)#adds 10 to each element in the tensor

#multiplication
print(tensor1*10)#multiplies 10 to each element in the tensor

#subtraction
print(tensor1-10)#subtracts 10 from each element in the tensor

#pytorch inbuild functions
print(torch.add(tensor1,10))#adds 10 to each element in the tensor
print(torch.mul(tensor1,10))#multiplies 10 to each element in
print(torch.sub(tensor1,10))#subtracts 10 from each element in the tensor

#matrix multiplication 
tensor2= torch.tensor([4,5,6])
print(f"element wise multiplication: {tensor1*tensor2}")#element wise multiplication
print(f"matrix multiplication: {torch.matmul(tensor1, tensor2)}")#matrix multiplication

#rules in matrix multiplication
#1 the inner dimensions must match
#example 1: (2,3)@ (2,3)not possible
#example 2: (2,3)@ (3,4) possible  any matrix with same columns of 1st and same row of 2nd is possible
#2 the result matrix has the shape of outer dimentions
#example (2,3)@ (3,4)gives (2,4) shape matirx

torch1= torch.rand(2,3)
#print(torch.matmul(torch1,torch1). will show error)
torch2=torch.rand(3,4)
print(torch.matmul(torch1,torch2).shape)#this will work because the inner dimensions match

#dealing with shape errors
tensor_a= torch.tensor([[1,2],
                         [3,4],
                         [5,6]])
tensor_b= torch.tensor([[7,8],
                         [9,10],
                         [11,12]])
#m=torch.mm(tensor_a,tensor_b) #this will show error because the inner dimensions do not match
#print(m)

#to fix this issue we can manupulate the shape of the tesnor, aka we use transpose funtion
print(tensor_b.T)#now we can do matrix multiplication
m=torch.mm(tensor_a,tensor_b.T)
print(m
      , m.shape)#no more errors

#finding min max mean and sum of tensors
tensor_c= torch.arange(1,10).reshape(3,3)
print(tensor_c)
print(f"Min: {torch.min(tensor_c)}")#gives the minimum value in the tensor
print(f"Max: {torch.max(tensor_c)}")#gives the maximum value
print(f"type: {tensor_c.type()}")#gives the type of the tensor
mean=tensor_c.type(torch.float32)#converting to float32
print(f"Mean: {torch.mean(mean)}")#gives the mean of the tensor, we need to have to tensor in float or complex
print(f"Sum: {torch.sum(tensor_c)}")#gives the sum of all the elements in the tensor

#finding the index of min and max values
print(f"Index of min: {torch.argmin(tensor_c)}")#"argmin" gives the index of the minimum value
print(f"Index of max: {torch.argmax(tensor_c)}")#"argmax" gives the index of the maximum value

#squeezinng, reshaping, stacking, unsqueezing, view
# reshaping - helps us change shape to a defined shpae
# view- return view of the original tensor with the same data but different shape 
# stacking- combine multiple tensores
# squeeze- remove all "1" dimesnions
# unsqueeze- add a "1" dimension
# permute- return the view of an input with dimension swapped in a certain way 
x= torch.arange(1,11)
#print(x.shape)

#reshapeing
y=x.reshape(2,5)#reshape the tensor to 2 rows and 5 columns
print(f"reshped: {y}")

#view
z=x.view(2,5)
z[0,2]=100# this will change value of z but also x since z is the view of x
print(f"view: {z}")
print(f"origial: {x}")

#stacking
x_stack=torch.stack([y,y,y,y], dim=1)#stacking 4 copies of y along the second dimension
print(x_stack)
x_stack2= torch.vstack([y,y])#stacking 2 copies of y along the first dimension
print(f"vstack.  {x_stack2}")
x_stack3= torch.hstack([y,y])#stacking 2 copies of y along the third dimension
print(f"hstack.  {x_stack3}")

#squeeze
tensor= torch.arange(1,31).reshape(1,5,3,2,1)# 1 tells batch, 5 tells the the number of grps, 2x3x1 is like 1 is for number of colums 2 for rows but 3 is the total number 2x1 matrix that can be kept in that for here its 3
print(tensor.shape)
print(tensor)
tensor_squeezed= tensor.squeeze()#removes the "1" dimesnions"
print(tensor_squeezed)
print(tensor_squeezed.shape)#the shape becomes 5,3,2 from 1,5,3,2,1 since it removed 1 dimension

#unsqueeze
tensor_unsqueezed= tensor_squeezed.unsqueeze(dim=1)#adds a "1" dimension at the end
print(tensor_unsqueezed)
print(tensor_unsqueezed.shape)#gives output 9,1 since we used dim=1 we can use dim=2 also for certain shapes

#permute in simple words its rearrange
tensor_x= torch.arange(1,21).reshape(2,5,2)#2 batch of matri 5x2
tensor_x[1,1,0]=69# 1 is batch index, 1 is row 0 is column
print(tensor_x)
tensor_y= tensor_x.permute(2,0,1)#outputs 2 batches of matrix 2x5
print(tensor_y)
#NOTE all of them will share same memory diffrent shape

#indexing - selection of data from tensors
x= torch.arange(1,21).reshape(2,2,5)
print(f"old: {x}")
x[1,1,1]=69 #changed the number at index 1,1 in dimension 0
print(x[:,1,1])#putting : gives the number at index 1,1 in all the dimensions
print(x[:,:,1])#gives all the numbers at column 1 in all the rows and dimension
print(x[:,1,:])
print(f"new: {x}")
