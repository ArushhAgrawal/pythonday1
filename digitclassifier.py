import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_data= datasets.MNIST(root= 'numbers', train= True, download= True, transform= transforms.ToTensor())#dowloads the training data
test_data= datasets.MNIST(root= 'numbers', train= False, download= True, transform= transforms.ToTensor())#downloads the testing data

train_loader= DataLoader(train_data, batch_size= 64, shuffle= True)#creates a downloader, batch size is for telling how many to go with once using all 70k image will crash ur pc, shuffle so computer dont memorises
test_loader= DataLoader(test_data, batch_size= 64, shuffle= False)# same thing but did not shuffle bc its not nessesary for testing data its for testing computer wont learn anything
#trains the model
class Digit_classifier(nn.Module): #class makes a blue print, nn.Module is going to handle maths
    def __init__(self):
        super().__init__()#super is like it setup adn run ur stuff
        self.layer1= nn.Linear(784,128)#layers through them data will flow
        self.layer2=nn.Linear(128,64)
        self.layer3=nn.Linear(64,10)
        self.relu=nn.ReLU()#its gonna convert all the number to 0 to1, all negeative are 0
        self.softmax=nn.Softmax(dim=1)#it gives probability say 30% chance its 1 like this
    def forward(self,x):#forward is the function that runs the data through the layers
        x= x.view(-1,784)#flattens the image
        x=self.relu(self.layer1(x))#x is now after layer 1 we wrote x inside of layer one bc it will wrap everyting again in x form
        x=self.relu(self.layer2(x))#same thing for layer 2
        x=self.layer3(x)#same thing for layer 3 but with no relu
        return x
#training loop
model= Digit_classifier()#creates the model
criterion= nn.CrossEntropyLoss()#loss function, its gonna tell us how wrong the model is
optimizer= optim.Adam(model.parameters(), lr= 0.001)#optimizer is gonna update the weights of the model, lr learning rate is how much to update the weight too big too small is bad

epochs=5# number of times we want it to go thrugh the data
for epoch in range(epochs):
    total_loss=0
    for images, labels in train_loader:# loads 64 images from trainer loader
        optimizer.zero_grad()#resets the gradient to zero else it will pile up energy and time will lose
        output=model(images)#runs image and gets the pridiction
        loss= criterion(output, labels)# comapres the output to the labels and gives the loss
        loss.backward()#punishes the weight caused the loss
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch+1}/5, Loss: {avg_loss:.4f}")#prints the epoch and loss
#testing loop
correct = 0
total = 0

with torch.no_grad():#stops comaparion and updationg weights
    for images, labels in test_loader: #loads 64 images and there labels
        output = model(images)#runs the image and gets the pridiction
        predicted = torch.argmax(output, dim=1)#gets the pridcition with the highest value
        total += labels.size(0)#total number of images
        correct += (predicted == labels).sum().item()#comapres prediction to labels and print sum of how many times it printed correct nd item convert it into normal python from tensor

print(f"Accuracy: {100 * correct / total:.2f}%")#print acurracy %