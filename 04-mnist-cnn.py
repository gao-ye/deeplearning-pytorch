# -*- coding: utf-8 -*-
#pytorch.__version__ = 0.4.0
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import time
#begin time
start = time.clock()

#Hyper parameters
num_epochs = 50
batch_size = 1000
learning_rate = 0.001

#load data
train_dataset = dsets.MNIST(root='mnist-data',
							train =True,
							transform = transforms.ToTensor(),
							download= True)
							
test_dataset = dsets.MNIST(root='mnist-data',
							train = False,
							transform = transforms.ToTensor())
# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
							
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
										batch_size=batch_size, 
										shuffle=False)

										
'''
torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递
的顺序添加到模块中。
使用torch.nn.Sequential会自动加入激励函数

'''


#model 
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size = 5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
            nn.MaxPool2d(2))
	
		self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
		
		self.fc = nn.Linear(7*7*32, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

cnn = CNN()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), learning_rate)

#traing the  model
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)
	
		
		#forward + backward + optimizer
		optimizer.zero_grad()
		outputs = cnn(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
		if (i+1) % 10 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.8f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
		
	#Test the model	
	if ((epoch+1) %5 ==0):
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = Variable(images)
			labels = Variable(labels)
			outputs = cnn(images)
			_, pred = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (pred == labels).sum()
	
		print('Epoch is %d   Accuracy  test images: %d %%' %( epoch+1,(100 * correct / total)))	

#end time
end = time.clock()
second = end-start
minute = int(second /60)
second = int(second - minute*60)
print ("time is  {0} minute {1} second ".format(minute, second))

		