import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import time
#begin time
start = time.clock()



#Hyper parameters
input_size = 784
hidden_size = 500
num_class = 10
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


#model 
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_class):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_class)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

net = Net(input_size, hidden_size, num_class)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images.view(-1, 28*28))
		labels = Variable(labels)
		
		#forward + backward + optimizer
		optimizer.zero_grad()
		outputs = net(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
			
	if epoch+1 %5 ==0:
		#Test the model
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = Variable(images.view(-1, 28*28))
			outputs = net(images)
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

		