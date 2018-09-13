import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

input_size = 1
output_size = 1
iter_size = 500
learning_rate = 1e-3


x_train = np.random.rand(20, 1)* 10
x_train = np.array(x_train, dtype=np.float32)
y_train = x_train *3 

class LinearRegression(nn.Module):
	def __init__(self, input_size, output_size):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(input_size, output_size)
		
	def forward(self, x):
		out = self.linear(x)
		return out
		
model = LinearRegression(input_size, output_size)

##Loss and Optimezer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#train the model
for epoch in range(iter_size):
	inputs = Variable(torch.from_numpy(x_train))
	targets = Variable(torch.from_numpy(y_train))
	
	optimizer.zero_grad()
	outputs = model(inputs)
	loss = criterion(outputs, targets)
	loss.backward()
	optimizer.step()
	
	if (epoch+1) % 5 == 0:
		print("Epoch {0}/{1}, Loss is {2}".format(epoch+1, iter_size,loss.data[0]))
	
pred = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, pred)
plt.legend()
plt.show()

# Save the Model
torch.save(model.state_dict(), 'model.pkl')
	
	
	
	
	
	
	
	
	
	
	
	