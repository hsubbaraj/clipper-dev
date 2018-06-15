
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import json
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define model
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

#Delcare model, use SGD Optimizer w/ cross entropy loss
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(2):  
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')

print("Doing testing now")
dataiter = iter(testloader)
image = dataiter.next()
print(image[0].data.numpy().shape)
x= image[0].data.numpy().flatten().tolist()
y = json.dumps({'data': image[0].data.numpy().flatten().tolist()})

i = np.array(x)
input_arr = np.reshape(x, (-1, 3, 32, 32))
input_tensor = torch.FloatTensor(input_arr)
print(input_tensor)
print("result")
result = net(input_tensor)

print(result)
print(result.data.numpy().shape)
print(len(result))


# for i in range(2):
# 	image = dataiter.next()
# 	print(image)
# 	print(net(image[0]))

# print(image[0])
# print(type(image))
# x = image[0].data.numpy()
# print(x)
# print(type(x))
# print(x.tolist())
# y = json.dumps({'data': image.data.numpy().tolist()})
# print()

# for i in range(10):
# 	image = dataiter.next()
# 	print("request " + str(i))
# 	print("result " + str(net(image[0])))







