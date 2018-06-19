#imports
from __future__ import print_function
from clipper_admin import ClipperConnection, KubernetesContainerManager, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model
from clipper_admin.deployers import python as python_deployer
import json
import requests
from datetime import datetime
import time
import numpy as np
import signal
import sys



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim




def predict(addr, x, batch=False):
	url = "http://%s/pytorch-example/predict" % addr
	if batch:
		req_json = json.dumps({'input_batch': x})
	else:
		print('going to non batch')
		print(x)
		print(type(x))
		print(type(x[0]))
		req_json = json.dumps({'input': x})
	headers = {'Content-type': 'application/json'}
	start = datetime.now()
	print('sending request')
	r = requests.post(url, headers=headers, data=req_json)
	print('after request')
	end = datetime.now()
	latency = (end - start).total_seconds() * 1000.0
	print("'%s', %f ms" % (r.text, latency))


def feature_sum(xs):
	return [str(sum(x)) for x in xs]


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
	print("Stopping Clipper...")
	clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
	# clipper_conn = ClipperConnection(DockerContainerManager())
	clipper_conn.stop_all()
	sys.exit(0)




if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	# Loading data:

	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	#Define model
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



	# try:
	clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
	# clipper_conn = ClipperConnection(DockerContainerManager())
	clipper_conn.start_clipper()
	clipper_conn.register_application(name="pytorch-example-2", input_type="doubles", default_output="-1.0", slo_micros=100000)
	# model = nn.Linear(1, 1)

# Define a shift function to normalize prediction inputs
	def pred(model, inputs):
		preds = []
		for i in inputs:
			transform_input = torch.FloatTensor(np.reshape(i, (-1, 3, 32, 32)))
			print(type(transform_input))
			print(transform_input)
			result = model(transform_input)
			preds.append(result.data.numpy())
		return preds

	deploy_pytorch_model(
		clipper_conn,
	    name="pytorch-nn",
	    version=1,
	    input_type="doubles",
	    func=pred,
	    pytorch_model=net,
	    registry="hsubbaraj"
	    )

	clipper_conn.link_model_to_app(app_name="pytorch-example-2", model_name="pytorch-nn")
	time.sleep(2)
	print("deployed model")



