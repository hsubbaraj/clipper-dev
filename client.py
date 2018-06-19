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


#Loading data

def predict(addr, x, batch=False):
	url = "http://%s/pytorch-example-2/predict" % addr
	if batch:
		req_json = json.dumps({'input_batch': [x]})
	else:
		print('going to non batch')
		print(x[0])
		print(type(x[0]))
		print(type(x[0]))
		req_json = json.dumps({'input': x})
	headers = {'Content-type': 'application/json'}
	start = datetime.now()
	# print('sending request')
	r = requests.post(url, headers=headers, data=req_json)
	# print('after request')
	end = datetime.now()
	latency = (end - start).total_seconds() * 1000.0
	print(r.json())
	print("%f ms" % (latency))

def signal_handler(signal, frame):
	print("Stopping Clipper...")
	clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
	# clipper_conn = ClipperConnection(DockerContainerManager())
	clipper_conn.stop_all()
	sys.exit(0)


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#setup Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.connect()

batch_size = 2

dataiter = iter(testloader)
# image = dataiter.next()
# print(image)
# print(image[0].data.numpy().tolist())
# print(type(image[0].data.numpy().tolist()))
for i in range(10):
	print("request " + str(i))
	if batch_size > 1:
		input_list = []
		for j in range(batch_size):
			image = dataiter.next()
			input_list += image[0].data.numpy().flatten().tolist()
		# print("input list")
		# print(type(input_list))
		# print(len(input_list))
		predict(
			clipper_conn.get_query_addr(),
			input_list,
			batch=True)
	else:
		image = dataiter.next()
		predict(clipper_conn.get_query_addr(), image[0].data.numpy().flatten().tolist())
	time.sleep(0.2)
print("finished running clipper")