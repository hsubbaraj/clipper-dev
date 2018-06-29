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
import seaborn as sns

from lineage import Lineage, Client



#Loading data

def predict(addr, model_name, input_lin, batch=False):
	url = "http://%s/%s/predict" % (addr, model_name) 
	if batch:
		req_json = json.dumps({'input_batch': [x]})
	else:
		req_json = json.dumps({'input': [input_lin.val]})
	headers = {'Content-type': 'application/json'}
	r = requests.post(url, headers=headers, data=req_json)
	# print(json.loads(r.text)["output"][0])
	input_lin.make_prediction()
	return Lineage.add_node(input_lin, model_name, json.loads(r.text)["output"][0])


#setup Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.connect()

batch_size = 1
# batches = np.array([])
# times = np.array([])
for i in range(20):
	# batch_size = np.random.randint(5, high=50)
	print("request " + str(i))
	if batch_size > 1:
		input_list = [np.random.random_sample() for i in range(batch_size)]
		out_json = predict_1(clipper_conn.get_query_addr(), input_list, batch=True)
		print(json.loads(output_1)["output"][0])
	else:
		input_lin = Lineage(np.random.random_sample())
		out_lin_1 = predict(clipper_conn.get_query_addr(), "lineage1", input_lin)
		print(out_lin_1)
		out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage2", out_lin_1)
		print(out_lin_2)
print("finished running clipper")







