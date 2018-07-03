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
		req_json = json.dumps({'input_batch': [[x.val for x in input_lin]]})
	else:
		req_json = json.dumps({'input': [input_lin.val]})
	headers = {'Content-type': 'application/json'}
	r = requests.post(url, headers=headers, data=req_json)
	# print(json.loads(r.text))
	str_r = json.loads(r.text)["output"].replace("[", "").replace("]", "").split()
	vals = [float(i) for i in str_r]
	new_lineage_objs = [Lineage.add_node(input_lin[i], model_name, vals[i]) for i in range(len(input_lin))]
	return new_lineage_objs
	


#setup Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.connect()

batch_size = 2
# batches = np.array([])
# times = np.array([])
for i in range(20):
	# batch_size = np.random.randint(5, high=50)
	print("request " + str(i))
	if batch_size > 1:
		input_list = [Lineage(np.random.random_sample()) for i in range(batch_size)]
		out_lin_1 = predict(clipper_conn.get_query_addr(), "lineage1", input_list, batch=True)
		print(out_lin_1)
		out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage2", out_lin_1, batch=True)
		print(out_lin_2)
	else:
		input_lin = Lineage(np.random.random_sample())
		out_lin_1 = predict(clipper_conn.get_query_addr(), "lineage1", input_lin)
		print(out_lin_1)
		out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage2", out_lin_1)
		print(out_lin_2)
print("finished running clipper")







