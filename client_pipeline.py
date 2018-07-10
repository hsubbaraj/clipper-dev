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

from lineage import Lineage



#Loading data

def predict(addr, model_name, input_lins, batch=False):
	url = "http://%s/%s/predict" % (addr, model_name)
	if batch:
		req_json = json.dumps({'input_batch': [[i.val for i in input_lins]]})
	else:
		req_json = json.dumps({'input': [i.val for i in input_lins]})
	headers = {'Content-type': 'application/json'}
	# print(req_json)
	r = requests.post(url, headers=headers, data=req_json)
	print(json.loads(r.text))
	# for i in input_lins:
	# 	i.make_prediction()
	return Lineage.add_node(input_lins, model_name, json.loads(r.text)["output"][0])


def pipeline(input_lin, clipper_conn):
	out_lin_1 = predict(clipper_conn.get_query_addr(), "lineage1", [input_lin])
	# print(out_lin_1)
	if(out_lin_1.val < 6.65):
		out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage2", [out_lin_1])
	else:
		out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage3", [out_lin_1])
	# print(out_lin_2)
	return out_lin_2

def pipeline_merge(input_lin, clipper_conn):
	out_lin_1 = predict(clipper_conn.get_query_addr(), "lineage1", [input_lin])
	out_lin_2 = predict(clipper_conn.get_query_addr(), "lineage2", [out_lin_1])

	out_lin_3 = predict(clipper_conn.get_query_addr(), "lineage3", [input_lin])
	out_lin_4 = predict(clipper_conn.get_query_addr(), "lineage4", [out_lin_3])

	out_lin_5 = predict(clipper_conn.get_query_addr(), "lineage5", [out_lin_2, out_lin_4])
	return out_lin_5




#setup Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.connect()


rand_input = Lineage(np.random.random_sample())
output = pipeline_merge(rand_input, clipper_conn)
print(type(output))
print(output.val)
print(output.graph)
print(output.input_node)
print(output.used)

# lineages = []
# for i in range(20):
# 	rand_input = Lineage(np.random.random_sample())
# 	output = pipeline_merge(rand_input, clipper_conn)
# 	lineages.append(output)
# 	# print(output.graph.adj_list)

# nodes = set()
# edges = set()
# for lin in lineages:
# 	nodes = nodes.union(lin.graph.nodes)
# 	edges = edges.union(lin.graph.edges)

# print(nodes)
# print(edges)










