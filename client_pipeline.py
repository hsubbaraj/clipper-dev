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
from graphviz import Digraph

from lineage import Lineage, Graph



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
	out_lin_1 = predict(clipper_conn.get_query_addr(), "linear1", [input_lin])
	if(out_lin_1.val < 6.5):
		out_lin_2 = predict(clipper_conn.get_query_addr(), "linear2", [out_lin_1])
		out_lin_4 = predict(clipper_conn.get_query_addr(), "linear4", [out_lin_2])
		return out_lin_4
	else:
		out_lin_3 = predict(clipper_conn.get_query_addr(), "linear3", [out_lin_1])
		return out_lin_3

def pipeline_merge(input_lin, clipper_conn):
	out_lin_1 = predict(clipper_conn.get_query_addr(), "linear1", [input_lin])
	out_lin_2 = predict(clipper_conn.get_query_addr(), "linear2", [out_lin_1])

	out_lin_3 = predict(clipper_conn.get_query_addr(), "linear3", [input_lin])
	out_lin_4 = predict(clipper_conn.get_query_addr(), "linear4", [out_lin_3])

	out_lin_5 = predict(clipper_conn.get_query_addr(), "linear5", [out_lin_2, out_lin_4])
	return out_lin_5




#setup Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.connect()




# rand_input = Lineage(np.random.random_sample())
# output = pipeline_merge(rand_input, clipper_conn)
# print(type(output))
# print(output.val)
# print(output.graph)
# print(output.input_node)
# print(output.used)

# dot = Digraph(comment='Merge Example')
# for n in output.graph.nodes:
# 	dot.node(n)
# for e in output.graph.edges:
# 	dot.edge(e[0], e[1])
# dot.render('graphs/pipeline_merge.gv', view=True)

lineages = []
node_counts = {}
for i in range(20):
	rand_input = Lineage(np.random.random_sample())
	output = pipeline(rand_input, clipper_conn)
	lineages.append(output)
	for n in output.graph.nodes:
		if n in node_counts.keys():
			node_counts[n] += 1
		else:
			node_counts[n] = 1
	# total_counts = np.append(total_counts, counts)
	# print(output.graph.adj_list)


	# create a dict of counts instead of list

combined = Graph.merge_graphs([l for l in lineages])
print(node_counts)

dot = Digraph(comment='Merge Example')
for n in combined.nodes:
	dot.node(n + " count: " + str(node_counts[n]/20))
for e in combined.edges:
	dot.edge(e[0], e[1])
dot.render('graphs/pipeline_if.gv', view=True)







