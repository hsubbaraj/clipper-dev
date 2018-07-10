class Graph():
	def __init__(self, nodes=set(), edges=set(), adj_list={}):
		self.nodes = nodes
		self.edges = edges
		self.adj_list = adj_list

	def __repr__(self):
		l = [self.nodes, self.edges, self.adj_list]
		return str(l)

	def add_node(self, new_node):
		self.adj_list[new_node] = set([])
		self.nodes.add(new_node)

	def add_edge(self, source, target):
		if source not in self.nodes:
			self.add_node(source)
		if target not in self.nodes:
			self.add_node(target)

		self.adj_list[source].add(target)
		self.edges.add((source, target))

	@classmethod
	def merge_graphs(self, lin_list):
		nodes_combined = set()
		edges_combined = set()
		adj_list_combined = {}
		for lin in lin_list:
			nodes_combined = nodes_combined.union(lin.graph.nodes)
			edges_combined = edges_combined.union(lin.graph.edges)
		for node in nodes_combined:
			adj_list_combined[node] = set([v[1] for i, v in enumerate(edges_combined) if v[0] == node])

		return Graph(nodes=nodes_combined, edges=edges_combined, adj_list=adj_list_combined)


class Lineage():
	def __init__(self, val, graph = Graph(), input_node="source"):
		self.val = val
		self.used = False
		self.graph = graph
		self.input_node = input_node
		self.graph.add_node(input_node)

	def __repr__(self):
		l = [self.val, self.graph, self.input_node, self.used]
		return str(l)

	@classmethod
	def add_node(self, parents, new_node_name, new_val):
		graph = Graph.merge_graphs(parents)
		for p in parents:
			graph.add_edge(p.input_node, new_node_name)
		return Lineage(new_val, graph=graph, input_node=new_node_name)

	#creates lineage obj with same vals but NOT used
	def fork(self):
		return Lineage(self.val, self.path.copy())

	def make_prediction(self):
		#print("updating used")
		if self.used == True:
			raise Exception("Already made prediction from this lineage object")		
		self.used = True

	@classmethod
	def join(forks, val):
		#Create graph struct in list
		new_path = [f.path for f in forks]
		return Lineage(val, new_path)

