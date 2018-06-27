class Lineage():
	def __init__(self, val, path=[]):
		self.val = val
		self.path = path
		self.used = False

	@classmethod
	def add_node(parent, new_node_name, new_val):
		path = parent.path.copy()
		path.append(new_node_name)
		return Lineage(new_val, path)

	#creates lineage obj with same vals but NOT used
	def fork(self):
		return Lineage(self.val, self.path.copy())

	def make_prediction(self):
		if self.used == True:
			raise Exception("Already made prediction from this lineage object")		
		self.used = True

	@classmethod
	def join(forks, val):
		#Create graph struct in list
		new_path = [f.path for f in forks]
		return Lineage(val, new_path)


class Client():
	def __init__(self, addr):
		self.rpc = RPC(addr).connect()

	def predict(model_name, input_obj):
		result = self.rpc.predict(model_name, input_obj.val)
		return Lineage.add_node(input_obj, model_name, result)



def pred_clipper_chain(x, client):
	y = client.predict("lin_model_1", x)
	z = client.predict("lin_model_2", y) 
	return z

def pred_clipper_parallel_forks(x, client):
	y = client.predict("lin_model_1", x)
	z = client.predict("lin_model_2", y) 
	return z



def main():
	addr = 'localhost:8000'
	session = InferlineSession(addr)
	pipeline = session.deploy_pipeline(pred_clipper_chain)
	pipeline.serve()


def main_embedded():
	addr = 'localhost:8000'
	session = InferlineSession(addr)
	pipeline = session.deploy_pipeline(pred_clipper_chain)
	for i in range(1000):
		inp = np.random.random((1, 10))
		print(pipeline.predict(inp))

