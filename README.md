#Clipper-dev

Deployed an image classifier built using pytorch on Clipper

Install: 
	1. Clipper
	2. Pytorch
	3. Seaborn

1. Start Docker and minikube (via minikube start --vm-driver hyperkit)
2. Init Clipper & deploy model: run clipper_dev.py to deploy pytorch neural network, run clipper_randop.py to deploy a model that returns a random vector
3. Open batch_client.ipynb in a jupyter notebook. Change the filename to save the plot to a different file.
4. Run stop_clipper.py to stop Clipper.

Visit www.clipper.ai for more information about Clipper