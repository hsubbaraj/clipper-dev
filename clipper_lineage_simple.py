#imports
from __future__ import print_function
from clipper_admin import ClipperConnection, KubernetesContainerManager, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model
from clipper_admin.deployers.python import deploy_python_closure
from clipper_admin.deployers import python as python_deployer
import json
import requests
from datetime import datetime
import time
import numpy as np
import signal
import sys

from lineage import Lineage, Client



def lin_model_1(inputs):
	return [2.7*x - 5.3 for x in inputs]

def lin_model_2(inputs):
	return [5.6*x - 3.2 for x in inputs]




clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
# clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.start_clipper()

#Deploy lin_model_1
clipper_conn.register_application(name="lineage1", input_type="doubles", default_output="-1.0", slo_micros=100000)

deploy_python_closure(
	clipper_conn,
    name="lin-model-1",
    version=1,
    input_type="doubles",
    func=lin_model_1,
    registry="hsubbaraj"
    )

clipper_conn.link_model_to_app(app_name="lineage1", model_name="lin-model-1")
print("deployed model 1")

#Deploy lin_model_2
clipper_conn.register_application(name="lineage2", input_type="doubles", default_output="-1.0", slo_micros=100000)

deploy_python_closure(
	clipper_conn,
    name="lin-model-2",
    version=1,
    input_type="doubles",
    func=lin_model_2,
    registry="hsubbaraj"
    )

clipper_conn.link_model_to_app(app_name="lineage2", model_name="lin-model-2")
print("deployed model 2")

