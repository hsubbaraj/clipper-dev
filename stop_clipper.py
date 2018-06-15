from clipper_admin import ClipperConnection, KubernetesContainerManager, DockerContainerManager

clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True))
clipper_conn.stop_all()