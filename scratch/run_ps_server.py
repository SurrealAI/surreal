from surreal.distributed import *
import torch


server = ParameterServer(
    publish_host='localhost',
    publish_port=8001,
    agent_port=8002,
)
server.run_loop()

