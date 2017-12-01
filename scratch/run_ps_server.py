from surreal.distributed import *
import torch


server = ParameterServer(
    learner_host='localhost',
    learner_port=8001,
    agent_port=8002,
)
server.run_loop()

