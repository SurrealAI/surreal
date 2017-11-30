import zmq
import time
import pprint

def result_collector():
    context = zmq.Context()
    results_receiver = context.socket(zmq.PULL)
    results_receiver.bind("tcp://127.0.0.1:5557")
    collecter_data = {}
    for x in range(1000):
        result = results_receiver.recv_json()
        pprint.pprint(collecter_data)

result_collector()