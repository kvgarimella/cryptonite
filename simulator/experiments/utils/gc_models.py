import numpy as np

"""
Garbling and Evaluation of Garbled ReLUs
as a function of:
- the number of threads
- the device

example:
garble_latency('client', 303104, 4):
    returns the latency of garbling 303104 relus
    on a client-like device

Assumptions:
- Server model is an AMD EPYC 7502
- Client model is Intel Atom Z8350 2GB DDR3 32GB
- Garbling and Evaluation time scales linearly
- Linear speedup for the number of threads available

See the subdirectory "garbled_circuits" to see the raw data for collecting GC garbling and evaluation latencies.
"""

def garble_latency(garbler, num_relus, num_threads):
    if garbler == 'server':
        return garble_server(num_relus) / num_threads
    elif garbler == 'client':
        return garble_client(num_relus) / num_threads

def eval_latency(evaluator, num_relus, num_threads):
    if evaluator == 'server':
        return eval_server(num_relus) / num_threads
    elif evaluator == 'client':
        return eval_client(num_relus) / num_threads

num_relus      = np.array([1,10,100])
relu_ev_server = np.array([5.19322887e-05, 5.22861572e-04, 5.44515682e-03])
relu_gb_server = np.array([0.00010212, 0.00102062, 0.01036614])
relu_ev_client = np.array([0.00028193, 0.00335818, 0.034515  ])
relu_gb_client = np.array([0.00056736, 0.00566518, 0.06809248])

# fit a first-order approximation
eval_server = np.poly1d(np.polyfit(num_relus, relu_ev_server, 1))
eval_client = np.poly1d(np.polyfit(num_relus, relu_ev_client, 1))

garble_server = np.poly1d(np.polyfit(num_relus, relu_gb_server, 1))
garble_client = np.poly1d(np.polyfit(num_relus, relu_gb_client, 1))
