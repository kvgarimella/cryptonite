import os
import json
import simpy 
import random 
import argparse
import itertools
import numpy as np
from tqdm import tqdm


from utils import gc_models
from utils import he_models

from utils import cifar10_client_garbler_utils_r18
from utils import cifar10_client_garbler_utils_r32
from utils import cifar10_client_garbler_utils_vgg16
from utils import tiny_client_garbler_utils_r18
from utils import tiny_client_garbler_utils_r32
from utils import tiny_client_garbler_utils_vgg16

mapping = {"cifar10":            {"resnet18"  : cifar10_client_garbler_utils_r18,
                                  "resnet32"  : cifar10_client_garbler_utils_r32,
                                  "vgg16"     : cifar10_client_garbler_utils_vgg16},
           "tinyimagenet" :      {"resnet18"  : tiny_client_garbler_utils_r18,
                                  "resnet32"  : tiny_client_garbler_utils_r32,
                                  "vgg16"     : tiny_client_garbler_utils_vgg16}}

parser = argparse.ArgumentParser(description='End-to-End Private Inference Latency')
parser.add_argument("--sim-time",              type=int,   default=1440,       help="simulation time (minutes)")
parser.add_argument("--client-storage",        type=int,   default=8,          help="Client Storage capacity (GB)")
parser.add_argument("--server-storage",        type=int,   default=10000,      help="Server Storage Capacity (GB)")
parser.add_argument("--client-nonlinear",      type=float, default=3.59,       help="amount of nonlinear storage required by client (KB/ReLU)")
parser.add_argument("--server-nonlinear",      type=float, default=18.248,     help="amount of nonlinear storage required by client (KB/ReLU)")
parser.add_argument("--upload-bandwidth",      type=float, default=100.,       help="Upload Bandwidth in Mbps")
parser.add_argument("--download-bandwidth",    type=float, default=500.,       help="Download Bandwidth in Mbps")
parser.add_argument("--number-of-runs",        type=int,   default=1,          help="number of times to run the simulator")
parser.add_argument("--network",               type=str,   default="resnet18", help="resnet18, resnet32, vgg16")
parser.add_argument("--dataset",               type=str,   default="cifar10",  help="cifar10, tinyimagenet")
parser.add_argument("--num-threads-client",    type=int,   default=1,          help="Number of threads client-side")
parser.add_argument("--num-threads-server-he", type=int,   default=1,          help="Number of threads - Server HE" )
parser.add_argument("--num-threads-server-gc", type=int,   default=1,          help="Number of threads - Server GC")
parser.add_argument("--start",                 type=float, default=0.001,      help='requests per second')
parser.add_argument("--end",                   type=float, default=0.01,       help='requests per second')
parser.add_argument("--step" ,                 type=int,   default=10,         help='discretize between start and end')
parser.add_argument("--name",                  type=str,   default='tmp',      help='experiment name')
args = parser.parse_args()

upload_bandwidth   = args.upload_bandwidth * (1/8) * 1e6   # units: Mbps -> MBps -> (bytes / second)
download_bandwidth = args.download_bandwidth * (1/8) * 1e6 # units: Mbps -> MBps -> (bytes / second)
SIM_TIME           = args.sim_time * 60                        # units: min -> sec  (simulation time in seconds)
CLIENT_STORAGE     = args.client_storage * 1e6                 # units: GB -> KB    (storage size in KB for client)
SERVER_STORAGE     = args.server_storage * 1e6                 # units: GB -> KB    (storage size in KB for server)
CLIENT_NONLINEAR_STORAGE = args.client_nonlinear               # units: KB / ReLU
SERVER_NONLINEAR_STORAGE = args.server_nonlinear               # units: KB / ReLU

measurements = mapping[args.dataset][args.network]

client_storage_per_inf = measurements.CLIENT_LINEAR_STORAGE * 8 / 1000 + CLIENT_NONLINEAR_STORAGE * measurements.NUM_RELU  #kilobytes
server_storage_per_inf = measurements.SERVER_LINEAR_STORAGE * 8 / 1000 + SERVER_NONLINEAR_STORAGE * measurements.NUM_RELU  #kilobytes

print("client_storage_per_inf and total client storage:", np.around(client_storage_per_inf,2), np.around(CLIENT_STORAGE,2))
print("Percentage of total client storage:", np.around(client_storage_per_inf / CLIENT_STORAGE * 100,2), "%")
print("server_storage_per_inf and total server storage:", np.around(server_storage_per_inf,2), np.around(SERVER_STORAGE,2))
print("Percentage of total server storage:", np.around(server_storage_per_inf / SERVER_STORAGE * 100,2), "%")

all_online = False
if client_storage_per_inf < CLIENT_STORAGE:
    all_online = True


def inference_generator(env, pipe, arrival_rate):
    """
    Generate a new inference request on the client-side.
    """

    global num_clients, trace, last_inf_times, request_times
    for i in itertools.count():
        random_request_time = random.expovariate(arrival_rate)
        cumulative_request_time = last_inf_times + random_request_time
        last_inf_times = cumulative_request_time
        request_times.append(cumulative_request_time)
        yield env.timeout(random_request_time)
        num_clients +=1
        d = {'idx' : num_clients, 'request_time' : env.now}
        pipe.put(d)

def run_online_only(env, pipe):

    """
    If there is not enough storage on the client device
    for a single precompute, then run everything online
    once an inference request comes in
    """

    global clienf_inf_times, num_infs_completed

    while True:

        request = yield pipe.get()
        start_time = env.now
        waiting_time = start_time - request['request_time'] 
        waiting_times.append(waiting_time)

        yield env.timeout(measurements.off_client_compute_keygen)                      # client generates key
        yield env.timeout(measurements.off_client_write_key / upload_bandwidth)                # client sends key to server
        # simulate linear layers
        yield env.timeout(measurements.off_client_compute_he_encrypt.sum())
        yield env.timeout(measurements.off_client_write_linear.sum() / upload_bandwidth)
        yield env.timeout(he_models.he_eval_latency(args.dataset, args.network, args.num_threads_server_he))
        yield env.timeout(measurements.off_server_write_linear.sum() / download_bandwidth)
        yield env.timeout(measurements.off_client_compute_he_decrypt.sum())

        # simulate ReLU layers
        yield env.timeout(gc_models.garble_latency("client", measurements.NUM_RELU, args.num_threads_client)) 
        yield env.timeout(measurements.off_client_compute_encode)
        yield env.timeout(measurements.off_client_write_garbled_c / upload_bandwidth)          # client sends garbled circuit to server

        yield env.timeout(measurements.off_server_write_base_ot / download_bandwidth)          # server sends labels (k_0, k_1)..... BASE OT


        yield env.timeout(measurements.on_client_write_linear[0] / upload_bandwidth)        # client sends initial linear share to server

        for i in range(len(measurements.on_client_write_ext_ot_send)):
            yield env.timeout(measurements.on_server_compute_linear[i])                     # server evaluates linear layers
            yield env.timeout(measurements.on_client_compute_encode[i])                     # server labels can now be encoded
            yield env.timeout(measurements.on_server_write_ext_ot_setup[i] / download_bandwidth) # server sends u_i to server ...    EXT OT
            yield env.timeout(measurements.on_client_write_ext_ot_send[i] / upload_bandwidth)    # client sends (y_0, y_1) to server EXT OT
        yield env.timeout(gc_models.eval_latency("server", measurements.NUM_RELU, args.num_threads_server_gc))

        # process FC layer
        yield env.timeout(measurements.on_server_compute_linear[-1])                        # server computes final layer
        # send prediction to client
        yield env.timeout(measurements.on_server_write_pred / download_bandwidth)                    # server sends prediction to client

        num_infs_completed +=1
        client_inf_times.append(env.now-start_time)


def check_precompute(env, client_storage, server_storage):
    """
    """

    while True:
        yield env.timeout(1) # check every second
        if (client_storage.capacity - client_storage.level) > client_storage_per_inf:
            if (server_storage.capacity - server_storage.level) > server_storage_per_inf:
                yield env.process(offline_client_garbler_phase(env, client_storage, server_storage))

def offline_client_garbler_phase(env, client_storage, server_storage):
    """
    simulates Client_Garbler's Offline Protocol for a network with all non-linear layers as ReLU
    """

    # key generation
    now = env.now
    yield env.timeout(measurements.off_client_compute_keygen)                      # client generates key
    yield env.timeout(measurements.off_client_write_key / upload_bandwidth)                # client sends key to server
    # simulate linear layers
    yield env.timeout(measurements.off_client_compute_he_encrypt.sum())
    yield env.timeout(measurements.off_client_write_linear.sum() / upload_bandwidth)
    yield env.timeout(he_models.he_eval_latency(args.dataset, args.network, args.num_threads_server_he))
    yield env.timeout(measurements.off_server_write_linear.sum() / download_bandwidth)
    yield env.timeout(measurements.off_client_compute_he_decrypt.sum())

    # simulate ReLU layers
    yield env.timeout(gc_models.garble_latency("client", measurements.NUM_RELU, args.num_threads_client)) 
    yield env.timeout(measurements.off_client_compute_encode)
    yield env.timeout(measurements.off_client_write_garbled_c / upload_bandwidth)          # client sends garbled circuit to server

    # oblivious transfer protocol (protocol 4 of https://eprint.iacr.org/2016/602)
    yield env.timeout(measurements.off_server_write_base_ot / download_bandwidth)          # server sends labels (k_0, k_1)..... BASE OT
    yield client_storage.put(client_storage_per_inf)
    yield server_storage.put(server_storage_per_inf)


def online_client_garbler_phase(env, pipe, client_storage, server_storage):
    """
    simulates Client_Garbler's Online Protocol for a network with all non-linear layers as ReLU
    """

    global clienf_inf_times, num_infs_completed, request_times, waiting_times, offline_times
    while True:
        request = yield pipe.get()
        start_time = env.now
        waiting_time = start_time - request['request_time']
        waiting_times.append(waiting_time)

        before_gc = env.now
        yield client_storage.get(client_storage_per_inf)
        yield server_storage.get(server_storage_per_inf)

        offline_times.append(env.now - before_gc)

        yield env.timeout(measurements.on_client_write_linear[0] / upload_bandwidth)        # client sends initial linear share to server

        for i in range(len(measurements.on_client_write_ext_ot_send)):
            yield env.timeout(measurements.on_server_compute_linear[i])                     # server evaluates linear layers
            yield env.timeout(measurements.on_client_compute_encode[i])                     # server labels can now be encoded
            yield env.timeout(measurements.on_server_write_ext_ot_setup[i] / download_bandwidth) # server sends u_i to server ...    EXT OT
            yield env.timeout(measurements.on_client_write_ext_ot_send[i] / upload_bandwidth)    # client sends (y_0, y_1) to server EXT OT
        yield env.timeout(gc_models.eval_latency("server", measurements.NUM_RELU, args.num_threads_server_gc))


        # process FC layer
        yield env.timeout(measurements.on_server_compute_linear[-1])                        # server computes final layer


        # send prediction to client
        yield env.timeout(measurements.on_server_write_pred / download_bandwidth)                    # server sends prediction to client

        num_infs_completed +=1
        client_inf_times.append(env.now-start_time)


avg_inf_times     = np.zeros((args.number_of_runs, args.step))
total_num_clients = np.zeros((args.number_of_runs,args.step))
left_to_service   = np.zeros_like(total_num_clients)
avg_waiting_times = np.zeros_like(total_num_clients)
avg_offline_times = np.zeros_like(total_num_clients)
arrival_rates     = np.linspace(args.start,args.end,args.step)

for exp_number in tqdm(range(args.number_of_runs)):
    for j in range(len(arrival_rates)):

        num_clients        = 0
        num_infs_completed = 0
        client_inf_times   = []
        last_inf_times     = 0
        request_times      = []
        waiting_times      = []
        offline_times      = []

        env = simpy.Environment()
        client_storage = simpy.Container(env, capacity=CLIENT_STORAGE, init=0)
        server_storage = simpy.Container(env, capacity=SERVER_STORAGE, init=0)
        pipe = simpy.Store(env)

        env.process(inference_generator(env, pipe, arrival_rates[j]))                           # begins sampling for clients

        if client_storage_per_inf < CLIENT_STORAGE:
            env.process(check_precompute(env, client_storage, server_storage))            # begins checking to perform offline
            env.process(online_client_garbler_phase(env, pipe, client_storage, server_storage))
        else:
            env.process(run_online_only(env, pipe))


        env.run(until=SIM_TIME)
        avg_inf_times[exp_number, j]      = np.mean(client_inf_times)
        total_num_clients[exp_number, j]  = len(client_inf_times)
        left_to_service[exp_number, j]    = len(pipe.items)
        avg_waiting_times[exp_number, j]  = np.mean(waiting_times)
        if client_storage_per_inf < CLIENT_STORAGE:
            avg_offline_times[exp_number, j]  = np.mean(offline_times)

os.mkdir("{}".format(args.name))
np.save("{}/client_garbler_avg_inf_times.npy".format(args.name), avg_inf_times)
np.save("{}/client_garbler_arrival_rates.npy".format(args.name), arrival_rates )
np.save("{}/client_garbler_total_num_clients.npy".format(args.name), total_num_clients)
np.save("{}/client_garbler_avg_waiting_times.npy".format(args.name), avg_waiting_times)
np.save("{}/client_garbler_left_to_service.npy".format(args.name), left_to_service)
if client_storage_per_inf < CLIENT_STORAGE:
    np.save("{}/client_garbler_avg_offline_times.npy".format(args.name), avg_offline_times)
with open("{}/args.json".format(args.name), 'wt') as f:
    json.dump(vars(args), f, indent=4)
