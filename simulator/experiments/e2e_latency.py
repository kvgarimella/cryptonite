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

from utils import cifar10_server_garbler_utils_r18
from utils import cifar10_server_garbler_utils_r32
from utils import cifar10_server_garbler_utils_vgg16
from utils import tiny_server_garbler_utils_r18
from utils import tiny_server_garbler_utils_r32
from utils import tiny_server_garbler_utils_vgg16

from utils import cifar10_client_garbler_utils_r18
from utils import cifar10_client_garbler_utils_r32
from utils import cifar10_client_garbler_utils_vgg16
from utils import tiny_client_garbler_utils_r18
from utils import tiny_client_garbler_utils_r32
from utils import tiny_client_garbler_utils_vgg16

mapping = {0 : {"cifar10":       {"resnet18" : cifar10_server_garbler_utils_r18,
                                  "resnet32"  : cifar10_server_garbler_utils_r32,
                                  "vgg16"     : cifar10_server_garbler_utils_vgg16},
                "tinyimagenet" : {"resnet18" : tiny_server_garbler_utils_r18,
                                  "resnet32"  : tiny_server_garbler_utils_r32,
                                  "vgg16"     : tiny_server_garbler_utils_vgg16}},

           1 : {"cifar10":       {"resnet18" : cifar10_client_garbler_utils_r18,
                                  "resnet32"  : cifar10_client_garbler_utils_r32,
                                  "vgg16"     : cifar10_client_garbler_utils_vgg16},
                "tinyimagenet" : {"resnet18" : tiny_client_garbler_utils_r18,
                                  "resnet32"  : tiny_client_garbler_utils_r32,
                                  "vgg16"     : tiny_client_garbler_utils_vgg16}}}

parser = argparse.ArgumentParser(description='End-to-End Private Inference Latency')
parser.add_argument("--upload-bandwidth",   type=float, default=100.,       help="Upload Bandwidth in Mbps")
parser.add_argument("--download-bandwidth", type=float, default=500.,       help="Download Bandwidth in Mbps")
parser.add_argument("--network",            type=str,   default="resnet18", help="resnet18, resnet32, vgg16")
parser.add_argument("--dataset",            type=str,   default="cifar10",  help="cifar10, tinyimagenet")
parser.add_argument("--num-threads-client", type=int,   default=1,          help="Number of threads client-side")
parser.add_argument("--num-threads-server-he", type=int,default=1,          help="Number of threads - Server HE" )
parser.add_argument("--num-threads-server-gc", type=int,default=1,          help="Number of threads - Server GC")
parser.add_argument("--protocol",           type=int,   default=0, help="Server-Garber (0) or Client-Garbler (1)")
parser.add_argument("--he-speedup",         type=float, default=1., help="HE Speedup")
parser.add_argument("--gc-speedup",         type=float, default=1., help="GC Speedup")
parser.add_argument("--bw-speedup",         type=float, default=1., help='BW Speedup')

args = parser.parse_args()

upload_bandwidth   = args.upload_bandwidth * (1/8) * 1e6   # units: Mbps -> MBps -> (bytes / second)
download_bandwidth = args.download_bandwidth * (1/8) * 1e6 # units: Mbps -> MBps -> (bytes / second)


measurements = mapping[args.protocol][args.dataset][args.network]


def run_client_garbler_protocol():

    total_time = 0
    communication_time = 0
    downlink_time = 0
    uplink_time = 0


    OT_TIME           = 0
    GC_GARBLE_TIME    = 0
    HE_ENC_TIME       = 0
    HE_DEC_TIME       = 0
    HE_EVAL_TIME      = 0
    GC_EVAL_TIME      = 0
    ONLINE_COMM_TIME  = 0
    OFFLINE_COMM_TIME = 0
    SS_EVAL_TIME      = 0



    

    ## OFFLINE PORTION
    # generate key and send to server
    total_time += measurements.off_client_compute_keygen
    before = total_time
    total_time += measurements.off_client_write_key / upload_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.off_client_write_key / upload_bandwidth

    # encrypt and send linear layers to server
    before = total_time
    total_time += measurements.off_client_compute_he_encrypt.sum()
    HE_EVAL_TIME += (total_time - before)
    before = total_time
    total_time += measurements.off_client_write_linear.sum() / upload_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.off_client_write_linear.sum() / upload_bandwidth

    # evaluate linear layers on the server
    before = total_time
    total_time += he_models.he_eval_latency(args.dataset, args.network, args.num_threads_server_he)
    HE_EVAL_TIME +=  (total_time - before)

    # send encrypted evaluations to client
    before = total_time
    total_time += measurements.off_server_write_linear.sum() / download_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.off_server_write_linear.sum() / download_bandwidth

    # decrypt linear evaluations on the client device
    before = total_time
    total_time += measurements.off_client_compute_he_decrypt.sum()
    HE_EVAL_TIME += (total_time - before)

    # client garbles ReLUS and encodes server inputs
    before = total_time
    total_time += gc_models.garble_latency("client", measurements.NUM_RELU, args.num_threads_client) 
    total_time += measurements.off_client_compute_encode
    GC_GARBLE_TIME = total_time - before

    # send GC and encodings to server
    before = total_time
    total_time += measurements.off_client_write_garbled_c / upload_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.off_client_write_garbled_c / upload_bandwidth


    # engage in base oblivious transfer protocol
    before = total_time
    total_time += measurements.off_server_write_base_ot / download_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.off_server_write_base_ot / download_bandwidth
  


    ## ONLINE PORTION
    # send initial linear layer
    offline_time = total_time + 0
    before = total_time
    total_time += measurements.on_client_write_linear[0] / upload_bandwidth
    ONLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.on_client_write_linear[0] / upload_bandwidth

    # encode client inputs for GCs
    before = total_time
    total_time += measurements.on_client_compute_encode.sum()
    GC_EVAL_TIME += (total_time - before)

    # engage in extended oblivious transfer
    before = total_time
    total_time += measurements.on_server_write_ext_ot_setup.sum() / download_bandwidth
    total_time += measurements.on_client_write_ext_ot_send.sum() / upload_bandwidth
    ONLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.on_server_write_ext_ot_setup.sum() / download_bandwidth
    uplink_time += measurements.on_client_write_ext_ot_send.sum() / upload_bandwidth




    # evaluate linear layers using secret sharing on the server
    before = total_time
    total_time += measurements.on_server_compute_linear.sum()
    SS_EVAL_TIME = total_time - before

    # evaulate Garbled Circuits on the server
    before = total_time
    total_time += gc_models.eval_latency("server", measurements.NUM_RELU, args.num_threads_server_gc)
    GC_EVAL_TIME += (total_time - before)


    # send final prediction to the client
    before = total_time
    total_time += measurements.on_server_write_pred / download_bandwidth
    ONLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.on_server_write_pred / download_bandwidth
    online_time = total_time - offline_time


    print("Offline:", offline_time)
    print("Online:", online_time)
    print("Uplink Time:", uplink_time)
    print("Downlink Time:", downlink_time)
    print("Comm Time:", uplink_time + downlink_time)
    print("-"*50)
    print("GC_GARBLE_TIME    =",GC_GARBLE_TIME/args.gc_speedup)
    print("HE_EVAL_TIME      =",HE_EVAL_TIME/args.he_speedup)
    print("GC_EVAL_TIME      =",GC_EVAL_TIME/args.gc_speedup)
    print("ONLINE_COMM_TIME  =",ONLINE_COMM_TIME/args.bw_speedup)
    print("OFFLINE_COMM_TIME =",OFFLINE_COMM_TIME/args.bw_speedup)
    print("SS_EVAL_TIME      =",SS_EVAL_TIME)



    return total_time





def run_server_garbler_protocol():

    total_time = 0
    communication_time = 0
    downlink_time = 0
    uplink_time = 0

    GC_GARBLE_TIME    = 0.
    HE_EVAL_TIME      = 0.
    GC_EVAL_TIME      = 0.
    ONLINE_COMM_TIME  = 0.
    OFFLINE_COMM_TIME = 0.
    SS_EVAL_TIME      = 0.


    

    ## OFFLINE PORTION
    # generate key and send to server
    total_time += measurements.off_client_compute_keygen
    before = total_time
    total_time += measurements.off_client_write_key / upload_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.off_client_write_key / upload_bandwidth

    # encrypt and send linear layers to server
    before = total_time
    total_time += measurements.off_client_compute_he_encrypt.sum()
    HE_EVAL_TIME += (total_time - before)
    before = total_time
    total_time += measurements.off_client_write_linear.sum() / upload_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    uplink_time += measurements.off_client_write_linear.sum() / upload_bandwidth

    # evaluate linear layers on the server
    before = total_time
    total_time += he_models.he_eval_latency(args.dataset, args.network, args.num_threads_server_he)
    HE_EVAL_TIME += (total_time - before)

    # send encrypted evaluations to client
    before = total_time
    total_time += measurements.off_server_write_linear.sum() / download_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.off_server_write_linear.sum() / download_bandwidth

    # decrypt linear evaluations on the client device
    before = total_time
    total_time += measurements.off_client_compute_he_decrypt.sum()
    HE_EVAL_TIME += (total_time - before)

    # server garbles ReLUS and encodes server inputs
    before = total_time
    total_time += gc_models.garble_latency("server", measurements.NUM_RELU, args.num_threads_server_gc)
    total_time += measurements.off_server_compute_encode
    GC_GARBLE_TIME = total_time - before

    # send GC and encodings to server
    before = total_time
    total_time += measurements.off_server_write_garbled_c / download_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.off_server_write_garbled_c / download_bandwidth

    # engage in base+ext oblivious transfer protocol
    before = total_time
    total_time += measurements.off_client_write_base_ot / upload_bandwidth
    total_time += measurements.off_client_write_ext_ot_setup / upload_bandwidth
    total_time += measurements.off_server_write_ext_ot_send / download_bandwidth
    OFFLINE_COMM_TIME += (total_time - before)

    uplink_time += measurements.off_client_write_base_ot / upload_bandwidth
    uplink_time += measurements.off_client_write_ext_ot_setup / upload_bandwidth
    downlink_time += measurements.off_server_write_ext_ot_send / download_bandwidth




  


    ## ONLINE PORTION
    # send linear layers to server
    offline_time = total_time+0
    before = total_time
    total_time += measurements.on_client_write_linear.sum() / upload_bandwidth
    ONLINE_COMM_TIME = total_time - before
    uplink_time += measurements.on_client_write_linear.sum() / upload_bandwidth

    # encode and send client inputs for GCs
    before = total_time
    total_time += measurements.on_server_compute_encode.sum()
    GC_EVAL_TIME += (total_time - before)
    before = total_time
    total_time += measurements.on_server_write_relu.sum() / download_bandwidth
    ONLINE_COMM_TIME += (total_time - before)
    downlink_time += measurements.on_server_write_relu.sum() / download_bandwidth

    # evaluate linear layers using secret sharing on the server
    before = total_time
    total_time += measurements.on_server_compute_linear.sum()
    SS_EVAL_TIME = total_time - before

    # evaulate Garbled Circuits on the client device
    before = total_time
    total_time += gc_models.eval_latency("client", measurements.NUM_RELU, args.num_threads_client)
    GC_EVAL_TIME += (total_time - before)
    

    # send final prediction to the client
    before = total_time
    total_time += measurements.on_server_write_pred / download_bandwidth
    ONLINE_COMM_TIME += (total_time - before)

    downlink_time += measurements.on_server_write_pred / download_bandwidth
    online_time = total_time - offline_time
    print("Offline:", offline_time)
    print("Online:", online_time)
    print("Uplink Time:", uplink_time)
    print("Downlink Time:", downlink_time)
    print("Comm Time:", uplink_time + downlink_time)
    print("-"*50)
    print("GC_GARBLE_TIME    =",GC_GARBLE_TIME/args.gc_speedup)
    print("HE_EVAL_TIME      =",HE_EVAL_TIME/args.he_speedup)
    print("GC_EVAL_TIME      =",GC_EVAL_TIME/args.gc_speedup)
    print("ONLINE_COMM_TIME  =",ONLINE_COMM_TIME/args.bw_speedup)
    print("OFFLINE_COMM_TIME =",OFFLINE_COMM_TIME/args.bw_speedup)
    print("SS_EVAL_TIME      =",SS_EVAL_TIME)



    return total_time


if args.protocol == 0:
    # server-garbler model
    time = run_server_garbler_protocol()
    print("Time using Server-Garbler:", time)
else:
    time = run_client_garbler_protocol()
    print("Time using Client-Garbler:", time)


