import os
import re
import json
import pprint
import numpy as np
import matplotlib.pyplot as plt

"""
Helper files
"""
def get_prefix(npys):
    for file in npys:
        if "garbler" in file:
            return file.split("_")[0]

def get_params(base, directory):
    f = open(base+directory + '/args.json')
    data = json.load(f)
    f.close()
    return data

"""
Loads data for a particular network and dataset
"""
def get_data(ds_nw, base):
    if base[-1] != '/':
        base = base + "/"
    directories = [dir for dir in os.listdir(base) if ds_nw in dir]
    data = {}
    for directory in directories:
        npys = sorted(os.listdir(base + directory))
        data[directory] = {}
        prefix = get_prefix(npys)
        data[directory]["arrival_rates"]     = np.load(base + directory + "/{}_garbler_arrival_rates.npy".format(prefix))
        data[directory]["avg_inf_times"]     = np.load(base + directory + "/{}_garbler_avg_inf_times.npy".format(prefix))
        data[directory]["avg_offline_times"] = np.load(base + directory + "/{}_garbler_avg_offline_times.npy".format(prefix))
        data[directory]["avg_waiting_times"] = np.load(base + directory + "/{}_garbler_avg_waiting_times.npy".format(prefix))
        data[directory]["left_to_service"]   = np.load(base + directory + "/{}_garbler_left_to_service.npy".format(prefix))
        data[directory]["num_client_served"] = np.load(base + directory + "/{}_garbler_total_num_clients.npy".format(prefix))
        data[directory]['params'] = get_params(base, directory)
    return data

all_data = get_data("garbler_", './')
THRESHOLD = 2
SEC_PER_MIN = 60.
plt.figure(figsize=(10,7))
colors = ['cornflowerblue', 'lightcoral', 'mediumaquamarine', 'sandybrown']

experiment_keys = ['server_garbler_16gb', 'server_garbler_32gb', 'server_garbler_64gb', 'client_garbler_16gb']

for k in experiment_keys:
    if "server" in k:
        label='Server-Garbler {}GB'
    else:
        label='Client-Garbler {}GB'
    i = re.findall(r'\d+',k)[0]

    total_time = all_data[k]['avg_waiting_times'] + all_data[k]['avg_inf_times']
    mean_inf_times = np.mean(total_time, 0) / SEC_PER_MIN
    mask = all_data[k]['left_to_service'].mean(0) < THRESHOLD


    plt.plot(all_data[k]['arrival_rates'][mask], mean_inf_times[mask],label=label.format(i), color=colors.pop(), linewidth=5)


plt.xlabel("Workload (requests per second)")
plt.ylabel("Mean Inference Latency (minutes)")
plt.xticks(rotation = 45);
plt.legend(loc='upper left')
plt.show()
