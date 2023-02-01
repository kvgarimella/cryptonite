import os
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

all_data = get_data("fig7_data", './')
k = "fig7_data"
print(all_data.keys())
print("-"*50)
print("arrival rates:")
print(all_data[k]['arrival_rates'])
print("-"*50)
pprint.pprint(all_data[k]['params'])
print("-"*50)



THRESHOLD = 2
plt.figure(figsize=(10,7))
    
online_time = all_data[k]['avg_inf_times'] - all_data[k]['avg_offline_times']
online_time = np.mean(online_time, 0) / 60.
offline_time = np.mean(all_data[k]['avg_offline_times'],0) / 60.
waiting_time = np.mean(all_data[k]['avg_waiting_times'], 0) / 60.

mask = all_data[k]['left_to_service'].mean(0) < THRESHOLD

y1 = np.zeros_like(online_time)
y2 = online_time
plt.fill_between(all_data[k]['arrival_rates'][mask],y1=y1[mask], y2=y2[mask], color='lightslategray', label='Online Time')
y1 = online_time
y2 = online_time + offline_time
plt.fill_between(all_data[k]['arrival_rates'][mask],y1=y1[mask], y2=y2[mask], color='cornflowerblue', label='Offline Time')
y1 = online_time + offline_time
y2 = online_time + offline_time + waiting_time
plt.fill_between(all_data[k]['arrival_rates'][mask],y1=y1[mask], y2=y2[mask], color='lightcoral', label='Waiting in Queue')
    
plt.xlabel("Workload (requests per second)")
plt.ylabel("Mean Inference Latency (minutes)")
plt.xticks(rotation = 45);
plt.legend(loc='upper left')
plt.show()
