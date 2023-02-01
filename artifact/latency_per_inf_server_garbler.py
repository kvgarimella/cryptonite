import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../simulator/experiments/utils/")
import gc_models

num_relus = {'resnet18' : [557056, 2228224],
             'resnet32' : [303104, 1212416],
             'vgg16'    : [276480, 1105920]}


NUM_THREADS = 4
SEC_PER_MIN = 60.
import cifar10_server_garbler_utils_r18
import cifar10_server_garbler_utils_r32
import cifar10_server_garbler_utils_vgg16

import tiny_server_garbler_utils_r18
import tiny_server_garbler_utils_r32
import tiny_server_garbler_utils_vgg16

# PLOT CIFAR
r32_prev = 0
vgg_prev = 0
r18_prev = 0

r32_after = cifar10_server_garbler_utils_r32.off_server_compute_he_eval.sum() / SEC_PER_MIN
vgg_after = cifar10_server_garbler_utils_vgg16.off_server_compute_he_eval.sum() / SEC_PER_MIN 
r18_after = cifar10_server_garbler_utils_r18.off_server_compute_he_eval.sum() / SEC_PER_MIN

plt.bar(0, r32_after, 1, bottom=r32_prev, color='cornflowerblue', edgecolor='k', label='HE.Eval')
plt.bar(1, vgg_after, 1, bottom=vgg_prev, color='cornflowerblue', edgecolor='k')
plt.bar(2, r18_after, 1, bottom=r18_prev, color='cornflowerblue', edgecolor='k')

r32_prev += r32_after
vgg_prev += vgg_after
r18_prev += r18_after

r32_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['resnet32'][0],num_threads=NUM_THREADS) / SEC_PER_MIN
vgg_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['vgg16'][0],num_threads=NUM_THREADS) / SEC_PER_MIN
r18_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['resnet18'][0],num_threads=NUM_THREADS) / SEC_PER_MIN

plt.bar(0, r32_after, 1, bottom=r32_prev, color='lightcoral', edgecolor='k', label='GC.Eval')
plt.bar(1, vgg_after, 1, bottom=vgg_prev, color='lightcoral', edgecolor='k')
plt.bar(2, r18_after, 1, bottom=r18_prev, color='lightcoral', edgecolor='k')

r32_prev += r32_after
vgg_prev += vgg_after
r18_prev += r18_after

r32_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['resnet32'][0],num_threads=NUM_THREADS) / SEC_PER_MIN
vgg_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['vgg16'][0],num_threads=NUM_THREADS) / SEC_PER_MIN
r18_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['resnet18'][0],num_threads=NUM_THREADS) / SEC_PER_MIN

plt.bar(0, r32_after, 1, bottom=r32_prev, color='mediumaquamarine', edgecolor='k', label='GC.GARBLE')
plt.bar(1, vgg_after, 1, bottom=vgg_prev, color='mediumaquamarine', edgecolor='k')
plt.bar(2, r18_after, 1, bottom=r18_prev, color='mediumaquamarine', edgecolor='k')


# PLOT TINYIMAGENET
r32_prev = 0
vgg_prev = 0
r18_prev = 0

r32_after = tiny_server_garbler_utils_r32.off_server_compute_he_eval.sum() / SEC_PER_MIN
vgg_after = tiny_server_garbler_utils_vgg16.off_server_compute_he_eval.sum() / SEC_PER_MIN 
r18_after = tiny_server_garbler_utils_r18.off_server_compute_he_eval.sum() / SEC_PER_MIN

plt.bar(4, r32_after, 1, bottom=r32_prev, color='cornflowerblue', edgecolor='k')
plt.bar(5, vgg_after, 1, bottom=vgg_prev, color='cornflowerblue', edgecolor='k')
plt.bar(6, r18_after, 1, bottom=r18_prev, color='cornflowerblue', edgecolor='k')

r32_prev += r32_after
vgg_prev += vgg_after
r18_prev += r18_after

r32_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['resnet32'][1],num_threads=NUM_THREADS) / SEC_PER_MIN
vgg_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['vgg16'][1],num_threads=NUM_THREADS) / SEC_PER_MIN
r18_after  = gc_models.eval_latency(evaluator="client",num_relus=num_relus['resnet18'][1],num_threads=NUM_THREADS) / SEC_PER_MIN

plt.bar(4, r32_after, 1, bottom=r32_prev, color='lightcoral', edgecolor='k')
plt.bar(5, vgg_after, 1, bottom=vgg_prev, color='lightcoral', edgecolor='k')
plt.bar(6, r18_after, 1, bottom=r18_prev, color='lightcoral', edgecolor='k')

r32_prev += r32_after
vgg_prev += vgg_after
r18_prev += r18_after

r32_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['resnet32'][1],num_threads=NUM_THREADS) / SEC_PER_MIN
vgg_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['vgg16'][1],num_threads=NUM_THREADS) / SEC_PER_MIN
r18_after  = gc_models.garble_latency(garbler="server",num_relus=num_relus['resnet18'][1],num_threads=NUM_THREADS) / SEC_PER_MIN

plt.bar(4, r32_after, 1, bottom=r32_prev, color='mediumaquamarine', edgecolor='k')
plt.bar(5, vgg_after, 1, bottom=vgg_prev, color='mediumaquamarine', edgecolor='k')
plt.bar(6, r18_after, 1, bottom=r18_prev, color='mediumaquamarine', edgecolor='k')

plt.legend()

plt.ylabel("Minutes")
plt.xticks([0,1,2,4,5,6], ['C-R32', 'C-VGG16', 'C-R18', 'T-R32', 'T-VGG16', 'T-R18'])

plt.show()
