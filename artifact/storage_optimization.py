import numpy as np
import matplotlib.pyplot as plt

# NUM ReLUs from PyTorch
num_relus = {'resnet18' : [557056, 2228224, 27295744],
             'resnet32' : [303104, 1212416, 14852096],
             'vgg16'    : [276480, 1105920, 13547520]}

# GC Storage per ReLU
server_garbler_storage_per_relu = 18248 # Bytes / ReLU
for k, v in num_relus.items():
    storage = [server_garbler_storage_per_relu * num_relu * 1e-9  for num_relu in v]
    num_relus[k] = storage 


idx = 0
plt.bar(0,num_relus['vgg16'][idx],.4, color='lightcoral', label='Server-Garbler', edgecolor='k', align='edge')
plt.bar(1,num_relus['resnet32'][idx],.4, color='lightcoral', edgecolor='k', align='edge')
plt.bar(2,num_relus['resnet18'][idx],.4, color='lightcoral',  edgecolor='k', align='edge')

idx += 1
plt.bar(4,num_relus['vgg16'][idx],.4, color='lightcoral', edgecolor='k', align='edge')
plt.bar(5,num_relus['resnet32'][idx],.4, color='lightcoral', edgecolor='k', align='edge')
plt.bar(6,num_relus['resnet18'][idx],.4, color='lightcoral', edgecolor='k', align='edge')




# CLIENT-GARBLER
num_relus = {'resnet18' : [557056, 2228224, 27295744],
             'resnet32' : [303104, 1212416, 14852096],
             'vgg16'    : [276480, 1105920, 13547520]}

# GC Storage per ReLU
client_garbler_storage_per_relu = 3590   # Bytes / ReLU
for k, v in num_relus.items():
    storage = [client_garbler_storage_per_relu * num_relu * 1e-9  for num_relu in v]
    num_relus[k] = storage 


idx = 0
plt.bar(0.4,num_relus['vgg16'][idx],.4, color='cornflowerblue', label='Client-Garbler', edgecolor='k', align='edge')
plt.bar(1.4,num_relus['resnet32'][idx],.4, color='cornflowerblue',  edgecolor='k', align='edge')
plt.bar(2.4,num_relus['resnet18'][idx],.4, color='cornflowerblue',  edgecolor='k', align='edge')

idx += 1
plt.bar(4.4,num_relus['vgg16'][idx],.4, color='cornflowerblue', edgecolor='k', align='edge')
plt.bar(5.4,num_relus['resnet32'][idx],.4, color='cornflowerblue', edgecolor='k', align='edge')
plt.bar(6.4,num_relus['resnet18'][idx],.4, color='cornflowerblue', edgecolor='k', align='edge')


plt.xlabel("Dataset")
plt.ylabel("GB")
plt.xticks([.4, 1.4, 2.4, 4.4, 5.4, 6.4], ['C-VGG16', 'C-R32', 'C-R18', 'T-VGG16', 'T-R32', 'T-R18'])
plt.legend()
plt.show()
