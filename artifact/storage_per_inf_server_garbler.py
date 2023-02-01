import numpy as np
import matplotlib.pyplot as plt

# NUM ReLUs from PyTorch
num_relus = {'resnet18' : [557056, 2228224, 27295744],
             'resnet32' : [303104, 1212416, 14852096],
             'vgg16'    : [276480, 1105920, 13547520]}

# GC Storage per ReLU
storage_per_relu = 18248  # Bytes / ReLU
for k, v in num_relus.items():
    storage = [storage_per_relu * num_relu * 1e-9  for num_relu in v]
    num_relus[k] = storage 


idx = 0
plt.bar(0,num_relus['vgg16'][idx],1, color='cornflowerblue', label='VGG16', edgecolor='k')
plt.bar(1,num_relus['resnet32'][idx],1, color='lightcoral', label='R32', edgecolor='k')
plt.bar(2,num_relus['resnet18'][idx],1, color='mediumaquamarine', label='R18', edgecolor='k')

idx += 1
plt.bar(4,num_relus['vgg16'][idx],1, color='cornflowerblue', edgecolor='k')
plt.bar(5,num_relus['resnet32'][idx],1, color='lightcoral', edgecolor='k')
plt.bar(6,num_relus['resnet18'][idx],1, color='mediumaquamarine', edgecolor='k')

idx += 1
plt.bar(8,num_relus['vgg16'][idx],1, color='cornflowerblue', edgecolor='k')
plt.bar(9,num_relus['resnet32'][idx],1, color='lightcoral', edgecolor='k')
plt.bar(10,num_relus['resnet18'][idx],1, color='mediumaquamarine', edgecolor='k')

plt.xticks([1,5,9], labels=['C10', 'Tiny', 'ImageNet'])
plt.yscale("log")

plt.xlabel("Dataset")
plt.ylabel("GB")
plt.legend()

plt.show()
