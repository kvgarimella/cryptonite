import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../simulator/experiments/utils/")
import he_models


plt.bar(0,he_models.he_eval_latency('cifar10', 'resnet32', 1),.4, color='lightcoral', edgecolor='k', align='edge')
plt.bar(1,he_models.he_eval_latency('cifar10', 'vgg16', 1),.4, color='lightcoral', label='Sequential', edgecolor='k', align='edge')
plt.bar(2,he_models.he_eval_latency('cifar10', 'resnet18', 1),.4, color='lightcoral',  edgecolor='k', align='edge')

plt.bar(4,he_models.he_eval_latency('tinyimagenet', 'resnet32', 1),.4, color='lightcoral', edgecolor='k', align='edge')
plt.bar(5,he_models.he_eval_latency('tinyimagenet', 'vgg16', 1),.4, color='lightcoral',  edgecolor='k', align='edge')
plt.bar(6,he_models.he_eval_latency('tinyimagenet', 'resnet18', 1),.4, color='lightcoral',  edgecolor='k', align='edge')


plt.bar(0.4,he_models.he_eval_latency('cifar10', 'resnet32', 31),.4, color='cornflowerblue', edgecolor='k', align='edge')
plt.bar(1.4,he_models.he_eval_latency('cifar10', 'vgg16', 13),.4, color='cornflowerblue', label='Layer Parallel HE', edgecolor='k', align='edge')
plt.bar(2.4,he_models.he_eval_latency('cifar10', 'resnet18', 17),.4, color='cornflowerblue',  edgecolor='k', align='edge')

plt.bar(4.4,he_models.he_eval_latency('tinyimagenet', 'resnet32', 31),.4, color='cornflowerblue', edgecolor='k', align='edge')
plt.bar(5.4,he_models.he_eval_latency('tinyimagenet', 'vgg16', 13),.4, color='cornflowerblue',  edgecolor='k', align='edge')
plt.bar(6.4,he_models.he_eval_latency('tinyimagenet', 'resnet18', 17),.4, color='cornflowerblue',  edgecolor='k', align='edge')

plt.xlabel("Dataset")
plt.ylabel("GB")
plt.xticks([.4, 1.4, 2.4, 4.4, 5.4, 6.4], ['C-R32', 'C-VGG16', 'C-R18', 'T-R32', 'T-VGG16', 'T-R18'])
plt.legend()
plt.show()

