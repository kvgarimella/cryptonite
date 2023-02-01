import torch
import torchvision
import torch.nn as nn
import argparse

import resnet18
import resnet32
import vgg16

parser = argparse.ArgumentParser(description='Count ReLUs in ResNet-[18/32] or VGG-16')
parser.add_argument("--network", type=str,   default="resnet18", help="resnet18, resnet32, vgg16")
parser.add_argument("--dataset", type=str,   default="cifar10",  help="cifar10, tinyimagenet")
args = parser.parse_args()

mapping = {'cifar'        : [10, 32],
           'tinyimagenet' : [200, 64],
           'imagenet'     : [1000, 224]}
try:
  num_classes = mapping[args.dataset][0]
  image_size = mapping[args.dataset][1]
except:
  raise Exception("Error: unrecognized dataset")

if args.network == 'resnet18':
  net = resnet18.ResNet18(num_classes=num_classes)
elif args.network == 'resnet32':
  net = resnet32.ResNet32(num_classes=num_classes)
elif args.network == 'vgg16':
  net = vgg16.VGG16(num_classes=num_classes)
else:
  raise Exception("Error: unrecognized network")


# create hook function
total_num_relus = 0
def update_num_relus(self, input, output):
  global total_num_relus
  total_num_relus += output.numel()

# instantiate model and register hook for each relu call
for name, module in net.named_modules():
  if isinstance(module, nn.ReLU):
    module.register_forward_hook(update_num_relus)

net(torch.randn(1,3,image_size,image_size))
print("Number of ReLUs:", total_num_relus)