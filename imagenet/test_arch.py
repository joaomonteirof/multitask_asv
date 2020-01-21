from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG19')
elif args.model == 'resnet':
	model = resnet.ResNet50()
elif args.model == 'densenet':
	model = densenet.DenseNet121()

batch = torch.rand(3, 3, 224, 224)

emb = model.forward(batch)

print(emb.size())

out = model.out_proj(emb, torch.zeros(emb.size(0)))

print(out.size())
