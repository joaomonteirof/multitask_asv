'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax



class Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(4*growth_rate)
		self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat([out,x], 1)
		return out


class Transition(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Transition, self).__init__()
		self.bn = nn.BatchNorm2d(in_planes)
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv(F.relu(self.bn(x)))
		out = F.avg_pool2d(out, 2)
		return out


class DenseNet(nn.Module):
	def __init__(self, block, nblocks, sm_type, growth_rate=12, reduction=0.5, num_classes=1000):
		super(DenseNet, self).__init__()
		self.growth_rate = growth_rate

		self.n_classes = num_classes

		num_planes = 2*growth_rate
		self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, padding=1, bias=False)

		self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
		num_planes += nblocks[0]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans1 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
		num_planes += nblocks[1]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans2 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
		num_planes += nblocks[2]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans3 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
		num_planes += nblocks[3]*growth_rate

		self.bn = nn.BatchNorm2d(num_planes)

		self.lin_proj = nn.Sequential(nn.Linear(1024*6*6, 350))

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=1024*6*6, output_features=num_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=1024*6*6, output_features=num_classes)
		else:
			raise NotImplementedError

	def _make_dense_layers(self, block, in_planes, nblock):
		layers = []
		for i in range(nblock):
			layers.append(block(in_planes, self.growth_rate))
			in_planes += self.growth_rate
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.trans1(self.dense1(out))
		out = self.trans2(self.dense2(out))
		out = self.trans3(self.dense3(out))
		out = self.dense4(out)
		out = F.avg_pool2d(F.relu(self.bn(out)), 4)
		out = out.view(out.size(0), -1)
		emb = self.lin_proj(out)

		return emb, out

def DenseNet121(sm_type='softmax', n_classes=1000):
	return DenseNet(Bottleneck, [6,12,24,16], sm_type, growth_rate=32, num_classes=n_classes)

def DenseNet169(sm_type='softmax'):
	return DenseNet(Bottleneck, [6,12,32,32], sm_type, growth_rate=32, num_classes=n_classes)

def DenseNet201(sm_type='softmax'):
	return DenseNet(Bottleneck, [6,12,48,32], sm_type, growth_rate=32, num_classes=n_classes)

def DenseNet161(sm_type='softmax'):
	return DenseNet(Bottleneck, [6,12,36,24], sm_type, growth_rate=48, num_classes=n_classes)

def densenet_cifar(sm_type='softmax'):
	return DenseNet(Bottleneck, [6,12,24,16], sm_type, growth_rate=12, num_classes=n_classes)
