import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax


cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, sm_type='softmax', n_classes=1000):
		super(VGG, self).__init__()

		self.n_classes = n_classes
		
		self.features = self._make_layers(cfg[vgg_name])
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.lin_proj = nn.Sequential(nn.Linear(512 * 7 * 7, 350))

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=512 * 7 * 7, output_features=self.n_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=512 * 7 * 7, output_features=self.n_classes)
		else:
			raise NotImplementedError

	def forward(self, x):
		features = self.avgpool(self.features(x))
		features_out = features.view(features.size(0), -1)
		features_emb = self.lin_proj(features_out)
		return features_emb, features_out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)
