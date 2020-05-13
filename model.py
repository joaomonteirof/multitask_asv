import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.losses import AMSoftmax, Softmax

class SelfAttention(nn.Module):
	def __init__(self, hidden_size, n_heads=1, mean_only=False):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Conv1d(hidden_size, n_heads, kernel_size=1, stride=1, padding=0, bias=False)

		self.mean_only = mean_only

		init.kaiming_uniform_(self.att_weights.weight)

	def forward(self, inputs):

		weights = self.att_weights(inputs)

		attentions = F.softmax(torch.tanh(weights),dim=-1)
		inputs = inputs.unsqueeze(1).repeat(1,attentions.size(1), 1, 1)
		weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		if self.mean_only:
			return weighted.sum(-1).view(inputs.size(0), -1)
		else:
			noise = 1e-5*torch.randn(weighted.size())

			noise = noise.to(inputs.device)

			avg_repr, std_repr = weighted.sum(-1).view(inputs.size(0), -1), (weighted+noise).std(-1).view(inputs.size(0), -1)
			representations = torch.cat((avg_repr,std_repr),1)

			return representations

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes))

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential( nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes) )

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out


class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out

class ResNet_mfcc(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_mfcc, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

class ResNet_34(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBlock, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_34, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

class ResNet_lstm(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_lstm, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.lstm = nn.LSTM(block.expansion*512, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(2*512+256,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2, 0, 1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 256)
		c0 = torch.zeros(2*2, batch_size, 256)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, (h_, c_) = self.lstm(x, (h0, c0))


		stats = self.attention(out_seq.permute(1,2,0).contiguous())

		x = torch.cat([stats,h_.mean(0)],dim=1)

		fc = F.relu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)
		return mu, fc

class ResNet_qrnn(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_qrnn, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		from torchqrnn import QRNN

		self.qrnn = QRNN(block.expansion*512, 512, num_layers=2, dropout=0.3)

		self.fc = nn.Linear(1536,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2,0,1)

		out_seq, h_ = self.qrnn(x)
		stats = self.attention(out_seq.permute(1,2,0).contiguous())
		x = torch.cat([stats,h_.mean(0)],dim=1)
		fc = F.relu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)
		return mu, fc

class ResNet_large(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,23,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_large, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)

		return mu, fc

class ResNet_stats(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 32
		super(ResNet_stats, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		x = torch.cat([x.mean(-1), x.std(-1)], dim=1)

		fc = F.relu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu, fc

class ResNet_small(nn.Module):
	def __init__(self, n_z=256, layers=[2,2,2,2], block=PreActBlock, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 16
		super(ResNet_small, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

class ResNet_2d(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=PreActBlock, proj_size=0, ncoef=23, sm_type='none', delta=False):
		self.in_planes = 16
		super(ResNet_2d, self).__init__()

		self.conv1 = nn.Conv2d(3 if delta else 1, 16, kernel_size=3, stride=1, padding=1, bias=False)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.conv_out = nn.Conv2d(block.expansion*512, 512, kernel_size=(6,1), stride=1, padding=0, bias=False)

		self.fc = nn.Linear(512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.conv_out(x)
		x = x.squeeze(2)

		stats = self.attention(x.contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		mu = x.mean(dim=2, keepdim=True)
		std = (x+torch.randn_like(x)*1e-6).std(dim=2, keepdim=True)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super(TDNN, self).__init__()
		self.delta=delta
		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=2, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=3, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = self.model(x.squeeze(1))
		x = self.pooling(x)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class TDNN_att(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super(TDNN_att, self).__init__()
		self.delta=delta

		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = SelfAttention(1500)

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(1500*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = self.model(x.squeeze(1))
		x = self.pooling(x).unsqueeze(-1)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class TDNN_multihead(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, n_heads=4, sm_type='none', delta=False):
		super(TDNN_multihead, self).__init__()
		self.delta=delta

		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.attention = nn.TransformerEncoderLayer(d_model=1500, nhead=n_heads, dim_feedforward=512, dropout=0.1)
		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(1500*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = x.squeeze(1)
		x = self.model(x)
		x = x.permute(2,0,1)
		x = self.attention(x)
		x = x.permute(1,2,0)
		x = self.pooling(x)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class TDNN_lstm(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super(TDNN_lstm, self).__init__()
		self.delta=delta

		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = nn.LSTM(1500, 512, 2, bidirectional=True, batch_first=False)
		self.attention = SelfAttention(1024)

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(2560, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		# get output features at affine after stats pooling
		# self.model = nn.Sequential(*list(self.model.children())[:-5])

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = self.model(x.squeeze(1))

		x = x.permute(2, 0, 1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 512)
		c0 = torch.zeros(2*2, batch_size, 512)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, (h_, c_) = self.pooling(x, (h0, c0))

		x = self.attention(out_seq.permute(1,2,0).contiguous())
		x = torch.cat([x,h_.mean(0)],dim=1).unsqueeze(-1)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class TDNN_aspp(nn.Module):

	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super().__init__()

		self.delta = delta

		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.ASPP_block = ASPP(1500, 1500)

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(1500, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		x=x.squeeze(1)

		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = self.model(x)
		x = self.ASPP_block(x)
		x = x.mean(dim=2, keepdim=True)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class TDNN_multipool(nn.Module):

	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super().__init__()

		self.delta = delta

		self.model_1 = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_2 = nn.Sequential( nn.Conv1d(512, 512, 5, padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_3 = nn.Sequential( nn.Conv1d(512, 512, 5, padding=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_4 = nn.Sequential( nn.Conv1d(512, 512, 7),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_5 = nn.Sequential( nn.Conv1d(512, 512, 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )

		self.stats_pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Linear(2048, 512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )

		self.post_pooling_2 = nn.Sequential(nn.Linear(512, 512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Linear(512, n_z) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		x_pool = []

		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = x.squeeze(1)

		x_1 = self.model_1(x)
		x_pool.append(self.stats_pooling(x_1))

		x_2 = self.model_2(x_1)
		x_pool.append(self.stats_pooling(x_2))

		x_3 = self.model_3(x_2)
		x_pool.append(self.stats_pooling(x_3))

		x_4 = self.model_4(x_3)
		x_pool.append(self.stats_pooling(x_4))

		x_5 = self.model_5(x_4)
		x_pool.append(self.stats_pooling(x_5))

		x_pool = torch.cat(x_pool, -1)

		x = self.stats_pooling(x_pool).squeeze(-1)

		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x, fc

class TDNN_mod(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super(TDNN_mod, self).__init__()
		self.delta=delta
		self.model = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = self.model(x.squeeze(1))
		x = self.pooling(x)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

class _ASPPModule(nn.Module):
	def __init__(self, inplanes, planes, kernel_size, padding, dilation):
		super(_ASPPModule, self).__init__()
		self.atrous_conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
		self.bn = nn.BatchNorm1d(planes)
		self.relu = nn.ReLU()

		self._init_weight()

	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)

		return self.relu(x)

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class ASPP(nn.Module):
	def __init__(self, inplanes, emb_dim, dilations=[1, 6, 12, 18], fmaps=48, dense=False):
		super(ASPP, self).__init__()

		if not dense:
			self.aspp1 = _ASPPModule(inplanes, fmaps, 1, padding=0, dilation=dilations[0])
			self.aspp2 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[1], dilation=dilations[1])
			self.aspp3 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[2], dilation=dilations[2])
			self.aspp4 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[3], dilation=dilations[3])

			self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d((1)),
							nn.Conv1d(inplanes, fmaps, 1, stride=1, bias=False),
							nn.BatchNorm1d(fmaps),
							nn.ReLU())

		else:
			self.aspp1 = _ASPPModule(inplanes, fmaps, dilations[0], padding=0, dilation=1)
			self.aspp2 = _ASPPModule(inplanes, fmaps, dilations[1], padding=dilations[1]//2, dilation=1)
			self.aspp3 = _ASPPModule(inplanes, fmaps, dilations[2], padding=dilations[2]//2, dilation=1)
			self.aspp4 = _ASPPModule(inplanes, fmaps, dilations[3], padding=dilations[3]//2, dilation=1)

			self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d((1)),
							nn.Conv1d(inplanes, fmaps, 1, stride=1, bias=False),
							nn.BatchNorm1d(fmaps),
							nn.ReLU())

		self.conv1 = nn.Conv1d(fmaps * 5, emb_dim, 1, bias=False)
		self.bn1 = nn.BatchNorm1d(emb_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)
		self._init_weight()

	def forward(self, x):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='linear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		return self.dropout(x)

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class transformer_enc(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none', delta=False):
		super(transformer_enc, self).__init__()
		self.delta=delta
		self.pre_encoder = nn.Sequential( nn.Conv1d(3*ncoef if delta else ncoef, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=768, dropout=0.1), num_layers=5 )

		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(1024, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):
		if self.delta:
			x=x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

		x = x.squeeze(1)
		x = self.pre_encoder(x)
		x = x.permute(2,0,1)
		x = self.transformer_encoder(x)
		x = x.permute(1,2,0)
		x = self.pooling(x)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)
