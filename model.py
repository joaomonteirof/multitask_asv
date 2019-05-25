import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.losses import AMSoftmax, Softmax


class cnn_lstm_mfcc(nn.Module):
	def __init__(self, n_z=256, proj_size=0, ncoef=23, sm_type='none'):
		super(cnn_lstm_mfcc, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(ncoef,3), padding=(0,2), stride=(1,1), bias=False),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,1), stride=(1,2), bias=False),
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,1), stride=(1,2), bias=False),
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 256, kernel_size=(1,5), padding=(0,1), stride=(1,2), bias=False),
			nn.BatchNorm2d(256),
			nn.ELU() )

		self.lstm = nn.LSTM(256, 512, 2, bidirectional=True, batch_first=False)

		self.fc_mu = nn.Sequential(
			nn.Linear(512*2, n_z) )

		self.initialize_params()

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		feats = self.features(x).squeeze(2)

		feats = feats.permute(2,0,1)

		batch_size = feats.size(1)
		seq_size = feats.size(0)

		h0 = torch.zeros(2*2, batch_size, 512)
		c0 = torch.zeros(2*2, batch_size, 512)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, h_c = self.lstm(feats, (h0, c0))

		out_end = out_seq.mean(0)

		mu = self.fc_mu(out_end)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu

	def initialize_params(self):
		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight)
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

class cnn_lstm_fb(nn.Module):
	def __init__(self, n_z=256, proj_size=0, sm_type='none'):
		super(cnn_lstm_fb, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,3), bias=False),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(32, 64, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,2), bias=False),
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 128, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 256, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
			nn.BatchNorm2d(256),
			nn.ELU() )

		self.lstm = nn.LSTM(256, 512, 2, bidirectional=True, batch_first=False)

		self.fc_mu = nn.Sequential(
			nn.Linear(512*2, n_z) )

		self.initialize_params()

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		feats = self.features(x).squeeze(2)

		feats = feats.permute(2,0,1)

		batch_size = feats.size(1)
		seq_size = feats.size(0)

		h0 = torch.zeros(2*2, batch_size, 512)
		c0 = torch.zeros(2*2, batch_size, 512)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, h_c = self.lstm(feats, (h0, c0))

		out_end = out_seq.mean(0)

		mu = self.fc_mu(out_end)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu

	def initialize_params(self):
		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight)
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

class SelfAttention(nn.Module):
	def __init__(self, hidden_size):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

		init.kaiming_uniform_(self.att_weights)

	def forward(self, inputs):

		batch_size = inputs.size(0)
		weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

		if inputs.size(0)==1:
			attentions = F.softmax(torch.tanh(weights), dim=1)
			weighted = torch.mul(inputs, attentions.expand_as(inputs))
		else:
			attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
			weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		noise = 1e-5*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.cuda(inputs.get_device())

		avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		return representations

class StructuredSelfAttention(nn.Module):
	def __init__(self, hidden_size, intermediate_size=256, attention_heads=64):
		super(StructuredSelfAttention, self).__init__()

		self.S1 = nn.Parameter(torch.Tensor(intermediate_size, hidden_size), requires_grad=True)
		self.S2 = nn.Parameter(torch.Tensor(attention_heads, intermediate_size), requires_grad=True)

		self.lin_proj = nn.Linear(hidden_size*2, hidden_size, bias=False)

		self.init_weights()

	def init_weights(self):

		initrange = 0.05
		init.uniform_(self.S1, -initrange, initrange )
		init.uniform_(self.S2, -initrange, initrange )
		init.kaiming_normal_(self.lin_proj.weight)

	def forward(self, inputs, regularize=False):

		batch_size = inputs.size(0)

		d = F.elu( torch.bmm(self.S1.unsqueeze(0).repeat(batch_size, 1, 1), inputs.permute(0, 2, 1) ) )

		attentions = F.softmax( torch.bmm(self.S2.unsqueeze(0).repeat(batch_size, 1, 1), d), dim=2 )

		weighted = torch.bmm(attentions, inputs)

		noise = 1e-4*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.cuda(inputs.get_device())

		avg_repr, std_repr = weighted.mean(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		if regularize:

			#Reg. term

			AAT = torch.bmm(attentions, attentions.transpose(-1,1))
			I = torch.eye(AAT.size(-1)).unsqueeze(0).repeat(batch_size, 1, 1)

			if AAT.is_cuda:
				I=I.cuda(AAT.get_device())

			reg = torch.norm(AAT - I).pow(2)

			return torch.tanh( self.lin_proj(representations) ), reg

		else:
			return torch.tanh( self.lin_proj(representations) )

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.activation = nn.ELU()
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.activation(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.activation = nn.ELU()
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.activation(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.activation(out)

		return out

class ResNet_fb(nn.Module):
	def __init__(self, n_z=256, layers=[2,2,2,2], block=Bottleneck, proj_size=0, sm_type='none'):
		self.inplanes = 16
		super(ResNet_fb, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,3), stride=(2,1), padding=(1,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.conv5 = nn.Conv2d(512, 512, kernel_size=(5,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn5 = nn.BatchNorm2d(512)

		self.fc = nn.Linear(2*512,512)
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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.conv5(x)
		x = self.activation(self.bn5(x)).squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))
		mu = self.fc_mu(fc)
		return mu

class ResNet_mfcc(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 32
		super(ResNet_mfcc, self).__init__()

		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.fc = nn.Linear(2*512,512)
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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu

	def embedd(self, x, out_index=-1):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x).squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		out=[stats]

		x = self.fc(stats)

		out.append(x)

		x = self.lbn(x)

		out.append(x)

		fc = F.elu(x)

		out.append(fc)

		mu = self.fc_mu(fc)

		out.append(mu)

		return out[out_index]

class ResNet_lstm(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 32
		super(ResNet_lstm, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.lstm = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=False)

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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2,0,1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 256)
		c0 = torch.zeros(2*2, batch_size, 256)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, (h_, c_) = self.lstm(x, (h0, c0))

		stats = self.attention(out_seq.permute(1,0,2).contiguous())

		x = torch.cat([stats,h_.mean(0)],dim=1)

		fc = F.elu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)
		return mu

class ResNet_large_lstm(nn.Module):
	def __init__(self, n_z=256, layers=[3,8,36,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 32
		super(ResNet_large_lstm, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.lstm = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=False)

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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2,0,1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 256)
		c0 = torch.zeros(2*2, batch_size, 256)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		out_seq, (h_, c_) = self.lstm(x, (h0, c0))

		stats = self.attention(out_seq.permute(1,0,2).contiguous())

		x = torch.cat([stats,h_.mean(0)],dim=1)

		fc = F.elu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)
		return mu

class ResNet_stats(nn.Module):
	def __init__(self, n_z=256, layers=[3,4,6,3], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 16
		super(ResNet_stats, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.fc = nn.Linear(2*512,512)
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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		x = torch.cat([x.mean(-1), x.std(-1)], dim=1)

		fc = F.elu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu

class ResNet_small(nn.Module):
	def __init__(self, n_z=256, layers=[2,2,2,2], block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.inplanes = 16
		super(ResNet_small, self).__init__()

		self.conv1 = nn.Conv2d(1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ELU()
		
		self.layer1 = self._make_layer(block, 16, layers[0],stride=1)
		self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

		self.fc = nn.Linear(2*512,512)
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

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion) )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu

def inception_v3():
	"""Inception v3 model architecture from
	`"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		if 'transform_input' not in kwargs:
			kwargs['transform_input'] = True
		model = Inception3(**kwargs)
		model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
		return model

	return Inception3(**kwargs)


class inception_v3(nn.Module):

	"""Inception v3 model architecture from
	Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>
	Implementation adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
	"""

	def __init__(self, n_z=256, block=Bottleneck, proj_size=0, ncoef=23, sm_type='none'):
		super(inception_v3, self).__init__()
		self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=(ncoef,3), stride=2)
		self.Conv1d_2a_3x3 = BasicConv1d(32, 32, kernel_size=3)
		self.Conv1d_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
		self.Conv1d_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
		self.Conv1d_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)
		self.Mixed_5b = InceptionA(192, pool_features=32)
		self.Mixed_5c = InceptionA(256, pool_features=64)
		self.Mixed_5d = InceptionA(288, pool_features=64)
		self.Mixed_6a = InceptionB(288)
		self.Mixed_6b = InceptionC(768, channels_7x7=128)
		self.Mixed_6c = InceptionC(768, channels_7x7=160)
		self.Mixed_6d = InceptionC(768, channels_7x7=160)
		self.Mixed_6e = InceptionC(768, channels_7x7=192)

		self.Mixed_7a = InceptionD(768)
		self.Mixed_7b = InceptionE(1280)
		self.Mixed_7c = InceptionE(2048)

		self.fc = nn.Linear(2*2048,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
				import scipy.stats as stats
				stddev = m.stddev if hasattr(m, 'stddev') else 0.1
				X = stats.truncnorm(-2, 2, scale=stddev)
				values = torch.Tensor(X.rvs(m.weight.numel()))
				values = values.view(m.weight.size())
				m.weight.data.copy_(values)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.attention = SelfAttention(2048)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		x = self.Conv2d_1a_3x3(x)
		x = x.squeeze(2)
		x = self.Conv1d_2a_3x3(x)
		x = self.Conv1d_2b_3x3(x)
		x = F.max_pool1d(x, kernel_size=3, stride=1) ## Originally stride=2
		x = self.Conv1d_3b_1x1(x)
		x = self.Conv1d_4a_3x3(x)
		x = F.max_pool1d(x, kernel_size=3, stride=1) ## Originally stride=2
		x = self.Mixed_5b(x)
		x = self.Mixed_5c(x)
		x = self.Mixed_5d(x)
		x = self.Mixed_6a(x)
		x = self.Mixed_6b(x)
		x = self.Mixed_6c(x)
		x = self.Mixed_6d(x)
		x = self.Mixed_6e(x)
		x = self.Mixed_7a(x)
		x = self.Mixed_7b(x)
		x = self.Mixed_7c(x)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))
		mu = self.fc_mu(fc)
		return mu

	def embedd(self, x, out_index=-1):

		x = self.Conv2d_1a_3x3(x)
		x = x.squeeze(2)
		x = self.Conv1d_2a_3x3(x)
		x = self.Conv1d_2b_3x3(x)
		x = F.max_pool1d(x, kernel_size=3, stride=1) ## Originally stride=2
		x = self.Conv1d_3b_1x1(x)
		x = self.Conv1d_4a_3x3(x)
		x = F.max_pool1d(x, kernel_size=3, stride=1) ## Originally stride=2
		x = self.Mixed_5b(x)
		x = self.Mixed_5c(x)
		x = self.Mixed_5d(x)
		x = self.Mixed_6a(x)
		x = self.Mixed_6b(x)
		x = self.Mixed_6c(x)
		x = self.Mixed_6d(x)
		x = self.Mixed_6e(x)
		x = self.Mixed_7a(x)
		x = self.Mixed_7b(x)
		x = self.Mixed_7c(x)

		stats = self.attention(x.permute(0,2,1).contiguous())

		out=[stats]

		fc = F.elu(self.lbn(self.fc(stats)))

		out.append(fc)

		mu = self.fc_mu(fc)

		out.append(mu)

		return out[out_index]

class InceptionA(nn.Module):

	def __init__(self, in_channels, pool_features):
		super(InceptionA, self).__init__()
		self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)

		self.branch5x5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
		self.branch5x5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)

		self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)

		self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionB(nn.Module):

	def __init__(self, in_channels):
		super(InceptionB, self).__init__()
		self.branch3x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)

		self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3(x)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

		outputs = [branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionC(nn.Module):

	def __init__(self, in_channels, channels_7x7):
		super(InceptionC, self).__init__()
		self.branch1x1 = BasicConv1d(in_channels, 192, kernel_size=1)

		c7 = channels_7x7
		self.branch7x7_1 = BasicConv1d(in_channels, c7, kernel_size=1)
		self.branch7x7_2 = BasicConv1d(c7, c7, kernel_size=7, padding=3)
		self.branch7x7_3 = BasicConv1d(c7, 192, kernel_size=1, padding=0)

		self.branch7x7dbl_1 = BasicConv1d(in_channels, c7, kernel_size=1)
		self.branch7x7dbl_2 = BasicConv1d(c7, c7, kernel_size=1, padding=0)
		self.branch7x7dbl_3 = BasicConv1d(c7, c7, kernel_size=7, padding=3)
		self.branch7x7dbl_4 = BasicConv1d(c7, c7, kernel_size=1, padding=0)
		self.branch7x7dbl_5 = BasicConv1d(c7, 192, kernel_size=7, padding=3)

		self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionD(nn.Module):

	def __init__(self, in_channels):
		super(InceptionD, self).__init__()
		self.branch3x3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
		self.branch3x3_2 = BasicConv1d(192, 320, kernel_size=3, stride=2)

		self.branch7x7x3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
		self.branch7x7x3_2 = BasicConv1d(192, 192, kernel_size=7, padding=3)
		self.branch7x7x3_3 = BasicConv1d(192, 192, kernel_size=1, padding=0)
		self.branch7x7x3_4 = BasicConv1d(192, 192, kernel_size=3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch7x7x3 = self.branch7x7x3_1(x)
		branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

		branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
		outputs = [branch3x3, branch7x7x3, branch_pool]
		return torch.cat(outputs, 1)


class InceptionE(nn.Module):

	def __init__(self, in_channels):
		super(InceptionE, self).__init__()
		self.branch1x1 = BasicConv1d(in_channels, 320, kernel_size=1)

		self.branch3x3_1 = BasicConv1d(in_channels, 384, kernel_size=1)
		self.branch3x3_2a = BasicConv1d(384, 384, kernel_size=3, padding=1)
		self.branch3x3_2b = BasicConv1d(384, 384, kernel_size=1, padding=0)

		self.branch3x3dbl_1 = BasicConv1d(in_channels, 448, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv1d(448, 384, kernel_size=3, padding=1)
		self.branch3x3dbl_3a = BasicConv1d(384, 384, kernel_size=3, padding=1)
		self.branch3x3dbl_3b = BasicConv1d(384, 384, kernel_size=1, padding=0)

		self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

	def __init__(self, in_channels, num_classes):
		super(InceptionAux, self).__init__()
		self.conv0 = BasicConv1d(in_channels, 128, kernel_size=1)
		self.conv1 = BasicConv1d(128, 768, kernel_size=5)
		self.conv1.stddev = 0.01
		self.fc = nn.Linear(768, num_classes)
		self.fc.stddev = 0.001

	def forward(self, x):
		# 17 x 17 x 768
		x = F.avg_pool1d(x, kernel_size=5, stride=3)
		# 5 x 5 x 768
		x = self.conv0(x)
		# 5 x 5 x 128
		x = self.conv1(x)
		# 1 x 1 x 768
		x = x.view(x.size(0), -1)
		# 768
		x = self.fc(x)
		# 1000
		return x


class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.elu(x, inplace=True)

class BasicConv1d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv1d, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.elu(x, inplace=True)
