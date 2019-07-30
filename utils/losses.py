import math
import torch
from torch import nn
from scipy.special import binom
import torch.nn.functional as F

class AMSoftmax(nn.Module):

	## adapted from https://github.com/Joker316701882/Additive-Margin-Softmax/blob/master/AM_softmax.py

	def __init__(self, input_features, output_features, m=0.35, s=30.0):
		super().__init__()
		self.input_dim = input_features  # number of input features
		self.output_dim = output_features  # number of classes
		self.s = s
		self.m = m

		# Initialize parameters
		self.w = nn.Parameter(torch.FloatTensor(input_features, output_features))

		self.init_parameters()

	def init_parameters(self):
		nn.init.kaiming_normal_(self.w)

	def forward(self, embeddings, target):

		self.w.to(embeddings.device)

		w_norm = F.normalize(self.w, p=2, dim=0)

		cos_theta = embeddings.mm(w_norm)
		cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

		phi_theta = cos_theta - self.m

		target_onehot = torch.zeros(embeddings.size(0), w_norm.size(1)).to(embeddings.device)
		target_onehot.scatter_(1, target.view(-1,1), 1)

		logits = self.s*torch.where(target_onehot==1, phi_theta, cos_theta)

		return logits

class Softmax(nn.Module):

	def __init__(self, input_features, output_features):
		super().__init__()

		self.w = nn.Linear(input_features, output_features)

		self.initialize_params()

	def initialize_params(self):

		for layer in self.modules():

			if isinstance(layer, nn.Linear):
				nn.init.kaiming_normal_(layer.weight)
				layer.bias.data.zero_()

	def forward(self, embeddings, *args):
		self.w.to(embeddings.device)
		return self.w(embeddings)

class LabelSmoothingLoss(nn.Module):
	"""
	Adapted from https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186-L213
	With label smoothing,
	KL-divergence between q_{smoothed ground truth prob.}(w)
	and p_{prob. computed by model}(w) is minimized.
	"""
	def __init__(self, label_smoothing, lbl_set_size, ignore_index=-100):
		assert 0.0 < label_smoothing <= 1.0
		self.ignore_index = ignore_index
		super(LabelSmoothingLoss, self).__init__()

		smoothing_value = label_smoothing / (lbl_set_size - 2)
		one_hot = torch.full((lbl_set_size,), smoothing_value)
		one_hot[self.ignore_index] = 0
		self.register_buffer('one_hot', one_hot.unsqueeze(0))

		self.confidence = 1.0 - label_smoothing

	def forward(self, output, target):
		"""
		output (FloatTensor): batch_size x n_classes
		target (LongTensor): batch_size
		"""

		output = F.softmax(output, dim=1)

		model_prob = self.one_hot.repeat(target.size(0), 1)
		model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
		model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

		return F.kl_div(output, model_prob, reduction='sum')
