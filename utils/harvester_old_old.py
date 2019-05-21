import torch
import torch.nn as nn
import numpy as np
import random
from random import randrange
import scipy.stats
from math import sqrt

class TripletHarvester(object):
	def __init__(self, triplet_type='dist', margin=0.2, reuse_dist=False, cuda_mode=False):
		super(TripletHarvester, self).__init__()
		self.reuse_dist = reuse_dist
		self.triplet_type = triplet_type
		self.margin = margin
		self.cuda=cuda_mode

	def calDistances(self, output):
		nSamples = output.size(0)
		tabDists = torch.Tensor(nSamples, nSamples).cuda(async=True) if self.cuda else torch.Tensor(nSamples, nSamples)
		for i in range(nSamples-1):
			for j in range(i + 1, nSamples):
				d = torch.dist(output[i],output[j],2)
				tabDists[i][j] = d
				tabDists[j][i] = d

		return tabDists

	def harvest_triplets(self, output, target, cap=-1.0, video_labels=None):

		output = torch.div(output, torch.norm(output, 2, 1).unsqueeze(1).expand_as(output))

		if self.triplet_type == 'dist':

			try:
				triplets_ = self.dist_sampling(output, target, cap, video_labels)

				if triplets_.size(0)==0:
					print('random')
					self.triplet_type = 'random'
				else:
					return triplets_
			except:
				print('random')
				self.triplet_type = 'random'

		soft_triplets = []
		hard_triplets = []
		cumfreq = self.calculate_cumfreq(target)
		current = 1
		tabDists = self.calDistances(output)
		losses = []
		while current < len(cumfreq):
			for ax in range(cumfreq[current-1],cumfreq[current]-1): # iterate through all samples with current label (anchor)
				for px in range(ax+1,cumfreq[current]):   # iterate through the rest of samples with the same label (positive)
					triplets = []
					if not target[px] == target[ax]:
						continue
					max_loss = 0.0
					for nx, negative in enumerate(output): # iterate through all samples with different labels (negative)
						if target[nx] == target[ax]:
							continue
						if not video_labels is None:
							if video_labels[nx] == video_labels[ax]:
								continue
 
						loss = tabDists[ax][px] - tabDists[ax][nx] + self.margin # d(a,p) - d(a, n) + alpha

						if loss <= 0:  # normal triplet, useless
							continue
						 
						if self.triplet_type == 'semihard':
							if loss > max_loss and loss < self.margin:
								t = (ax, px, nx)
								max_loss = loss
							continue

						if self.triplet_type == 'random':
							triplets.append((ax, px, nx))
							continue
						if loss < self.margin: # soft-negative triplet
							soft_triplets.append((ax, px, nx))
							continue

						# hard-negative triplet
						losses.append(loss)
						hard_triplets.append((ax, px, nx))

					if self.triplet_type == 'semihard' and max_loss > 0.0:
						hard_triplets.append(t)

					if self.triplet_type == "random" and len(triplets) > 0:
						hard_triplets.append(random.choice(triplets))

			current = current + 1

		if len(soft_triplets) + len(hard_triplets) == 0:
			triplet_tensor = torch.Tensor(0)
		else:
			if self.triplet_type == 'all':
				triplet_tensor = torch.Tensor(np.array(soft_triplets + hard_triplets)).long()

			if self.triplet_type == 'soft':
				triplet_tensor = torch.Tensor(np.array(soft_triplets)).long() 

			if self.triplet_type == 'semihard' or self.triplet_type == 'random':
				triplet_tensor = torch.Tensor(np.array(hard_triplets)).long()

			if self.triplet_type == 'hard':
				ratio = 0.1
				N = len(soft_triplets) + len(hard_triplets)
				sort_idx = [i[0] for i in sorted(enumerate(losses), key=lambda x:x[1], reverse=True)]
				num = int(ratio * N)
				hard_ran = random.sample(sort_idx, min(num, len(sort_idx)))
				triplets = [hard_triplets[i] for i in hard_ran]
				soft_ran = random.sample(np.random.permutation(len(soft_triplets)), min(N - num, len(soft_triplets)))
				triplets = triplets + [soft_triplets[i] for i in soft_ran]
				triplet_tensor = torch.Tensor(np.array(hard_triplets)).long()

		if self.reuse_dist == False:
			return triplet_tensor

		return triplet_tensor, tabDists

	def dist_sampling(self, output, target, cap=-1.0, video_labels=None):
		triplets = []
		cumfreq = self.calculate_cumfreq(target)

		current = 1

		nSamples = output.size(0)
		nDim = output.size(1)
		gaussian = scipy.stats.norm(sqrt(2), sqrt(1.0 / (2 * nDim)))
		tabWeights = torch.Tensor(nSamples, nSamples)
		tabDists = torch.Tensor(nSamples, nSamples)

		# pair-wise distances
		while current < len(cumfreq):
			for ax in range(cumfreq[current-1],cumfreq[current]-1): # iterate through all samples with current label (anchor)
				choice_list = []
				weight_list = []
				for nx, negative in enumerate(output): # iterate through all samples with different labels (negative)
					if target[nx] == target[ax]:
						continue
					if tabWeights[ax][nx] == 0:
						d = torch.dist(output[ax],output[nx], 2.0)
						if (d > 1.4): # magical cutoff value from the paper
							continue
						d = max(d, 0.4) # magical cutoff value from the paper
						tabWeights[ax][nx] = self.get_sampling_weight(gaussian, d)
						tabWeights[nx][ax] = tabWeights[ax][nx]

						tabDists[ax][nx] = d
						tabDists[nx][ax] = d

					choice_list.append(nx)
					weight_list.append(tabWeights[ax][nx])

				weight_list = np.absolute(weight_list / np.sum(weight_list)) # np.absolute to prevent very small number underflowing to negative values
				weight_list = weight_list / np.sum(weight_list)			  # renormalize to 1 again
				for px in range(ax+1,cumfreq[current]):   # iterate through the rest of samples with the same label (positive)
					if not target[px] == target[ax]:
						continue
					if cap > 0:
						d = torch.dist(output[ax],output[px], 2.0)
						if d > cap:
							continue
					nx = np.random.choice(choice_list, p=weight_list)
					triplets.append((ax, px, nx))
			current = current + 1

		if len(triplets) == 0:

			triplet_tensor = torch.Tensor(0)

		triplet_tensor = torch.Tensor(np.array(triplets)).long()

		if self.reuse_dist == False:
			return triplet_tensor
		return triplet_tensor, tabDists

	def calculate_cumfreq(self, target):
		cumfreq = [0]
		s = 1
		for t in range(len(target) - 1):
			if not target[t + 1] == target[t]:
				cumfreq.append(s)
			s = s + 1

		cumfreq.append(s)
		return cumfreq

	def get_sampling_weight(self, gaussian, d, thr=1e+15):
		w = 1.0 / (gaussian.pdf(d))
		return min(w, thr)

if __name__ == '__main__':

	harvester = TripletHarvester()

	embs, labels = torch.rand(40, 128), torch.from_numpy(np.random.choice(np.arange(8), replace=True, size=40)).long()

	trip = harvester.harvest_triplets(embs, labels.numpy())

	anchor = torch.index_select(embs, 0, trip[:, 0])
	positive = torch.index_select(embs, 0, trip[:, 1])
	negative = torch.index_select(embs, 0, trip[:, 2])

	print(anchor.size(), positive.size(), negative.size())

	'''
	from ResNet import resnet34
	model = resnet34(tp='emb')

	data = torch.rand(10,300,257)

	out = model(data)

	print(out.size())

	'''
