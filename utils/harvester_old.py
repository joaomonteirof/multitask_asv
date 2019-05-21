import torch
import torch.nn as nn
import numpy as np
import random
import itertools

class TripletHarvester(object):
	def __init__(self, triplet_type='all', margin=0.2, cuda_mode=False):
		super(TripletHarvester, self).__init__()
		self.triplet_type = triplet_type
		self.margin = margin
		self.cuda=cuda_mode
		self.old_nSamples = -1

	def calDistances(self, x):
		return torch.norm(x[:, None] - x, dim=2, p=2)

	def harvest_triplets(self, output, target):

		with torch.no_grad():

			output = torch.div(output, torch.norm(output, 2, 1).unsqueeze(1).expand_as(output))

			soft_triplets = []
			hard_triplets = []
			cumfreq = self.calculate_cumfreq(target)
			current = 1
			tabDists = self.calDistances(output)
			losses = []
			while current < len(cumfreq):
				for ax in range(cumfreq[current-1], cumfreq[current]-1): # iterate through all samples with current label (anchor)
					for px in range(ax+1, cumfreq[current]):   # iterate through the rest of samples with the same label (positive)
						triplets = []
						if not target[px] == target[ax]:
							continue
						max_loss = 0.0
						for nx, negative in enumerate(output): # iterate through all samples with different labels (negative)
							if target[nx] == target[ax]:
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
				soft_ran = np.random.choice(np.arange(len(soft_triplets)), size=min(N - num, len(soft_triplets)), replace=False)
				triplets = triplets + [soft_triplets[i] for i in soft_ran]
				triplet_tensor = torch.Tensor(np.array(hard_triplets)).long()

		return triplet_tensor

	def calculate_cumfreq(self, target):
		cumfreq = [0]
		s = 1
		for t in range(len(target) - 1):
			if not target[t + 1] == target[t]:
				cumfreq.append(s)
			s = s + 1

		cumfreq.append(s)
		return cumfreq

if __name__ == '__main__':

	harvester = TripletHarvester()

	embs, labels = torch.rand(40, 128), torch.from_numpy(np.random.choice(np.arange(3), replace=True, size=40)).long()

	trip = harvester.harvest_triplets(embs, labels)

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
