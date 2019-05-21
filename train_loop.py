import torch
import torch.nn.functional as F

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

from utils.harvester import HardestNegativeTripletSelector

from sklearn import metrics

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	return eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, margin, lambda_, patience, verbose=-1, device=0, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, swap=False, softmax=False, pretrain=False, mining=False, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.softmax = softmax!='none'
		self.pretrain = pretrain
		self.mining = mining
		self.model = model
		self.swap = swap
		self.lambda_ = lambda_
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.margin = margin
		self.harvester = HardestNegativeTripletSelector(margin=self.margin, cpu=not self.cuda_mode)
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = device
		self.history = {'train_loss': [], 'train_loss_batch': []}

		if self.valid_loader is not None:
			self.history['valid_loss'] = []
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True if self.verbose>0 else False, threshold=1e-4, min_lr=1e-7)
		else:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 100, 200, 300, 400], gamma=0.1)

		if self.softmax:
			self.history['softmax_batch']=[]
			self.history['softmax']=[]

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			train_loss_epoch=0.0

			if self.softmax:

				ce_epoch=0.0
				for t, batch in train_iter:
					train_loss, ce = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					self.history['softmax_batch'].append(ce)
					train_loss_epoch+=train_loss
					ce_epoch+=ce
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))
				self.history['softmax'].append(ce_epoch/(t+1))

				if self.verbose>0:
					print('Total train loss, Triplet loss, and Cross-entropy: {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['train_loss'][-1], (self.history['train_loss'][-1]-self.history['softmax'][-1]), self.history['softmax'][-1]))

			elif self.pretrain:

				ce_epoch=0.0
				for t, batch in train_iter:
					ce = self.pretrain_step(batch)
					self.history['train_loss_batch'].append(ce)
					ce_epoch+=ce
					self.total_iters += 1

				self.history['train_loss'].append(ce_epoch/(t+1))

				if self.verbose>0:
					print('Train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			else:

				for t, batch in train_iter:
					train_loss = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					train_loss_epoch+=train_loss
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))

				if self.verbose>0:
					print('Total train loss, {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				scores, labels = None, None

				for t, batch in enumerate(self.valid_loader):
					scores_batch, labels_batch = self.valid(batch)

					try:
						scores = np.concatenate([scores, scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores, labels = scores_batch, labels_batch

				self.history['valid_loss'].append(compute_eer(labels, scores))
				if self.verbose>0:
					print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

				self.scheduler.step(self.history['valid_loss'][-1])

			else:
				self.scheduler.step()

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.valid_loader is not None and self.save_cp and (self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1])):
					self.checkpointing()
			elif self.save_cp and self.cur_epoch % save_every == 0:
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			return np.min(self.history['valid_loss'])
		else:
			return np.min(self.history['train_loss'])

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		if self.mining:
			utterances, y = batch
			utterances.resize_(utterances.size(0)*utterances.size(1), utterances.size(2), utterances.size(3), utterances.size(4))
			y.resize_(y.numel())
		elif self.softmax:
			utt_a, utt_p, utt_n, y = batch
		else:
			utt_a, utt_p, utt_n = batch

		entropy_indices = None

		if self.mining:

			ridx = np.random.randint(utterances.size(3)//4, utterances.size(3))
			utterances = utterances[:,:,:,:ridx]
			if self.cuda_mode:
				utterances = utterances.cuda(self.device)

			embeddings = self.model.forward(utterances)

			embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

			triplets_idx, entropy_indices = self.harvester.get_triplets(embeddings_norm.detach(), y)

			if self.cuda_mode:
				triplets_idx = triplets_idx.cuda(self.device)

			emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

		else:
			ridx = np.random.randint(utt_a.size(3)//4, utt_a.size(3))
			utt_a, utt_p, utt_n = utt_a[:,:,:,:ridx], utt_p[:,:,:,:ridx], utt_n[:,:,:,:ridx]

			if self.cuda_mode:
				utt_a, utt_p, utt_n = utt_a.cuda(self.device), utt_p.cuda(self.device), utt_n.cuda(self.device)

			emb_a, emb_p, emb_n = self.model.forward(utt_a), self.model.forward(utt_p), self.model.forward(utt_n)

			emb_a = torch.div(emb_a, torch.norm(emb_a, 2, 1).unsqueeze(1).expand_as(emb_a))
			emb_p = torch.div(emb_p, torch.norm(emb_p, 2, 1).unsqueeze(1).expand_as(emb_p))
			emb_n = torch.div(emb_n, torch.norm(emb_n, 2, 1).unsqueeze(1).expand_as(emb_n))

		loss = self.triplet_loss(emb_a, emb_p, emb_n)

		loss_log = loss.item()

		if entropy_indices is not None:
			entropy_regularizer = torch.nn.functional.pairwise_distance(embeddings_norm, embeddings_norm[entropy_indices,:]).mean()
			loss -= entropy_regularizer*self.lambda_

		if self.softmax:
			if self.cuda_mode:
				y = y.cuda(self.device).squeeze()

			ce = F.cross_entropy(self.model.out_proj(embeddings, y), y) if self.mining else F.cross_entropy(self.model.out_proj(emb_a, y), y)
			loss += ce
			loss.backward()
			self.optimizer.step()
			return loss_log+ce.item(), ce.item()
		else:
			loss.backward()
			self.optimizer.step()
			return loss_log

	def pretrain_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		utt, y = batch

		ridx = np.random.randint(utt.size(3)//2, utt.size(3))
		utt = utt[:,:,:,:ridx]

		if self.cuda_mode:
			utt, y = utt.cuda(self.device), y.cuda(self.device)

		embeddings = self.model.forward(utt)

		loss = F.cross_entropy(self.model.out_proj(embeddings), y.squeeze())

		loss.backward()
		self.optimizer.step()
		return loss.item()


	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			xa, xp, xn = batch

			ridx = np.random.randint(xa.size(3)//2, xa.size(3))

			xa, xp, xn = xa[:,:,:,:ridx], xp[:,:,:,:ridx], xn[:,:,:,:ridx]

			if self.cuda_mode:
				xa = xa.contiguous().cuda(self.device)
				xp = xp.contiguous().cuda(self.device)
				xn = xn.contiguous().cuda(self.device)

			emb_a = self.model.forward(xa)
			emb_p = self.model.forward(xp)
			emb_n = self.model.forward(xn)

			scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

		return np.concatenate([scores_p.detach().cpu().numpy(), scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(scores_p.size(0)), np.zeros(scores_n.size(0))], 0)

	def triplet_loss(self, emba, embp, embn, reduce_=True):

		loss_ = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=self.swap, reduction='mean' if reduce_ else 'none')(emba, embp, embn)

		return loss_

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.scheduler.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))
