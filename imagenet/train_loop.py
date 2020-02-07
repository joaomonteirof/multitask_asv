import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from data_load import Loader
import numpy as np
import os
from tqdm import tqdm

from harvester import HardestNegativeTripletSelector, AllTripletSelector

from utils import compute_eer, adjust_learning_rate, correct_topk

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, margin, lambda_, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, swap=False, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.history = {'train_loss': [], 'train_loss_batch': [], 'triplet_loss': [], 'triplet_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'EER': [], 'acc_1':[], 'acc_5':[]}
		self.total_iters = 0
		self.cur_epoch = 0
		self.lambda_ = lambda_
		self.swap = swap
		self.margin = margin
		self.harvester = HardestNegativeTripletSelector(margin=0.1, cpu=not self.cuda_mode)
		self.harvester_val = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = next(self.model.parameters()).device
		self.base_lr = self.optimizer.param_groups[0]['lr']

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while self.cur_epoch < n_epochs:

			np.random.seed()
			if isinstance(self.train_loader.dataset, Loader):
				self.train_loader.dataset.update_lists()

			adjust_learning_rate(self.optimizer, self.cur_epoch, self.base_lr)

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			ce=0.0
			triplet_loss=0.0
			train_loss=0.0

			# Train step
			for t, batch in train_iter:
				ce_batch, triplet_loss_batch = self.train_step(batch)
				ce += ce_batch
				triplet_loss += triplet_loss_batch
				train_loss += ce_batch + triplet_loss_batch
				self.history['train_loss_batch'].append(ce_batch + triplet_loss_batch)
				self.history['triplet_loss_batch'].append(triplet_loss_batch)
				self.history['ce_loss_batch'].append(ce_batch)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss/(t+1))
			self.history['triplet_loss'].append(triplet_loss/(t+1))
			self.history['ce_loss'].append(ce/(t+1))

			if self.verbose>0:
				print(' ')
				print('Total train loss, Triplet loss, and Cross-entropy: {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['train_loss'][-1], self.history['triplet_loss'][-1], self.history['ce_loss'][-1]))

			# Validation

			tot_correct_1, tot_correct_5, tot_ = 0, 0, 0
			scores, labels = None, None

			for t, batch in enumerate(self.valid_loader):

				correct_1, correct_5, total, scores_batch, labels_batch = self.valid(batch)

				try:
					scores = np.concatenate([scores, scores_batch], 0)
					labels = np.concatenate([labels, labels_batch], 0)
				except:
					scores, labels = scores_batch, labels_batch

				tot_correct_1 += correct_1
				tot_correct_5 += correct_5
				tot_ += total

			self.history['EER'].append(compute_eer(labels, scores))
			self.history['acc_1'].append(float(tot_correct_1)/tot_)
			self.history['acc_5'].append(float(tot_correct_5)/tot_)

			if self.verbose>0:
				print(' ')
				print('Current, best validation EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['EER'][-1], np.min(self.history['EER']), 1+np.argmin(self.history['EER'])))
				print('Current Top 1 Acc, best Top 1 Acc, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['acc_1'][-1], np.max(self.history['acc_1']), 1+np.argmax(self.history['acc_1'])))
				print('Current Top 5 Acc, best Top 5 Acc, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['acc_5'][-1], np.max(self.history['acc_5']), 1+np.argmax(self.history['acc_5'])))

			if self.verbose>0:
				print(' ')
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			if self.save_cp and (self.cur_epoch % save_every == 0 or self.history['EER'][-1] < np.min([np.inf]+self.history['EER'][:-1])):
				self.checkpointing()

			self.cur_epoch += 1

		if self.verbose>0:
			print('Training done!')

			if self.valid_loader is not None:
				print('Best EER and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['EER']), 1+np.argmin(self.history['EER'])))

		return np.min(self.history['EER'])

	def train_step(self, batch):

		self.model.train()

		self.optimizer.zero_grad()

		if isinstance(self.train_loader.dataset, Loader):
			x_1, x_2, x_3, x_4, x_5, y = batch
			x = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=0)
			y = torch.cat(5*[y], dim=0).squeeze().contiguous()
		else:
			x, y = batch

		if self.cuda_mode:
			x = x.to(self.device, non_blocking=True)
			y = y.to(self.device, non_blocking=True)

		embeddings = self.model.forward(x)
		embeddings_norm = F.normalize(embeddings, p=2, dim=1)

		loss_class = torch.nn.CrossEntropyLoss()(self.model.out_proj(embeddings_norm, y), y)

		triplets_idx, entropy_indices = self.harvester.get_triplets(embeddings_norm.detach(), y)

		if self.cuda_mode:
			triplets_idx = triplets_idx.to(self.device, non_blocking=True)

		emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

		loss_metric = self.triplet_loss(emb_a, emb_p, emb_n)

		loss = loss_class + loss_metric

		entropy_regularizer = torch.nn.functional.pairwise_distance(embeddings_norm, embeddings_norm[entropy_indices,:]).mean()
		loss -= entropy_regularizer*self.lambda_

		loss.backward()

		self.optimizer.step()

		return loss_class.item(), loss_metric.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			if isinstance(self.valid_loader.dataset, Loader):
				x_1, x_2, x_3, x_4, x_5, y = batch
				x = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=0)
				y = torch.cat(5*[y], dim=0).squeeze().contiguous()
			else:
				x, y = batch

			if self.cuda_mode:
				x = x.to(self.device, non_blocking=True)
				y = y.to(self.device, non_blocking=True)

			embeddings = self.model.forward(x)
			embeddings_norm = F.normalize(embeddings, p=2, dim=1)

			out = self.model.out_proj(embeddings_norm, y)

			pred = F.softmax(out, dim=1)
			(correct_1, correct_5) = correct_topk(pred, y, (1,5))

			triplets_idx = self.harvester_val.get_triplets(embeddings, y)

			embeddings = embeddings.cpu()

			emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

			scores_p = F.cosine_similarity(emb_a, emb_p)
			scores_n = F.cosine_similarity(emb_a, emb_n)

		return correct_1, correct_5, x.size(0), np.concatenate([scores_p.detach().cpu().numpy(), scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(scores_p.size(0)), np.zeros(scores_n.size(0))], 0)

	def triplet_loss(self, emba, embp, embn, reduce_=True):

		loss_ = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=self.swap, reduction='mean' if reduce_ else 'none')(emba, embp, embn)

		return loss_

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print(' ')
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'n_classes': self.model.n_classes}

		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!')

	def initialize_params(self):
		for layer in self.model.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
