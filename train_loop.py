import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import os

from utils.harvester import HardestNegativeTripletSelector, AllTripletSelector
from utils.losses import LabelSmoothingLoss
from utils.utils import compute_eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, margin, lambda_, label_smoothing, warmup_its, max_gnorm=10.0, verbose=-1, device=0, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, swap=False, lr_red_epoch=100, lr_factor=0.1, softmax=False, pretrain=False, mining=False, cuda=True, logger=None):
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
		self.lr_red_epoch = lr_red_epoch
		self.lr_factor = lr_factor
		self.max_gnorm = max_gnorm
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.margin = margin
		self.harvester_mine = HardestNegativeTripletSelector(margin=self.margin, cpu=not self.cuda_mode)
		self.harvester_all = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = device
		self.history = {'train_loss': [], 'train_loss_batch': []}
		self.logger = logger
		self.base_lr = self.optimizer.optimizer.param_groups[0]['lr']

		its_per_epoch = len(train_loader.dataset)//(train_loader.batch_size) + 1 if len(train_loader.dataset)%(train_loader.batch_size)>0 else len(train_loader.dataset)//(train_loader.batch_size)

		if self.softmax:
			if label_smoothing>0.0:
				self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=train_loader.dataset.n_speakers)
			else:
				self.ce_criterion = torch.nn.CrossEntropyLoss()

		if self.valid_loader is not None:
			self.history['valid_loss_emb'] = []
			self.history['valid_loss_out'] = []
			self.history['valid_loss_fus'] = []

		if self.softmax:
			self.history['softmax_batch']=[]
			self.history['softmax']=[]

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()
			self.train_loader.dataset.update_lists()
			self.update_lr()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				print('Number of training examples given new list: {}'.format(len(self.train_loader.dataset)))
				train_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			train_loss_epoch=0.0

			if self.softmax and not self.pretrain:

				ce_epoch=0.0
				for t, batch in train_iter:
					train_loss, ce = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					self.history['softmax_batch'].append(ce)
					train_loss_epoch+=train_loss
					ce_epoch+=ce
					if self.logger:
						self.logger.add_scalar('Train/Train Loss', train_loss, self.total_iters)
						self.logger.add_scalar('Train/Triplet Loss', train_loss-ce, self.total_iters)
						self.logger.add_scalar('Train/Cross enropy', ce, self.total_iters)
						self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)
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
					if self.logger:
						self.logger.add_scalar('Cross enropy', ce, self.total_iters)
						self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)
					self.total_iters += 1

				self.history['train_loss'].append(ce_epoch/(t+1))

				if self.verbose>0:
					print('Train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			else:
				for t, batch in train_iter:
					train_loss = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					train_loss_epoch+=train_loss
					if self.logger:
						self.logger.add_scalar('Train/Train Loss', train_loss, self.total_iters)
						self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))

				if self.verbose>0:
					print('Total train loss, {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				emb_scores, out_scores, labels, emb, y_ = None, None, None, None, None

				for t, batch in enumerate(self.valid_loader):
					emb_scores_batch, out_scores_batch, labels_batch, emb_batch, y_batch = self.valid(batch)

					try:
						emb_scores = np.concatenate([emb_scores, emb_scores_batch], 0)
						out_scores = np.concatenate([out_scores, out_scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
						emb = np.concatenate([emb, emb_batch], 0)
						y_ = np.concatenate([y_, y_batch], 0)
					except:
						emb_scores, out_scores, labels, emb, y_ = emb_scores_batch, out_scores_batch, labels_batch, emb_batch, y_batch

				fus_scores = (emb_scores + out_scores)*0.5

				self.history['valid_loss_emb'].append(compute_eer(labels, emb_scores))
				self.history['valid_loss_out'].append(compute_eer(labels, out_scores))
				self.history['valid_loss_fus'].append(compute_eer(labels, fus_scores))
				if self.verbose>0:
					print('Current embedding-level validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss_emb'][-1], np.min(self.history['valid_loss_emb']), 1+np.argmin(self.history['valid_loss_emb'])))
					print('Current output-level validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss_out'][-1], np.min(self.history['valid_loss_out']), 1+np.argmin(self.history['valid_loss_out'])))
					print('Current fused validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss_fus'][-1], np.min(self.history['valid_loss_fus']), 1+np.argmin(self.history['valid_loss_fus'])))
				if self.logger:
					self.logger.add_scalar('Valid/E-EER', self.history['valid_loss_emb'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best E-EER', np.min(self.history['valid_loss_emb']), self.total_iters-1)
					self.logger.add_pr_curve('Valid. E-ROC', labels=labels, predictions=emb_scores, global_step=self.total_iters-1)
					self.logger.add_scalar('Valid/O-EER', self.history['valid_loss_out'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best O-EER', np.min(self.history['valid_loss_out']), self.total_iters-1)
					self.logger.add_pr_curve('Valid. O-ROC', labels=labels, predictions=out_scores, global_step=self.total_iters-1)
					self.logger.add_scalar('Valid/F-EER', self.history['valid_loss_fus'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best F-EER', np.min(self.history['valid_loss_fus']), self.total_iters-1)
					self.logger.add_pr_curve('Valid. F-ROC', labels=labels, predictions=fus_scores, global_step=self.total_iters-1)

					if emb.shape[0]>20000:
						idxs = np.random.choice(np.arange(emb.shape[0]), size=20000, replace=False)
						emb, y_ = emb[idxs, :], y_[idxs]

					self.logger.add_histogram('Valid/Embeddings', values=emb, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/E-Scores', values=emb_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/O-Scores', values=out_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/F-Scores', values=fus_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/Labels', values=labels, global_step=self.total_iters-1)

					if self.verbose>1:
						self.logger.add_embedding(mat=emb, metadata=list(y_), global_step=self.total_iters-1)

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.valid_loader is not None and self.save_cp and (self.cur_epoch % save_every == 0 or self.history['valid_loss_emb'][-1] < np.min([np.inf]+self.history['valid_loss_emb'][:-1])):
					self.checkpointing()
			elif self.save_cp and self.cur_epoch % save_every == 0:
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best embedding-level validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss_emb']), 1+np.argmin(self.history['valid_loss_emb'])))
				print('Best output-level validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss_out']), 1+np.argmin(self.history['valid_loss_out'])))
				print('Best fused validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss_fus']), 1+np.argmin(self.history['valid_loss_fus'])))

			return np.min(self.history['valid_loss_emb'])
		else:
			return np.min(self.history['train_loss'])

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		utt_1, utt_2, utt_3, utt_4, utt_5, y = batch
		utterances = torch.cat([utt_1, utt_2, utt_3, utt_4, utt_5], dim=0)
		y = torch.cat(5*[y], dim=0).squeeze().contiguous()

		entropy_indices = None

		ridx = np.random.randint(utterances.size(3)//4, utterances.size(3))
		utterances = utterances[:,:,:,:ridx].contiguous()

		if self.cuda_mode:
			utterances = utterances.to(self.device, non_blocking=True)
			y = y.to(self.device, non_blocking=True)

		out, embeddings = self.model.forward(utterances)
		embeddings_norm = F.normalize(embeddings, p=2, dim=1)
		out_norm = F.normalize(out, p=2, dim=1)

		if self.mining:
			triplets_idx, entropy_indices = self.harvester_mine.get_triplets(embeddings_norm.detach(), y)
		else:
			triplets_idx = self.harvester_all.get_triplets(embeddings_norm.detach(), y)

		if self.cuda_mode:
			triplets_idx = triplets_idx.to(self.device, non_blocking=True)

		emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

		loss = self.triplet_loss(emb_a, emb_p, emb_n)

		loss_log = loss.item()

		if entropy_indices is not None:
			entropy_regularizer = torch.log(torch.nn.functional.pairwise_distance(embeddings_norm, embeddings_norm[entropy_indices,:])+1e-6).mean()
			loss -= entropy_regularizer*self.lambda_
			if self.logger:
				self.logger.add_scalar('Train/Entropy reg.', entropy_regularizer.item(), self.total_iters)

		if self.softmax:
			ce = self.ce_criterion(self.model.out_proj(out_norm, y), y)
			loss += ce
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
			self.optimizer.step()

			if self.logger:
				self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

			return loss_log+ce.item(), ce.item()
		else:
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
			self.optimizer.step()

			if self.logger:
				self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

			return loss_log

	def pretrain_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		utt_1, utt_2, utt_3, utt_4, utt_5, y = batch
		utterances = torch.cat([utt_1, utt_2, utt_3, utt_4, utt_5], dim=0)
		y = torch.cat(5*[y], dim=0).squeeze().contiguous()

		ridx = np.random.randint(utterances.size(3)//4, utterances.size(3))
		utterances = utterances[:,:,:,:ridx].contiguous()

		if self.cuda_mode:
			utterances = utterances.to(self.device, non_blocking=True)
			y = y.to(self.device, non_blocking=True)

		out, embeddings = self.model.forward(utterances)
		out_norm = F.normalize(out, p=2, dim=1)

		loss = self.ce_criterion(self.model.out_proj(out_norm, y), y)

		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

		return loss.item()


	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			utt_1, utt_2, utt_3, utt_4, utt_5, y = batch
			utterances = torch.cat([utt_1, utt_2, utt_3, utt_4, utt_5], dim=0)
			y = torch.cat(5*[y], dim=0).squeeze().contiguous()

			ridx = np.random.randint(utterances.size(3)//4, utterances.size(3))
			utterances = utterances[:,:,:,:ridx].contiguous()

			if self.cuda_mode:
				utterances = utterances.to(self.device, non_blocking=True)
				y = y.to(self.device, non_blocking=True)

			out, embeddings = self.model.forward(utterances)

			out_norm = F.normalize(out, p=2, dim=1)
			embeddings_norm = F.normalize(embeddings, p=2, dim=1)

			emb_triplets_idx = self.harvester_all.get_triplets(embeddings_norm.detach(), y)
			out_triplets_idx = self.harvester_all.get_triplets(out_norm.detach(), y)

			if self.cuda_mode:
				emb_triplets_idx = emb_triplets_idx.cuda(self.device)
				out_triplets_idx = out_triplets_idx.cuda(self.device)

			emb_a = torch.index_select(embeddings_norm, 0, emb_triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings_norm, 0, emb_triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings_norm, 0, emb_triplets_idx[:, 2])

			out_a = torch.index_select(out_norm, 0, out_triplets_idx[:, 0])
			out_p = torch.index_select(out_norm, 0, out_triplets_idx[:, 1])
			out_n = torch.index_select(out_norm, 0, out_triplets_idx[:, 2])

			emb_scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			emb_scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

			out_scores_p = torch.nn.functional.cosine_similarity(out_a, out_p)
			out_scores_n = torch.nn.functional.cosine_similarity(out_a, out_n)

		return (np.concatenate([emb_scores_p.detach().cpu().numpy(), emb_scores_n.detach().cpu().numpy()], 0), 
			np.concatenate([out_scores_p.detach().cpu().numpy(), out_scores_n.detach().cpu().numpy()], 0), 
			np.concatenate([np.ones(emb_scores_p.size(0)), np.zeros(emb_scores_n.size(0))], 0), 
			embeddings.detach().cpu().numpy(), y.detach().cpu().numpy() )

	def triplet_loss(self, emba, embp, embn, reduce_=True):

		loss_ = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=self.swap, reduction='mean' if reduce_ else 'none')(emba, embp, embn)

		return loss_

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.optimizer.state_dict(),
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
			self.optimizer.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def update_lr(self):
		pos = 0  ## Corresponds to the position where self.cur_epoch would be inserted in self.lr_red_epoch
		if self.lr_red_epoch is None:
			pass
		elif len(self.lr_red_epoch)==1:
			if self.cur_epoch<self.lr_red_epoch[-1]:
				pass
			else:
				self.optimizer.init_lr = self.base_lr*self.lr_factor
		else:
			i = len(self.lr_red_epoch) - 1
			while i >= 0 and self.lr_red_epoch[i] > self.cur_epoch:
				i -= 1
			self.optimizer.init_lr = self.base_lr*(self.lr_factor**(i+1))

