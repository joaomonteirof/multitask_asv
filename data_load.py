import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from librosa.feature import delta as delta_
from utils.utils import strided_app
import random

def augment_spec(example):

	with torch.no_grad():

		if random.random()>0.5:
			example = freq_mask(example, F=20, dim=1)
		if random.random()>0.5:
			example = freq_mask(example, F=50, dim=2)
		if random.random()>0.5:
			example += torch.randn_like(example)*random.choice([1e-1, 1e-2, 1e-3, 1e-4])

	return example

def freq_mask(spec, F=100, num_masks=1, replace_with_zero=False, dim=1):
	"""Frequency masking

	adapted from https://espnet.github.io/espnet/_modules/espnet/utils/spec_augment.html

	:param torch.Tensor spec: input tensor with shape (T, dim)
	:param int F: maximum width of each mask
	:param int num_masks: number of masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0,
		if False, filled with mean
	:param int dim: 1 or 2 indicating to which axis the mask corresponds
	"""

	assert dim==1 or dim==2, 'Only 1 or 2 are valid values for dim!'

	with torch.no_grad():

		cloned = spec.clone()
		num_bins = cloned.shape[dim]

		for i in range(0, num_masks):
			f = random.randrange(0, F)
			f_zero = random.randrange(0, num_bins - f)

			# avoids randrange error if values are equal and range is empty
			if f_zero == f_zero + f:
				return cloned

			mask_end = random.randrange(f_zero, f_zero + f)
			if replace_with_zero:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = 0.0
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = 0.0
			else:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = cloned.mean()
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = cloned.mean()

	return cloned

class Loader(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, delta=False):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)
		self.delta=delta

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt_1, utt_2, utt_3, utt_4, utt_5, spk, y= self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )
		utt_5_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_5] ) )

		return utt_1_data.contiguous(), utt_2_data.contiguous(), utt_3_data.contiguous(), utt_4_data.contiguous(), utt_5_data.contiguous(), y

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = augment_spec(data_)

		if self.delta:
			data_ = np.concatenate([data_, delta_(data_,width=3,order=1), delta_(data_,width=3,order=2)], axis=0)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.spk2label = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			self.spk2label[spk] = torch.LongTensor([i])

		open_file.close()

		self.n_speakers = len(self.spk2utt)

	def update_lists(self):

		self.utt_list = []

		for i, spk in enumerate(self.spk2utt):
			spk_utt_list = np.random.permutation(list(self.spk2utt[spk]))

			idxs = strided_app(np.arange(len(spk_utt_list)),5,5)

			for idxs_list in idxs:
				if len(idxs_list)==5:
					self.utt_list.append([spk_utt_list[utt_idx] for utt_idx in idxs_list])
					self.utt_list[-1].append(spk)
					self.utt_list[-1].append(self.spk2label[spk])

class Loader_valid(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, delta=False):
		super(Loader_valid, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)
		self.delta=delta

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		utt = self.utt_list[index]
		spk = self.utt2spk[utt]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = self.prep_utterance( self.open_file[spk][utt] )
		utt_data = torch.from_numpy( utt_data )

		utt_1, utt_2, utt_3, utt_4 = np.random.choice(self.spk2utt[spk], 4)

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )

		return utt_data.contiguous(), utt_1_data.contiguous(), utt_2_data.contiguous(), utt_3_data.contiguous(), utt_4_data.contiguous(), self.utt2label[utt]

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		if self.delta:
			data_ = np.concatenate([data_, delta_(data_,width=3,order=1), delta_(data_,width=3,order=2)], axis=0)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.n_speakers = len(open_file)

		self.utt2label = {}
		self.utt2spk = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			for utt in spk_utt_list:
				self.utt2label[utt] = torch.LongTensor([i])
				self.utt2spk[utt] = spk
				self.utt_list.append(utt)

		open_file.close()

class Loader_test(Dataset):

	def __init__(self, hdf5_name):
		super(Loader_test, self).__init__()
		self.hdf5_name = hdf5_name

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt_1, utt_2, utt_3, utt_4, utt_5, spk, y= self.utt_list[index]

		assert utt_1 in self.spk2utt[spk] and utt_2 in self.spk2utt[spk] and utt_3 in self.spk2utt[spk] and utt_4 in self.spk2utt[spk] and utt_5 in self.spk2utt[spk]

		utt_list_ = [utt_1, utt_2, utt_3, utt_4, utt_5]

		assert len(utt_list_) == len(set(utt_list_))

		return utt_1.contiguous(), utt_2.contiguous(), utt_3.contiguous(), utt_4.contiguous(), utt_5.contiguous(), spk, y

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		if self.delta:
			data_ = np.concatenate([data_, delta_(data_,width=3,order=1), delta_(data_,width=3,order=2)], axis=0)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.spk2label = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			self.spk2label[spk] = torch.LongTensor([i])

		open_file.close()

		self.n_speakers = len(self.spk2utt)

	def update_lists(self):

		self.utt_list = []

		print('\nNew List!!\n')

		utt_count = 0
		included_utt_count = 0

		for i, spk in enumerate(self.spk2utt):
			spk_utt_list = np.random.permutation(list(self.spk2utt[spk]))

			utt_count += len(spk_utt_list)

			idxs = strided_app(np.arange(len(spk_utt_list)),5,5)

			for idxs_list in idxs:

				if len(idxs_list)==5:
					self.utt_list.append([spk_utt_list[utt_idx] for utt_idx in idxs_list])
					included_utt_count+=len(self.utt_list[-1])
					if len(self.utt_list)>1:
						assert len(set(self.utt_list[-1]) & set(self.utt_list[-2]))==0
					self.utt_list[-1].append(spk)
					self.utt_list[-1].append(self.spk2label[spk])

		print('Total utts and included utts: {}, {}'.format(utt_count,included_utt_count))

		tot_list = [item for sublist in self.utt_list for item in sublist[:-2]]

		print(5*len(self.utt_list), len(tot_list))

if __name__=='__main__':

	import torch.utils.data
	import argparse

	def compare_spk2utts(l1, l2):
		assert len(l1)==len(l2)
		assert len(set(l1.keys()) & set(l2.keys()))==len(l1)
		count_1=0
		count_2=0
		for spk in l1:
			assert len(set(l1[spk]) & set(l2[spk]))==min(len(l1[spk]), len(l2[spk]))
			count_1+=len(l1[spk])
			count_2+=len(l2[spk])

		print(count_1, count_2)

	parser = argparse.ArgumentParser(description='Test data loader')
	parser.add_argument('--hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
	args = parser.parse_args()

	dataset = Loader_test(hdf5_name = args.hdf_file)
	loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

	loader.dataset.update_lists()

	print('Dataset length: {}, {}'.format(len(loader.dataset), len(loader.dataset.utt_list)))

	spk2utt = {}

	for batch in loader:
		utt_1, utt_2, utt_3, utt_4, utt_5, spk, y = batch

		for i in range(len(batch[-1])):
			if spk[i] in spk2utt:
				spk2utt[spk[i]]+=[utt_1[i], utt_2[i], utt_3[i], utt_4[i], utt_5[i]]
			else:
				spk2utt[spk[i]]=[utt_1[i], utt_2[i], utt_3[i], utt_4[i], utt_5[i]]

	compare_spk2utts(loader.dataset.spk2utt, spk2utt)
