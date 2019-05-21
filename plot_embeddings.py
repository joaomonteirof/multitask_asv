from __future__ import print_function
import argparse
import torch
import torchvision
import torch.utils.data
from PIL import ImageFilter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import model as model_
import numpy as np
from sklearn import preprocessing, manifold
import h5py
import os
import sys

rcParams.update({'figure.autolayout': True})

def get_freer_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	return torch.device('cuda:'+str(np.argmax(memory_available)))

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

def compute_embeddings(model, spk2utt_, data_loader, n_speakers, cuda_mode, device):

	model.eval()

	try:
		spk_list_ascii = data_loader['spk_list']
		speakers_list = [spk.decode('utf-8') for spk in spk_list_ascii]
	except:
		speakers_list = list(spk2utt_.keys())

	n_speakers = n_speakers if n_speakers<len(speakers_list) else len(speakers_list)

	speakers_idx = np.random.choice(np.arange(n_speakers), n_speakers, replace=False)

	emb = None
	spk = None

	for spk_idx in speakers_idx:

		spk_id = speakers_list[spk_idx]

		utt_list = spk2utt_[spk_id]

		n_samples_spk = len(utt_list)

		speakers = np.asarray([spk_id]*n_samples_spk)

		if spk is not None:
			spk = np.concatenate( [spk, speakers], axis=0 )
		else:
			spk = speakers

		for utt_ in utt_list:

			data = np.expand_dims(data_loader[utt_], 0)

			feats = torch.from_numpy(data).float()

			if cuda_mode:
				feats = feats.cuda(device)

			new_emb = model.forward(feats)

			new_emb = new_emb.detach().cpu().numpy()

			if emb is not None:
				emb = np.concatenate( [emb, new_emb], axis=0 )
			else:
				emb = new_emb

	return emb, spk

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Plot samples')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--data-info-path', type=str, default='./data/', metavar='Path', help='Path to folder containing spk2utt and utt2spk files')
	parser.add_argument('--hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
	parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'inception_mfcc', 'resnet_large'], default='fb', help='Model arch according to input type')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--n-speakers', type=int, default=20, metavar='N', help='number of speakers to compute (default: 1000)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = None

	dataset = h5py.File(args.hdf_file, 'r')

	spk2utt = read_spk2utt(args.data_info_path+'spk2utt')

	if args.model == 'mfcc':
		model = model_.cnn_lstm_mfcc(n_z=args.latent_size, proj_size=None)
	elif args.model == 'fb':
		model = model_.cnn_lstm_fb(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_fb':
		model = model_.ResNet_fb(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=args.latent_size, proj_size=None)
	elif args.model == 'inception_mfcc':
		model = model_.inception_v3(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_large':
		model = model_.ResNet_large_lstm(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=False)

	if args.cuda:
		model = model.cuda(device)

	embeddings, speakers_list = compute_embeddings(model=model, spk2utt_=spk2utt, data_loader=dataset, n_speakers=args.n_speakers, cuda_mode=args.cuda, device=device)

	C = np.cov(embeddings, rowvar=False)

	print('Condition number of the covariance matrix of embeddings: {}'.format(np.linalg.cond(C)))
	print('Rank of the matrix of embeddings: {}/{}'.format(np.linalg.matrix_rank(embeddings), embeddings.shape[1]))

	plt.figure(1)
	plt.title('Embeddings distribution per dimension')
	plt.boxplot(embeddings)
	plt.savefig('BP_UTT_TL.png')

	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(embeddings)

	tsne = manifold.TSNE(n_components=2, init='pca')
	X_tsne = tsne.fit_transform(X)

	plt.figure(2)
	plt.title('2d-tSNE of embeddings')

	colors = iter(cm.rainbow(np.linspace(0, 1, len(np.unique(speakers_list)))))

	for speaker in np.unique(speakers_list):
		
		plt.plot(X_tsne[np.where(speakers_list==speaker),0], X_tsne[np.where(speakers_list==speaker),1], color=next(colors), marker='o')

	plt.savefig('EMB_UTT_TL.png')

	plt.show()
