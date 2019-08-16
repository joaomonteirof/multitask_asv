import argparse
import numpy as np
import glob
import torch
import os
import sys
import pathlib
from kaldi_io import read_mat_scp, open_or_fd, write_vec_flt
import model as model_
import scipy.io as sio
from utils.utils import *
from librosa.feature import delta as delta_

def prep_feats(data_, delta=False):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	features = features[np.newaxis, :, :]

	if delta:
		features = np.concatenate([features, delta_(features,width=3,order=1), delta_(features,width=3,order=2)], axis=0)

	return torch.from_numpy(features[np.newaxis, :, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute embeddings')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to input data')
	parser.add_argument('--utt2spk', type=str, default=None, metavar='Path', help='Optional path for utt2spk')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod'], default='resnet_mfcc', help='Model arch according to input type')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--delta', action='store_true', default=False, help='Enables extra data channels')
	parser.add_argument('--eps', type=float, default=0.0, metavar='eps', help='Add noise to embeddings')
	parser.add_argument('--inner', action='store_true', default=False, help='Get embeddings from inner layer')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()
		if args.model == 'resnet_qrnn':
			import cupy
			cupy.cuda.Device(int(str(device).split(':')[-1])).use()

	if args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_34':
		model = model_.ResNet_34(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_qrnn':
		model = model_.ResNet_qrnn(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_large':
		model = model_.ResNet_large(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_small':
		model = model_.ResNet_small(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'resnet_2d':
		model = model_.ResNet_2d(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN_att':
		model = model_.TDNN_att(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN_multihead':
		model = model_.TDNN_multihead(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN_lstm':
		model = model_.TDNN_lstm(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN_aspp':
		model = model_.TDNN_aspp(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
	elif args.model == 'TDNN_mod':
		model = model_.TDNN_mod(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=False)

	model.eval()

	if args.cuda:
		model = model.to(device)

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	if args.utt2spk:
		utt2spk = read_utt2spk(args.utt2spk)

	print('Start of data embeddings computation')

	embeddings = {}

	with torch.no_grad():

		for file_ in scp_list:

			data = { k:m for k,m in read_mat_scp(file_) }

			for i, utt in enumerate(data):

				if args.utt2spk:
					if not utt in utt2spk:
						print('Skipping utterance '+ utt)
						continue

				feats = prep_feats(data[utt], args.delta)

				try:
					if args.cuda:
						feats = feats.to(device)
						model = model.to(device)

					emb = model.forward(feats)[1] if args.inner else model.forward(feats)[0]

				except:
					feats = feats.cpu()
					model = model.cpu()

					emb = model.forward(feats)[1] if args.inner else model.forward(feats)[0]

				embeddings[utt] = emb.detach().cpu().numpy().squeeze()

				if args.eps>0.0:
					embeddings[utt] += args.eps*np.random.randn(embeddings[utt].shape[0])

	print('Storing embeddings in output file')

	out_name = args.path_to_data.split('/')[-2] if not args.utt2spk else args.utt2spk.split('/')[-2]
	file_name = args.out_path+out_name+'.ark'

	if os.path.isfile(file_name):
		os.remove(file_name)
		print(file_name + ' Removed')

	with open_or_fd(file_name,'wb') as f:
		for k,v in embeddings.items(): write_vec_flt(f, v, k)

	print('End of embeddings computation.')
