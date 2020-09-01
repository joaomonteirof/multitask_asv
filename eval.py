import argparse
import numpy as np
import torch
import torch.nn.functional as F
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle
import os
import sys
import pathlib
from utils.utils import *
from librosa.feature import delta as delta_

def prep_feats(data_, delta=False):

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

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--test-data', type=str, default='./data/test/', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod', 'TDNN_multipool', 'transformer'], default='resnet_mfcc', help='Model arch according to input type')
	parser.add_argument('--delta', action='store_true', default=False, help='Enables extra data channels')
	parser.add_argument('--inner', action='store_true', default=False, help='Get embeddings from inner layer')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--read-scores', action='store_true', default=False, help='If set, reads precomputed scores at --scores-path')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-out', action='store_true', default=False, help='Disables writing scores in out file')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--out-prefix', type=str, default=None, metavar='Path', help='Prefix to be added to score files')
	parser.add_argument('--eval', action='store_true', default=False, help='Eval trials - Does not compute perf metrics')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print('\n', args, '\n')

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	if args.read_scores:

		print('Reading pre-computed scores from: {}'.format(args.scores_path))

		with open(args.scores_path, 'rb') as p:
			scores_dict = pickle.load(p)
			scores, labels = scores_dict['scores'], scores_dict['labels']
	else:

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
		elif args.model == 'TDNN_multipool':
			model = model_.TDNN_multipool(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)
		elif args.model == 'transformer':
			model = model_.transformer_enc(n_z=args.latent_size, proj_size=0, ncoef=args.ncoef, delta = args.delta)

		ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

		try:
			model.load_state_dict(ckpt['model_state'], strict=True)
		except RuntimeError as err:
			print("Runtime Error: {0}".format(err))
		except:
			print("Unexpected error:", sys.exc_info()[0])
			raise

		model.eval()

		if args.cuda:
			model = model.to(device)

		test_data = None
		files_list = glob.glob(args.test_data+'*.scp')

		for file_ in files_list:
			if test_data is None:
				test_data = { k:v for k,v in read_mat_scp(file_) }
			else:
				for k,v in read_mat_scp(file_):
					test_data[k] = v

		utterances_enroll, utterances_test, labels = read_trials(args.trials_path)

		print('\nAll data ready. Start of scoring')

		scores = []
		out_cos = []
		mem_embeddings = {}

		with torch.no_grad():

			for i in range(len(labels)):

				enroll_utt = utterances_enroll[i]

				try:
					emb_enroll = mem_embeddings[enroll_utt]
				except KeyError:

					enroll_utt_data = prep_feats(test_data[enroll_utt], args.delta)

					if args.cuda:
						enroll_utt_data = enroll_utt_data.to(device)

					emb_enroll = model.forward(enroll_utt_data)[1].detach() if args.inner else model.forward(enroll_utt_data)[0].detach()
					emb_enroll = F.normalize(emb_enroll, p=2, dim=1)
					mem_embeddings[enroll_utt] = emb_enroll

				test_utt = utterances_test[i]

				try:
					emb_test = mem_embeddings[test_utt]
				except KeyError:

					test_utt_data = prep_feats(test_data[test_utt], args.delta)

					if args.cuda:
						enroll_utt_data = enroll_utt_data.to(device)
						test_utt_data = test_utt_data.to(device)

					emb_test = model.forward(test_utt_data)[1].detach() if args.inner else model.forward(test_utt_data)[0].detach()
					emb_test = F.normalize(emb_test, p=2, dim=1)
					mem_embeddings[test_utt] = emb_test

				scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )
				out_cos.append([enroll_utt, test_utt, scores[-1]])

		print('\nScoring done')

		if not args.no_out:

			with open(args.out_path+args.out_prefix+'cos_scores.out' if args.out_prefix is not None else args.out_path+'cos_scores.out', 'w') as f:
				for el in out_cos:
					item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
					f.write("%s" % item)

	if not args.eval:
		eer, auc, avg_precision, acc, threshold = compute_metrics(np.asarray(labels), np.asarray(scores))
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))
