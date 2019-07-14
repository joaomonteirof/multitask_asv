import argparse
import numpy as np
import torch
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle
import os
import sys
from transformer_encoder import *
from utils.utils import *

def prep_feats(data_, min_nb_frames=100):

	features = data_.T

	if features.shape[1]<min_nb_frames:
		mul = int(np.ceil(min_nb_frames/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :min_nb_frames]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--enroll-data', type=str, default='./data/enroll/', metavar='Path', help='Path to input data')
	parser.add_argument('--test-data', type=str, default='./data/test/', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'se_resnet', 'TDNN', 'transformer'], default='resnet_mfcc', help='Model arch according to input type')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--scores-path', type=str, default='./scores.p', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--read-scores', action='store_true', default=False, help='If set, reads precomputed scores at --scores-path')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

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

		if args.model == 'resnet_mfcc':
			model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_34':
			model = model_.ResNet_34(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_lstm':
			model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_qrnn':
			model = model_.ResNet_qrnn(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_stats':
			model = model_.ResNet_stats(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_large':
			model = model_.ResNet_large(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'resnet_small':
			model = model_.ResNet_small(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'se_resnet':
			model = model_.SE_ResNet(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'TDNN':
			model = model_.TDNN(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
		elif args.model == 'transformer':
			model = make_model(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)

		ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
		model.load_state_dict(ckpt['model_state'], strict=False)

		model.eval()

		if args.cuda:
			model = model.cuda(device)

		enroll_data = None

		files_list = glob.glob(args.enroll_data+'*.scp')

		for file_ in files_list:
			if enroll_data is None:
				enroll_data = { k:v for k,v in read_mat_scp(file_) }
			else:
				for k,v in read_mat_scp(file_):
					enroll_data[k] = v

		files_list = glob.glob(args.test_data+'*.scp')

		test_data = None

		for file_ in files_list:
			if test_data is None:
				test_data = { k:v for k,v in read_mat_scp(file_) }
			else:
				for k,v in read_mat_scp(file_):
					test_data[k] = v

		utterances_enroll, utterances_test, labels = read_trials(args.trials_path)

		print('\nAll data ready. Start of scoring')

		scores = []
		mem_embeddings = {}

		with torch.no_grad():

			for i in range(len(labels)):

				enroll_utt = utterances_enroll[i]

				try:
					emb_enroll = mem_embeddings[enroll_utt]
				except KeyError:

					enroll_utt_data = prep_feats(enroll_data[enroll_utt])

					if args.cuda:
						enroll_utt_data = enroll_utt_data.cuda(device)

					emb_enroll = model.forward(enroll_utt_data).detach()
					mem_embeddings[enroll_utt] = emb_enroll



				test_utt = utterances_test[i]

				try:
					emb_test = mem_embeddings[test_utt]
				except KeyError:

					test_utt_data = prep_feats(test_data[test_utt])

					if args.cuda:
						enroll_utt_data = enroll_utt_data.cuda(device)
						test_utt_data = test_utt_data.cuda(device)

					emb_test = model.forward(test_utt_data).detach()
					mem_embeddings[test_utt] = emb_test

				scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )

		print('\nScoring done')

		with open(args.scores_path, 'wb') as p:
			pickle.dump({'scores':scores, 'labels':labels}, p, protocol=pickle.HIGHEST_PROTOCOL)

	eer, auc, avg_precision, acc, threshold = compute_metrics(np.asarray(labels), np.asarray(scores))

	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))
