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
	parser.add_argument('--unlab-data', type=str, default=None, metavar='Path', help='Path to unlabeled data for centering')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod', 'TDNN_multipool', 'transformer'], default='resnet_mfcc', help='Model arch according to input type')
	parser.add_argument('--delta', action='store_true', default=False, help='Enables extra data channels')
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
		else:
			device = torch.device('cpu')

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

		print('\n', model.load_state_dict(ckpt['model_state'], strict=False), '\n')

		model.eval()
		model = model.to(device)

		test_data = None
		files_list = glob.glob(args.test_data+'*.scp')

		for file_ in files_list:
			if test_data is None:
				test_data = { k:v for k,v in read_mat_scp(file_) }
			else:
				for k,v in read_mat_scp(file_):
					test_data[k] = v

		unlab_emb_outer, unlab_emb_inner = None, None

		if args.unlab_data:

			files_list = glob.glob(args.unlab_data+'*.scp')

			unlab_emb_outer, unlab_emb_inner = [], []

			for file_ in files_list:

				for k,v in read_mat_scp(file_):

					unlab_utt_data = prep_feats(v, args.delta).to(device)

					with torch.no_grad():
						u_emb_outer, u_emb_inner = model.forward(unlab_utt_data)

					unlab_emb_outer.append(u_emb_outer.detach().cpu())
					unlab_emb_inner.append(u_emb_inner.detach().cpu())

			unlab_emb_outer = torch.cat(unlab_emb_outer, 0).mean(0, keepdim=True).to(device)
			unlab_emb_inner = torch.cat(unlab_emb_inner, 0).mean(0, keepdim=True).to(device)

		utterances_enroll, utterances_test, labels = read_trials(args.trials_path)

		print('\nAll data ready. Start of scoring')

		scores = {}
		out_cos = {}
		for score_type in ['inner', 'outer', 'fus_emb', 'fus_score']:
			scores[score_type] = []
			out_cos[score_type] = []

		mem_embeddings_inner = {}
		mem_embeddings_outer = {}

		with torch.no_grad():

			for i in range(len(labels)):

				enroll_utt = utterances_enroll[i]

				try:
					emb_enroll_inner = mem_embeddings_inner[enroll_utt]
					emb_enroll_outer = mem_embeddings_outer[enroll_utt]
				except KeyError:

					enroll_utt_data = prep_feats(test_data[enroll_utt], args.delta).to(device)

					emb_enroll_outer, emb_enroll_inner = model.forward(enroll_utt_data)
					emb_enroll_outer, emb_enroll_inner = emb_enroll_outer.detach(), emb_enroll_inner.detach()

					if unlab_emb_inner is not None:
						emb_enroll_inner -= unlab_emb_inner
						emb_enroll_outer -= unlab_emb_outer

					mem_embeddings_inner[enroll_utt] = emb_enroll_inner
					mem_embeddings_outer[enroll_utt] = emb_enroll_outer

				test_utt = utterances_test[i]

				try:
					emb_test_inner = mem_embeddings_inner[test_utt]
					emb_test_outer = mem_embeddings_outer[test_utt]
				except KeyError:

					test_utt_data = prep_feats(test_data[test_utt], args.delta).to(device)

					emb_test_outer, emb_test_inner = model.forward(test_utt_data)
					emb_test_outer, emb_test_inner = emb_test_outer.detach(), emb_test_inner.detach()

					if unlab_emb_inner is not None:
						emb_test_inner -= unlab_emb_inner
						emb_test_outer -= unlab_emb_outer

					mem_embeddings_inner[test_utt] = emb_test_inner
					mem_embeddings_outer[test_utt] = emb_test_outer

				scores['inner'].append( torch.nn.functional.cosine_similarity(emb_enroll_inner, emb_test_inner).squeeze().item() )
				scores['outer'].append( torch.nn.functional.cosine_similarity(emb_enroll_outer, emb_test_outer).squeeze().item() )

				emb_enroll_fus = torch.cat((emb_enroll_outer, emb_enroll_inner), -1)
				emb_test_fus = torch.cat((emb_test_outer, emb_test_inner), -1)

				scores['fus_emb'].append( torch.nn.functional.cosine_similarity(emb_enroll_fus, emb_test_fus).squeeze().item() )
				scores['fus_score'].append( 0.5*(scores['inner']+scores['outer']) )

				for score_type in ['inner', 'outer', 'fus_emb', 'fus_score']:
					out_cos[score_type].append([enroll_utt, test_utt, scores[score_type][-1]])

		print('\nScoring done')

		if not args.no_out:

			for score_type in ['inner', 'outer', 'fus_emb', 'fus_score']:
				if args.out_prefix:
					out_file_name = f'{args.out_prefix}_{score_type}_cos_scores.out'
				else:
					out_file_name = f'{score_type}_cos_scores.out'

				with open(os.path.join(args.out_path, out_file_name), 'w') as f:
					for el in out_cos:
						item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
						f.write("%s" % item)

	if not args.eval:
		for score_type in ['inner', 'outer', 'fus_emb', 'fus_score']:
			print(f'Evaluation for {score_type} scores:\n')
			eer, auc, avg_precision, acc, threshold = compute_metrics(np.asarray(labels), np.asarray(scores[score_type]))
			print(f'ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {eer}, {auc}, {avg_precision}, {acc}, {threshold}')
