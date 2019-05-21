import argparse
import numpy as np
import torch
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle

def set_device(trials=10):
	a = torch.rand(1)

	for i in range(torch.cuda.device_count()):
		for j in range(trials):

			torch.cuda.set_device(i)
			try:
				a = a.cuda()
				print('GPU {} selected.'.format(i))
				return
			except:
				pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def compute_metrics(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	auc = metrics.auc(fpr, tpr)

	avg_precision = metrics.average_precision_score(y, y_score)

	pred = np.asarray([1 if score > eer_threshold else 0 for score in y_score])
	acc = metrics.accuracy_score(y ,pred)

	return eer, auc, avg_precision, acc, eer_threshold

def read_trials(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	enroll_spk_list, test_utt_list, labels_list = [], [], []

	for line in utt_labels:
		enroll_spk, test_utt, label = line.split(' ')
		enroll_spk_list.append(enroll_spk)
		test_utt_list.append(test_utt)
		labels_list.append(1 if label=='target\n' else 0)

	return enroll_spk_list, test_utt_list, labels_list

def compute_avg_emb(data_path):

	files_list = glob.glob(data_path+'*.npy')

	files = []

	for file_ in files_list:
		files.append(np.load(file_))

	files = np.asarray(files)

	return torch.from_numpy(files.mean(0))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to input data')
	parser.add_argument('--unlabeled-data-path', type=str, default=None, metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	args = parser.parse_args()

	utterances_enroll, utterances_test, labels = read_trials(args.trials_path)

	if args.unlabeled_data_path:
		avg_emb = compute_avg_emb(args.unlabeled_data_path)

	print('\nAll data ready. Start of scoring')

	scores = []
	mem_embeddings = {}

	with torch.no_grad():

		for i in range(len(labels)):

			enroll_utt = utterances_enroll[i]

			try:
				enroll_utt_emb = mem_embeddings[enroll_utt]

			except KeyError:

				enroll_utt_emb = torch.from_numpy(np.load(args.data_path+enroll_utt+'.npy'))

				if args.unlabeled_data_path:
					enroll_utt_emb-=avg_emb

				mem_embeddings[enroll_utt] = enroll_utt_emb


			test_utt = utterances_test[i]

			try:
				test_utt_emb = mem_embeddings[test_utt]

			except KeyError:

				test_utt_emb = torch.from_numpy(np.load(args.data_path+test_utt+'.npy'))

				if args.unlabeled_data_path:
					test_utt_emb-=avg_emb

				mem_embeddings[test_utt] = test_utt_emb

			scores.append( torch.nn.functional.cosine_similarity(enroll_utt_emb, test_utt_emb).item() )

	print('\nScoring done')

	eer, auc, avg_precision, acc, threshold = compute_metrics(np.asarray(labels), np.asarray(scores))

	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))
