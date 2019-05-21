import argparse
import numpy as np
from sklearn import metrics
import scipy.io as sio
import glob
import matplotlib
import matplotlib.pyplot as plt

def plot_det(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	"""
	plt.figure()
	lw = 2
	plt.plot(100.*fpr, 100.*fnr, color='darkorange', lw=lw, label='DET curve (EER = %0.4f)' % eer)
	#plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.05])
	plt.xscale('linear')
	plt.yscale('linear')
	plt.xlabel('False Positive Rate')
	plt.ylabel('False Negative Rate')
	plt.title('Detection Error Tradeoff Curve')
	plt.legend(loc="upper right")
	plt.show()
	"""

	fps,fns=100*fpr,100*fnr

	fig,ax = plt.subplots()
	plt.plot(fps,fns)
	plt.yscale('log')
	plt.xscale('log')
	ticks_to_use = [0.1,0.5,1,2,5,10,20,40,60]
	ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	ax.set_xticks(ticks_to_use)
	ax.set_yticks(ticks_to_use)
	plt.axis([0.09,65,0.09,65])
	plt.show()

def read_trials(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	trials_dict = {}

	for line in utt_labels:
		enroll_spk, test_utt, label = line.split(' ')
		trials_dict[enroll_spk+'_'+test_utt]=1. if label=='target\n' else 0.

	return trials_dict

def read_scores(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	scores_dict = {}

	for line in utt_labels:
		enroll_spk, test_utt, score = line.split(' ')
		scores_dict[enroll_spk+'_'+test_utt]=float(score.strip())

	return scores_dict

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='plot DET given scores')
	parser.add_argument('--scores-path', type=str, default='./data/scores', metavar='Path', help='Path to scores')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials')
	args = parser.parse_args()

	trials = read_trials(args.trials_path)
	scores = read_scores(args.scores_path)

	score_list, label_list = [], []

	for trial in trials:
		label_list.append(trials[trial])
		score_list.append(scores[trial])

	plot_det(label_list, score_list)
