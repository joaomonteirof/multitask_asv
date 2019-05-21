import argparse
import numpy as np
from sklearn import metrics
import scipy.io as sio
import glob
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_hist(y, y_score):

	y_score = (np.array(y_score)-np.min(y_score))/(np.max(y_score)-np.min(y_score))
	y=np.array(y)

	target = y_score[np.where(y==1.)]
	nontarget = y_score[np.where(y==0.)]

	print(target.shape)
	print(nontarget.shape)

	plt.figure()
	density = stats.gaussian_kde(target)
	n, x, _ = plt.hist(target, bins=np.linspace(0, 1, 30), histtype=u'step', density=True)  
	plt.plot(x, density(x), color='darkblue', lw=2, label='target')
	density = stats.gaussian_kde(nontarget)
	n, x, _ = plt.hist(nontarget, bins=np.linspace(0, 1, 30), histtype=u'step', density=True)  
	plt.plot(x, density(x), color='darkorange', lw=2, label='nontarget')
	plt.title('Scores Histogram')
	plt.legend(loc="upper right")
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

	plot_hist(label_list, score_list)
