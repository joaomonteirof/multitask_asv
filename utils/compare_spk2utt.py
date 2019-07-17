import argparse
import h5py
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

def parse_spk2utt(spk2utt_dict, min_utt=1, plot=False):

	spk2count_list = []

	for spk, utt_list in spk2utt_dict.items():
		spk2count_list.append(len(utt_list))

	print('Original spk2utt:')

	print('Number of speakers: {}'.format(len(spk2count_list)))
	print('Number of recordings: {}'.format(np.sum(spk2count_list)))
	print('Max: {}, Min: {}, AVG: {}, STD: {} recordings per speaker'.format(np.max(spk2count_list), np.min(spk2count_list), np.mean(spk2count_list), np.std(spk2count_list)))

	print('Filtered spk2utt:')

	spk2count_min = [i for i in spk2count_list if i>= min_utt]

	print('Number of speakers: {}'.format(len(spk2count_min)))
	print('Number of recordings: {}'.format(np.sum(spk2count_min)))
	print('Max: {}, Min: {}, AVG: {}, STD: {} recordings per speaker'.format(np.max(spk2count_min), np.min(spk2count_min), np.mean(spk2count_min), np.std(spk2count_min)))

	if plot:
		plt.figure(1)
		plt.hist(spk2count_list, bins=80)
		plt.show()

def hdf_to_spk2utt(hdf_path):

	open_file = h5py.File(hdf_path, 'r')

	speakers_list = list(open_file)

	spk2utt_ = {}

	for spk in speakers_list:
		spk2utt_[spk] = list(open_file[spk])

	open_file.close()

	return spk2utt_

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path1', type=str, default=None)
	parser.add_argument('--path2', type=str, default=None)
	args = parser.parse_args()

	spk2utt_1 = read_spk2utt(args.path1)
	spk2utt_2 = read_spk2utt(args.path2)

	overlap = set(spk2utt_1.keys()) & set(spk2utt_2.keys())

	print(overlap)
	print(len(overlap))

