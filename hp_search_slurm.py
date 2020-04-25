from concurrent import futures
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import subprocess
import shlex
import numpy as np
from time import sleep
import pickle
import glob
import torch
import os
import shutil

def get_file_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.p'

	while os.path.isfile(fname):
		fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.p'

	file_ = open(fname, 'wb')
	pickle.dump(None, file_)
	file_.close()

	return fname

def kill_job(id_):

	try:
		status = subprocess.check_output('scancel ' + id_, shell=True)
		print(' ')
		print('Job {} killed'.format(id_))
		print(' ')
	except:
		pass

def remove_err_out_files(id_):
	files_list = glob.glob('*'+id_+'.*')
	for file_ in files_list:
		os.remove(file_)

# Training settings
parser=argparse.ArgumentParser(description='HP search for ASV')
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for valid (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--slurm-sub-file', type=str, default='./run_hp.sh', metavar='Path', help='Path to sge submission file')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod', 'TDNN_multipool', 'transformer', 'all'], default='resnet_mfcc', help='Model arch according to input type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--temp-folder', type=str, default='temp', metavar='Path', help='Temp folder for pickle files')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()
args.cuda=True if not args.no_cuda else False

def train(lr, l2, max_gnorm, momentum, margin, lambda_, swap, latent_size, n_frames, model, ncoef, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, slurm_submission_file, tmp_dir, cp_path, softmax, delta, logdir):

	file_name = get_file_name(tmp_dir)
	np.random.seed()

	command = 'sbatch' + ' ' + slurm_submission_file + ' ' + str(lr) + ' ' + str(l2) + ' ' + str(max_gnorm) + ' ' + str(momentum) + ' ' + str(margin) + ' ' + str(lambda_) + ' ' + str(swap) + ' ' + str(int(latent_size)) + ' ' + str(int(n_frames)) + ' ' + str(model) + ' ' + str(ncoef) + ' ' + str(epochs) + ' ' + str(batch_size) + ' ' + str(valid_batch_size) + ' ' + str(n_workers) + ' ' + str(cuda) + ' ' + str(train_hdf_file) + ' ' + str(valid_hdf_file) + ' ' + str(file_name) + ' ' + str(cp_path) + ' ' + str(file_name.split('/')[-1]+'t') + ' ' + str(softmax) + ' ' + str(delta) + ' ' + str(logdir)

	for j in range(10):

		sleep(np.random.randint(10,120,1)[0])

		result=None

		p=subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

		out = p.communicate()
		job_id = out[0].decode('utf-8').split(' ')[3]

		result_file = open(file_name, 'rb')
		result = pickle.load(result_file)
		result_file.close()

		if result is not None:
			remove_err_out_files(job_id)
			os.remove(file_name)

			print(' ')
			print('Best EER in result file ' + file_name.split('/')[-1].split('.p')[0] + ' was: {}'.format(result))
			print(' ')
			print('With hyperparameters:')
			print('Model: {}'.format(model))
			print('N frames: {}'.format(int(n_frames)))
			print('Embeddings size: {}'.format(int(latent_size)))
			print('LR: {}'.format(lr))
			print('momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('Max. grad norm: {}'.format(max_gnorm))
			print('lambda: {}'.format(lambda_))
			print('Margin: {}'.format(margin))
			print('Swap: {}'.format(swap))
			print('Softmax Mode: {}'.format(softmax))
			print('Delta features: {}'.format(delta))
			print(' ')

			return result

	return 0.5

lr=instru.var.OrderedDiscrete([0.1, 0.01, 0.001, 0.0001, 0.00001])
l2=instru.var.OrderedDiscrete([0.001, 0.0005, 0.0001, 0.00005, 0.00001])
max_gnorm=instru.var.OrderedDiscrete([10.0, 100.0, 1000.0])
momentum=instru.var.OrderedDiscrete([0.1, 0.3, 0.5, 0.7, 0.9])
margin=instru.var.OrderedDiscrete([0.1, 0.01, 0.001, 0.0001, 0.00001])
lambda_=instru.var.OrderedDiscrete([0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.50])
swap=instru.var.OrderedDiscrete([True, False])
latent_size=instru.var.OrderedDiscrete([64, 128, 256, 512])
n_frames=instru.var.OrderedDiscrete([300, 400, 500, 600, 800])
model=instru.var.OrderedDiscrete(['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod', 'TDNN_multipool', 'transformer']) if args.model=='all' else args.model
ncoef=args.ncoef
epochs=args.epochs
batch_size=args.batch_size
valid_batch_size=args.valid_batch_size
n_workers=args.workers
cuda=args.cuda
train_hdf_file=args.train_hdf_file
valid_hdf_file=args.valid_hdf_file
slurm_sub_file=args.slurm_sub_file
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])
delta=instru.var.OrderedDiscrete([True, False])
logdir=args.logdir

tmp_dir = os.getcwd() + '/' + args.temp_folder + '/'

if not os.path.isdir(tmp_dir):
	os.mkdir(tmp_dir)

instrum=instru.Instrumentation(lr, l2, max_gnorm, momentum, margin, lambda_, swap, latent_size, n_frames, model, ncoef, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, slurm_sub_file, tmp_dir, checkpoint_path, softmax, delta, logdir)

hp_optimizer=optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget, num_workers=args.hp_workers)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor, verbosity=2))

shutil.rmtree(tmp_dir)
