from concurrent import futures
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_valid
import os
import sys
from transformer_encoder import *
from utils.utils import *

def get_file_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	file_ = open(fname, 'wb')
	pickle.dump(None, file_)
	file_.close()

	return fname

# Training settings
parser=argparse.ArgumentParser(description='HP random search for ASV')
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for valid (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'resnet_large', 'resnet_small', 'se_resnet', 'TDNN', 'transformer', 'all'], default='resnet_mfcc', help='Model arch according to input type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--data-info-path', type=str, default='./data/', metavar='Path', help='Path to folder containing spk2utt and utt2spk files')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()
args.cuda=True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, margin, lambda_, patience, swap, latent_size, n_frames, model, ncoef, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, cp_path, softmax, delta):

	if cuda:
		device=get_freer_gpu()

	train_dataset = Loader(hdf5_name = train_hdf_file, max_nb_frames = int(n_frames), delta = delta)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n.workers, worker_init_fn=set_np_randomseed)

	valid_dataset = Loader_valid(hdf5_name = valid_hdf_file, max_nb_frames = int(n_frames), delta = delta)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=n_workers, worker_init_fn=set_np_randomseed)

	if model == 'resnet_mfcc':
		model=model_.ResNet_mfcc(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=ncoef, sm_type=softmax, delta=delta)
	elif model == 'resnet_lstm':
		model=model_.ResNet_lstm(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=ncoef, sm_type=softmax, delta=delta)
	elif model == 'resnet_stats':
		model=model_.ResNet_stats(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=ncoef, sm_type=softmax, delta=delta)
	elif args.model == 'resnet_large':
		model = model_.ResNet_large(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=softmax, delta=delta)
	elif args.model == 'resnet_small':
		model = model_.ResNet_small(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=softmax, delta=delta)
	elif args.model == 'se_resnet':
		model = model_.SE_ResNet(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=softmax, delta=delta)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=softmax, delta=delta)
	elif args.model == 'transformer':
		model = make_model(n_z=int(latent_size), proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=softmax, delta=delta)

	if cuda:
		model=model.to(device)
	else:
		device=None

	optimizer=optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

	trainer=TrainLoop(model, optimizer, train_loader, valid_loader, margin=margin, lambda_=lambda_, patience=int(patience), verbose=-1, device=device, cp_name=get_file_name(cp_path), save_cp=True, checkpoint_path=cp_path, swap=swap, softmax=True, pretrain=False, mining=True, cuda=cuda)

	return trainer.train(n_epochs=epochs)

lr=instru.var.Array(1).asfloat().bounded(1, 4).exponentiated(base=10, coeff=-1)
l2=instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
momentum=instru.var.Array(1).asfloat().bounded(0.10, 0.95)
margin=instru.var.Array(1).asfloat().bounded(0.10, 1.00)
lambda_=instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
patience=instru.var.Array(1).asfloat().bounded(1, 100)
swap=instru.var.OrderedDiscrete([True, False])
latent_size=instru.var.Array(1).asfloat().bounded(64, 512)
n_frames=instru.var.Array(1).asfloat().bounded(600, 1000)
model=instru.var.OrderedDiscrete(['resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'inception_mfcc', 'resnet_large', 'resnet_small']) if args.model=='all' else args.model
ncoef=args.ncoef
epochs=args.epochs
batch_size=args.batch_size
valid_batch_size=args.valid_batch_size
n_workers=args.workers
cuda=args.cuda
train_hdf_file=args.train_hdf_file
valid_hdf_file=args.valid_hdf_file
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])
delta=instru.var.OrderedDiscrete([True, False])

instrum=instru.Instrumentation(lr, l2, momentum, margin, lambda_, patience, swap, latent_size, n_frames, model, ncoef, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, checkpoint_path, softmax, delta)

hp_optimizer=optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget, num_workers=args.hp_workers)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor))
