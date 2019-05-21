from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_softmax, Loader_mining, Loader_pretrain, Loader_test
import os
import sys
from time import sleep

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(20)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with combined loss')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--margin', type=float, default=0.3, metavar='m', help='margin fro triplet loss (default: 0.3)')
parser.add_argument('--lamb', type=float, default=0.001, metavar='l', help='Entropy regularization penalty (default: 0.001)')
parser.add_argument('--swap', action='store_true', default=False, help='Swaps anchor and positive depending on distance to negative example')
parser.add_argument('--patience', type=int, default=10, metavar='S', help='Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'inception_mfcc', 'resnet_large'], default='fb', help='Model arch according to input type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--n-frames', type=int, default=800, metavar='N', help='maximum number of frames per utterance (default: 800)')
parser.add_argument('--n-cycles', type=int, default=10, metavar='N', help='cycles over speakers list to complete 1 epoch')
parser.add_argument('--valid-n-cycles', type=int, default=500, metavar='N', help='cycles over speakers list to complete 1 epoch')
parser.add_argument('--softmax', choices=['none', 'softmax', 'am_softmax'], default='none', help='Softmax type')
parser.add_argument('--pretrain', action='store_true', default=False, help='Adds softmax layer for speaker identification and train exclusively with CE minimization')
parser.add_argument('--mine-triplets', action='store_true', default=False, help='Enables distance mining for triplets')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if args.pretrain:
	train_dataset = Loader_pretrain(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)
elif args.mine_triplets:
	train_dataset = Loader_mining(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)
elif args.softmax!='none':
	train_dataset = Loader_softmax(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)
else:
	train_dataset = Loader(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

if args.valid_hdf_file is not None:
	valid_dataset = Loader(hdf5_name = args.valid_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=set_np_randomseed)
else:
	valid_loader=None

if args.cuda:
	device = get_freer_gpu()
else:
	device = None

if args.model == 'mfcc':
	model = model_.cnn_lstm_mfcc(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'fb':
	model = model_.cnn_lstm_fb(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, sm_type=args.softmax)
elif args.model == 'resnet_fb':
	model = model_.ResNet_fb(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, sm_type=args.softmax)
elif args.model == 'resnet_mfcc':
	model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'resnet_lstm':
	model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'resnet_stats':
	model = model_.ResNet_stats(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'inception_mfcc':
	model = model_.inception_v3(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'resnet_large':
	model = model_.ResNet_large_lstm(n_z=args.latent_size, proj_size=len(train_dataset.speakers_list) if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax)

if args.pretrained_path is not None:
	ckpt = torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)

	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.cuda:
	model = model.cuda(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, margin=args.margin, lambda_=args.lamb, patience=args.patience, verbose=args.verbose, device=device, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, swap=args.swap, softmax=args.softmax, pretrain=args.pretrain, mining=args.mine_triplets, cuda=args.cuda)

if args.verbose >0:
	print(' ')
	print('Cuda Mode: {}'.format(args.cuda))
	print('Pretrain Mode: {}'.format(args.pretrain))
	print('Softmax Mode: {}'.format(args.softmax))
	print('Mining Mode: {}'.format(args.mine_triplets))
	print('Selected model: {}'.format(args.model))
	print('Embeddings size: {}'.format(args.latent_size))
	print('Batch size: {}'.format(args.batch_size))
	print('LR: {}'.format(args.lr))
	print('momentum: {}'.format(args.momentum))
	print('l2: {}'.format(args.l2))
	print('lambda: {}'.format(args.lamb))
	print('Margin: {}'.format(args.margin))
	print('Swap: {}'.format(args.swap))
	print('Patience: {}'.format(args.patience))
	print(' ')

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
