from __future__ import print_function
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
import pickle
from utils.utils import *
from transformer_encoder import *

# Training settings
parser = argparse.ArgumentParser(description='Train for hp search')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for valid (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--margin', type=float, default=0.3, metavar='m', help='margin fro triplet loss (default: 0.3)')
parser.add_argument('--lamb', type=float, default=0.001, metavar='l', help='Entropy regularization penalty (default: 0.001)')
parser.add_argument('--swap', type=str, default=None, help='Swaps anchor and positive depending on distance to negative example')
parser.add_argument('--patience', type=int, default=10, metavar='S', help='Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)')
parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'se_resnet', 'TDNN', 'TDNN_mod', 'TDNN_multihead', 'transformer', 'aspp_res', 'pyr_rnn'], default='resnet_mfcc', help='Model arch according to input type')
parser.add_argument('--softmax', choices=['none', 'softmax', 'am_softmax'], default='none', help='Softmax type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--n-frames', type=int, default=800, metavar='N', help='maximum number of frames per utterance (default: 800)')
parser.add_argument('--cuda', type=str, default=None)
parser.add_argument('--delta', type=str, default=None)
parser.add_argument('--out-file', type=str, default='./eer.p')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--cp-name', type=str, default=None)
args = parser.parse_args()
args.cuda = True if args.cuda=='True' and torch.cuda.is_available() else False
args.swap = True if args.swap=='True' else False
args.delta = True if args.delta=='True' else False

if args.cuda:
	device = get_freer_gpu()
	if args.model == 'resnet_qrnn':
		import cupy
		cupy.cuda.Device(int(str(device).split(':')[-1])).use()
else:
	device = None

train_dataset = Loader(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, delta = args.delta)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

valid_dataset = Loader_valid(hdf5_name = args.valid_hdf_file, max_nb_frames = args.n_frames, delta = args.delta)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

if args.model == 'resnet_mfcc':
	model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_34':
	model = model_.ResNet_34(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_lstm':
	model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_qrnn':
	model = model_.ResNet_qrnn(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_stats':
	model = model_.ResNet_stats(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_large':
	model = model_.ResNet_large(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_small':
	model = model_.ResNet_small(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'resnet_2d':
	model = model_.ResNet_2d(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'se_resnet':
	model = model_.SE_ResNet(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'TDNN':
	model = model_.TDNN(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'TDNN_mod':
	model = model_.TDNN_mod(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'TDNN_multihead':
	model = model_.TDNN_multihead(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'transformer':
	model = make_model(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'aspp_res':
	model = model_.aspp_res(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)
elif args.model == 'pyr_rnn':
	model = model_.pyr_rnn(n_z=args.latent_size, proj_size=train_dataset.n_speakers if args.softmax!='none' or args.pretrain else 0, ncoef=args.ncoef, sm_type=args.softmax, delta=args.delta)

if args.cuda:
	model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, margin=args.margin, lambda_=args.lamb, patience=args.patience, verbose=-1, device=device, cp_name=args.cp_name, save_cp=True, checkpoint_path=args.checkpoint_path, swap=args.swap, softmax=True, pretrain=False, mining=True, cuda=args.cuda)

print('Cuda Mode: {}'.format(args.cuda))
print('Softmax Mode: {}'.format(args.softmax))
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
print('Delta features: {}'.format(args.delta))
print('Max input length: {}'.format(args.n_frames))
print('Number of CCs: {}'.format(args.ncoef))
print(' ')

best_eer = trainer.train(n_epochs=args.epochs, save_every=args.epochs+10)

out_file = open(args.out_file, 'wb')
pickle.dump(best_eer, out_file)
out_file.close()
