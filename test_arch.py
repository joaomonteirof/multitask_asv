from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'inception_mfcc', 'resnet_large', 'resnet_small', 'all'], default='fb', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
args = parser.parse_args()

if args.model == 'mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.cnn_lstm_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('mfcc', mu.size())
if args.model == 'fb' or args.model == 'all':
	batch = torch.rand(3, 1, 40, 200)
	model = model_.cnn_lstm_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print('fb', mu.size())
if args.model == 'resnet_fb' or args.model == 'all':
	batch = torch.rand(3, 1, 40, 200)
	model = model_.ResNet_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print(mu.size())
if args.model == 'resnet_mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_mfcc', mu.size())
if args.model == 'resnet_lstm' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_lstm', mu.size())
if args.model == 'resnet_stats' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_stats(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_stats', mu.size())
if args.model == 'inception_mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.inception_v3(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('inception_mfcc', mu.size())
if args.model == 'resnet_large' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_large_lstm(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_large', mu.size())
if args.model == 'resnet_small' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_small(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_small', mu.size())
