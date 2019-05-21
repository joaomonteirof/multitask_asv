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
parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'inception_mfcc', 'resnet_large'], default='fb', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
args = parser.parse_args()

if args.model == 'mfcc':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.cnn_lstm_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
if args.model == 'fb':
	batch = torch.rand(3, 1, 40, 200)
	model = model_.cnn_lstm_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_fb':
	batch = torch.rand(3, 1, 40, 200)
	model = model_.ResNet_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_mfcc':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_lstm':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_stats':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_stats(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'inception_mfcc':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.inception_v3(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_large':
	batch = torch.rand(3, 1, args.ncoef, 200)
	model = model_.ResNet_large_lstm(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print(mu.size())
