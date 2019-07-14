from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_
from utils.utils import *
from transformer_encoder import *

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'se_resnet', 'TDNN', 'transformer', 'all'], default='all', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--delta', action='store_true', default=False, help='Enables extra data channels')
args = parser.parse_args()

if args.model == 'resnet_mfcc' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_mfcc', mu.size(), out.size())
if args.model == 'resnet_34' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_34(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_34', mu.size(), out.size())
if args.model == 'resnet_lstm' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_lstm', mu.size(), out.size())
if args.model == 'resnet_qrnn' or args.model == 'all':
	device = get_freer_gpu()
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200).to(device)
	model = model_.ResNet_qrnn(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)).to(device))
	print('resnet_qrnn', mu.size(), out.size())
if args.model == 'resnet_stats' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_stats(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_stats', mu.size(), out.size())
if args.model == 'resnet_large' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_large(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_large', mu.size(), out.size())
if args.model == 'resnet_small' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_small(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_small', mu.size(), out.size())
if args.model == 'se_resnet' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.SE_ResNet(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('se_resnet', mu.size(), out.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('TDNN', mu.size(), out.size())
if args.model == 'transformer' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = make_model(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('transformer', mu.size(), out.size())
