from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_
from utils.utils import *

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['resnet_mfcc', 'resnet_34', 'resnet_lstm', 'resnet_qrnn', 'resnet_stats', 'resnet_large', 'resnet_small', 'resnet_2d', 'TDNN', 'TDNN_att', 'TDNN_multihead', 'TDNN_lstm', 'TDNN_aspp', 'TDNN_mod', 'all'], default='all', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--delta', action='store_true', default=False, help='Enables extra data channels')
parser.add_argument('--inner', action='store_true', default=False, help='Get embeddings from inner layer')
args = parser.parse_args()

if args.model == 'resnet_mfcc' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_mfcc', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_mfcc', mu.size(), out.size())
if args.model == 'resnet_34' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_34(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_34', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_34', mu.size(), out.size())
if args.model == 'resnet_lstm' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_lstm', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_lstm', mu.size(), out.size())
if args.model == 'resnet_qrnn' or args.model == 'all' and torch.cuda.is_available():
	device = get_freer_gpu()
	import cupy
	cupy.cuda.Device(int(str(device).split(':')[-1])).use()
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200).to(device)
	model = model_.ResNet_qrnn(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_qrnn', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_qrnn', mu.size(), out.size())
if args.model == 'resnet_stats' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_stats(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_stats', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_stats', mu.size(), out.size())
if args.model == 'resnet_large' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_large(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_large', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_large', mu.size(), out.size())
if args.model == 'resnet_small' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.ResNet_small(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_small', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_small', mu.size(), out.size())
if args.model == 'resnet_2d' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, 40, 200)
	model = model_.ResNet_2d(n_z=args.latent_size, delta=args.delta, proj_size=10, sm_type='softmax')
	mu = model.forward(batch, inner=args.inner)
	if args.inner:
		print('resnet_2d', mu.size())
	else:
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('resnet_2d', mu.size(), out.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.model = torch.nn.Sequential(*list(model.model.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN', mu.size(), out.size())
if args.model == 'TDNN_att' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN_att(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.post_pooling = torch.nn.Sequential(*list(model.post_pooling.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN_att', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN_att', mu.size(), out.size())
if args.model == 'TDNN_multihead' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN_multihead(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.post_pooling = torch.nn.Sequential(*list(model.post_pooling.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN_multihead', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN_multihead', mu.size(), out.size())
if args.model == 'TDNN_lstm' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN_lstm(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.post_pooling = torch.nn.Sequential(*list(model.post_pooling.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN_lstm', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN_lstm', mu.size(), out.size())
if args.model == 'TDNN_aspp' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN_aspp(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.post_pooling = torch.nn.Sequential(*list(model.post_pooling.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN_aspp', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN_aspp', mu.size(), out.size())
if args.model == 'TDNN_mod' or args.model == 'all':
	batch = torch.rand(3, 3 if args.delta else 1, args.ncoef, 200)
	model = model_.TDNN_mod(n_z=args.latent_size, ncoef=args.ncoef, delta=args.delta, proj_size=10, sm_type='softmax')
	if args.inner:
		model.model = torch.nn.Sequential(*list(model.model.children())[:-5])
		mu = model.forward(batch, inner=args.inner)
		print('TDNN_mod', mu.size())
	else:
		mu = model.forward(batch, inner=args.inner)
		out = model.out_proj(mu, torch.ones(mu.size(0)))
		print('TDNN_mod', mu.size(), out.size())
