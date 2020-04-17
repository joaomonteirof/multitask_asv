from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *
from sklearn.metrics import pairwise_distances

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Retrieval Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-path', type=str, default=None, metavar='Path', help='Path to output embeddings.')
	parser.add_argument('--emb-path', type=str, default=None, metavar='Path', help='Path to precomputed embedding.')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--k-list', nargs='+', required=True, help='List of k values for R@K computation')
	parser.add_argument('--stats', choices=['cars', 'cub', 'sop', 'imagenet'], default='imagenet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
	if type(args.k_list[0]) is str:
		args.k_list = [int(x) for x in args.k_list[0].split(',')]

	print(args)

	if args.stats=='cars':
		mean, std = [0.4461, 0.4329, 0.4345], [0.2888, 0.2873, 0.2946]
	elif args.stats=='cub':
		mean, std = [0.4782, 0.4925, 0.4418], [0.2330, 0.2296, 0.2647]
	elif args.stats=='sop':
		mean, std = [0.5603, 0.5155, 0.4796], [0.2939, 0.2991, 0.3085]
	elif args.stats=='imagenet':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

	transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

	r_at_k_cos = {'R@'+str(x):0 for x in args.k_list}

	if args.model == 'vgg':
		model = vgg.VGG('VGG19')
	elif args.model == 'resnet':
		model = resnet.ResNet50()
	elif args.model == 'DenseNet121':
		model = densenet.DenseNet121()

	if args.pretrained:
		print('\nLoading pretrained encoder from torchvision\n')
		if args.model == 'vgg':
			model_pretrained = torchvision.models.vgg19(pretrained=True)
		elif args.model == 'resnet':
			model_pretrained = torchvision.models.resnet50(pretrained=True)
		elif args.model == 'densenet':
			model_pretrained = torchvision.models.densenet121(pretrained=True)

		model.load_state_dict(model_pretrained.state_dict(), strict=False)

	else:
		print('\nLoading pretrained model from {}\n'.format(args.cp_path))
		ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
		model.load_state_dict(model_pretrained.state_dict(), strict=False)

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)
	else:
		device = torch.device('cpu')

	if args.emb_path:

		emb_labels = torch.load(args.emb_path)
		embeddings, labels = emb_labels['embeddings'], emb_labels['labels']
		del emb_labels
		emb_labels = None

		print('\nEmbeddings loaded')

	else:

		embeddings = []
		labels = []

		iterator = tqdm(valid_loader, total=len(valid_loader))

		with torch.no_grad():

			for batch in iterator:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x)[0].detach()

				embeddings.append(emb.detach().cpu())
				labels.append(y)

		embeddings = torch.cat(embeddings, 0)
		labels = list(torch.cat(labels, 0).squeeze().numpy())

		if args.out_path:
			if os.path.isfile(args.out_path):
				os.remove(args.out_path)
				print(args.out_path+' Removed')
			torch.save({'embeddings':embeddings, 'labels':labels}, args.out_path)

		print('\nEmbedding done')

	with torch.no_grad():

		iterator = tqdm(enumerate(labels), total=len(labels))
		for i, label in iterator:

			enroll_emb = embeddings[i].unsqueeze(0).to(device)

			cos_scores = []

			for j in range(0, len(labels), args.batch_size):

				test_emb = embeddings[j:(min(j+args.batch_size, len(embeddings))),:].to(device)
				enroll_emb_repeated = enroll_emb.repeat(test_emb.size(0), 1)

				dist_cos = torch.nn.functional.cosine_similarity(enroll_emb_repeated, test_emb)
				
				for l in range(dist_cos.size(0)):

					if i==(j+l): continue ## skip same example

					cos_scores.append( [dist_cos[l].item(), labels[j+l]] )

			sorted_cos_classes = np.array(sorted(cos_scores, reverse=True))[:,1]

			for k in args.k_list:
				if label in sorted_cos_classes[:k]:
					r_at_k_cos['R@'+str(k)]+=1


	print('\nScoring done')

for k in args.k_list:
	r_at_k_cos['R@'+str(k)]/=(len(labels)-1)

print('\nCOS R@k:')
print(r_at_k_cos)
print('\n')
