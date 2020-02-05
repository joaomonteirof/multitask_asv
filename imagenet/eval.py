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

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='MiniImagenet Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	parser.add_argument('--out-path', type=str, default=None, metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

	labels_list = [x[1] for x in validset]

	if args.model == 'vgg':
		model = vgg.VGG('VGG19')
	elif args.model == 'resnet':
		model = resnet.ResNet50()
	elif args.model == 'DenseNet121':
		model = densenet.densenet_cifar()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)
	else:
		device = torch.device('cpu')

	embeddings = []
	labels = []

	model.eval()

	iterator = tqdm(enumerate(valid_loader), total=len(valid_loader))

	with torch.no_grad():

		for i, batch in iterator:

			x, y = batch

			if args.cuda:
				x = x.to(device)

			emb = model.forward(x).detach()

			embeddings.append(emb.detach().cpu())
			labels.append(y)

	n_batches = i+1
	embeddings = torch.cat(embeddings, 0)
	labels = list(torch.cat(labels, 0).squeeze().numpy())

	print('\nEmbedding done')

	idxs_enroll, idxs_test, labels = create_trials_labels(labels)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	cos_scores = []
	out_cos = []

	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(range(0, len(labels), args.batch_size), total=n_batches)
		for i in :

			enroll_ex = idxs_enroll[i:(min(i+args.batch_size, len(labels)))]
			test_ex = idxs_test[i:(min(i+args.batch_size, len(labels)))]

			enroll_emb = embeddings[enroll_ex,:].to(device)
			test_emb = embeddings[test_ex,:].to(device)

			dist_cos = torch.nn.functional.cosine_similarity(enroll_emb, test_emb)
				
			for k in range(dist_cos.size(0)):
				cos_scores.append( dist_cos[k].item() )
				out_cos.append([str(idxs_enroll[i+k]), str(idxs_test[i+k]), cos_scores[-1]])

	print('\nScoring done')

	if args.out_path:
		with open(args.out_path+'cos_scores.out', 'w') as f:
			for el in out_cos:
				item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
				f.write("%s" % item)

	cos_scores = np.asarray(cos_scores)
	labels = np.asarray(labels)

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
	print('\nCOS eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))
