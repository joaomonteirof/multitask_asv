from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
import glob

from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Cifar10 Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])

	validset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	labels_list = [x[1] for x in validset]

	if args.model == 'vgg':
		model = vgg.VGG('VGG16', sm_type=args.softmax)
	elif args.model == 'resnet':
		model = resnet.ResNet18(sm_type=args.softmax)
	elif args.model == 'densenet':
		model = densenet.densenet_cifar(sm_type=args.softmax)

	cp_list = glob.glob(args.cp_path+'*.pt')

	idxs_enroll, idxs_test, labels = create_trials_labels(labels_list)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	best_model, best_eer = None, float('inf')

	for cp in cp_list:

		ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
		try:
			model.load_state_dict(ckpt['model_state'], strict=True)
		except:
			print('\nSkipping model {}'.format(cp.split('/')[-1]))
			continue

		if args.cuda:
			device = get_freer_gpu()
			model = model.cuda(device)

		cos_scores = []
		out_cos = []

		mem_embeddings = {}

		model.eval()

		with torch.no_grad():

			for i in range(len(labels)):

				enroll_ex = str(idxs_enroll[i])

				try:
					emb_enroll = mem_embeddings[enroll_ex]
				except KeyError:

					enroll_ex_data = validset[idxs_enroll[i]][0].unsqueeze(0)

					if args.cuda:
						enroll_ex_data = enroll_ex_data.cuda(device)

					emb_enroll = model.forward(enroll_ex_data).detach()
					mem_embeddings[str(idxs_enroll[i])] = emb_enroll

				test_ex = str(idxs_test[i])

				try:
					emb_test = mem_embeddings[test_ex]
				except KeyError:

					test_ex_data = validset[idxs_test[i]][0].unsqueeze(0)

					if args.cuda:
						test_ex_data = test_ex_data.cuda(device)

					emb_test = model.forward(test_ex_data).detach()
					mem_embeddings[str(idxs_test[i])] = emb_test

				cos_scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )

				out_cos.append([str(idxs_enroll[i]), str(idxs_test[i]), cos_scores[-1]])

		cos_scores = np.asarray(cos_scores)
		labels = np.asarray(labels)
		model_id = cp.split('/')[-1]
		eer, auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
		print('\nCOS eval of model {}:'.format(model_id))
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}\n'.format(eer, auc, avg_precision, acc, threshold))
		if eer<best_eer:
			best_model, best_eer = model_id, eer

		print('Best model and corresponding eer: {} - {}'.format(best_model, best_eer))
