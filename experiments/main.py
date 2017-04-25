##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from option import Options
from StyleLoader import StyleLoader
import utils
from hang import HangSNet
from vgg16 import Vgg16


def train(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		kwargs = {'num_workers': 0, 'pin_memory': False}
	else:
		kwargs = {}

	transform = transforms.Compose([transforms.Scale(args.image_size),
									transforms.CenterCrop(args.image_size),
									transforms.ToTensor(),
									transforms.Lambda(lambda x: x.mul(255))])
	train_dataset = datasets.ImageFolder(args.dataset, transform)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

	transformer = HangSNet()
	print(transformer)
	optimizer = Adam(transformer.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()

	vgg = Vgg16()
	utils.init_vgg16(args.vgg_model_dir)
	vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

	if args.cuda:
		transformer.cuda()
		vgg.cuda()

	style_loader = StyleLoader(args.style_folder, args.style_size)
	"""
	style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
	style = style.unsqueeze(0)
	style = utils.preprocess_batch(style)
	if args.cuda:
		style = style.cuda()
	style_v = Variable(style, volatile=True)
	utils.subtract_imagenet_mean_batch(style_v)
	"""

	for e in range(args.epochs):
		transformer.train()
		agg_content_loss = 0.
		agg_style_loss = 0.
		count = 0
		for batch_id, (x, _) in enumerate(train_loader):
			n_batch = len(x)
			count += n_batch
			optimizer.zero_grad()
			x = Variable(utils.preprocess_batch(x))
			if args.cuda:
				x = x.cuda()

			style_v = style_loader.get(batch_id)
			utils.subtract_imagenet_mean_batch(style_v)
			features_style = vgg(style_v)
			gram_style = [utils.gram_matrix(y) for y in features_style]

			transformer.setTarget(Variable(gram_style[2].data, requires_grad=False))

			y = transformer(x)
			xc = Variable(x.data.clone(), volatile=True)

			utils.subtract_imagenet_mean_batch(y)
			utils.subtract_imagenet_mean_batch(xc)

			features_y = vgg(y)
			features_xc = vgg(xc)

			f_xc_c = Variable(features_xc[1].data, requires_grad=False)

			content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

			style_loss = 0.
			for m in range(len(features_y)):
				gram_y = utils.gram_matrix(features_y[m])
				gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)
				style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

			total_loss = content_loss + style_loss
			total_loss.backward()
			optimizer.step()

			agg_content_loss += content_loss.data[0]
			agg_style_loss += style_loss.data[0]

			if (batch_id + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
					time.ctime(), e + 1, count, len(train_dataset),
								  agg_content_loss / (batch_id + 1),
								  agg_style_loss / (batch_id + 1),
								  (agg_content_loss + agg_style_loss) / (batch_id + 1)
				)
				print(mesg)

			
			if (batch_id + 1) % (4 * args.log_interval) == 0:
				# save model
				transformer.eval()
				transformer.cpu()
				save_model_filename = "Epoch_" + str(e) +  "iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
					args.content_weight) + "_" + str(args.style_weight) + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(transformer.state_dict(), save_model_path)
				transformer.train()
				transformer.cuda()
				print("\nCheckpoint, trained model saved at", save_model_path)

	# save model
	transformer.eval()
	transformer.cpu()
	save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
		args.content_weight) + "_" + str(args.style_weight) + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(transformer.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


def evaluate(args):
	content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
	content_image = content_image.unsqueeze(0)
	style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
	style = style.unsqueeze(0)	
	style = utils.preprocess_batch(style)

	vgg = Vgg16()
	utils.init_vgg16(args.vgg_model_dir)
	vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

	style_model = HangSNet()
	style_model.load_state_dict(torch.load(args.model))

	if args.cuda:
		style_model.cuda()
		vgg.cuda()
		content_image = content_image.cuda()
		style = style.cuda()

	style_v = Variable(style, volatile=True)
	utils.subtract_imagenet_mean_batch(style_v)
	features_style = vgg(style_v)
	gram_style = [utils.gram_matrix(y) for y in features_style]

	content_image = Variable(utils.preprocess_batch(content_image))
	target = Variable(gram_style[2].data, requires_grad=False)
	style_model.setTarget(target)

	output = style_model(content_image)
	utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def main():
	args = Options().parse()
	if args.subcommand is None:
		print("ERROR: specify either train or eval")
		sys.exit(1)

	if args.cuda and not torch.cuda.is_available():
		print("ERROR: cuda is not available, try running on CPU")
		sys.exit(1)

	if args.subcommand == "train":
		check_paths(args)
		train(args)
	else:
		evaluate(args)

main()
