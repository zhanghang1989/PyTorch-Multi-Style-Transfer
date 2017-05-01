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

import utils
from option import Options
from StyleLoader import StyleLoader
from vgg16 import Vgg16

def main():
	"""
	For extending the package:
		1. Extending a new network type:
			1). define your own file under the folder 'net/' 
			2). implement your own nn.Module in mynn.py if need
			3). implement train and evaluate functions base on your need
			4). set up the import in the follows
		2. Extending a new experiment (changing the options)
			1). define the subcommand of the options
			2). implement the experiment function like optimize()
			3). set up the experiment as follows
	"""
	# figure out the experiments type
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	if args.subcommand == "train":
		# Training the model of user defined net-type
		if args.net_type == "v1":
			from net import msg_net_v1 as exp	
		else:
			raise ValueError('Unknow net-type')
		exp.train(args)

	elif args.subcommand == 'eval':
		# Test the pre-trained model
		if args.net_type == "v1":
			from net import msg_net_v1 as exp	
		else:
			raise ValueError('Unknow net-type')
		exp.evaluate(args)

	elif args.subcommand == 'optim':
		# Gatys et al. using optimization-based approach
		optimize(args)

	else:
		raise ValueError('Unknow experiment type')


def optimize(args):
	"""	Gatys et al. CVPR 2017
	ref: Image Style Transfer Using Convolutional Neural Networks
	"""
	# load the content and style target
	content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
	content_image = content_image.unsqueeze(0)
	content_image = Variable(utils.preprocess_batch(content_image), requires_grad=False)
	utils.subtract_imagenet_mean_batch(content_image)
	style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
	style_image = style_image.unsqueeze(0)	
	style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
	utils.subtract_imagenet_mean_batch(style_image)

	# load the pre-trained vgg-16 and extract features
	vgg = Vgg16()
	utils.init_vgg16(args.vgg_model_dir)
	vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
	if args.cuda:
		content_image = content_image.cuda()
		style_image = style_image.cuda()
		vgg.cuda()
	features_content = vgg(content_image)
	f_xc_c = Variable(features_content[1].data, requires_grad=False)
	features_style = vgg(style_image)
	gram_style = [utils.gram_matrix(y) for y in features_style]
	# init optimizer
	output = Variable(content_image.data, requires_grad=True)
	optimizer = Adam([output], lr=args.lr)
	mse_loss = torch.nn.MSELoss()
	# optimizing the images
	for e in range(args.iters):
		utils.add_imagenet_mean_batch(output)
		output.data.clamp_(0, 255)	
		utils.subtract_imagenet_mean_batch(output)

		optimizer.zero_grad()
		features_y = vgg(output)
		content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

		style_loss = 0.
		for m in range(len(features_y)):
			gram_y = utils.gram_matrix(features_y[m])
			gram_s = Variable(gram_style[m].data, requires_grad=False)
			style_loss += args.style_weight * mse_loss(gram_y, gram_s)

		total_loss = content_loss + style_loss

		if (e + 1) % args.log_interval == 0:
			print(total_loss.data.cpu().numpy()[0])
		total_loss.backward()
		
		optimizer.step()
	# save the image	
	utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


if __name__ == "__main__":
   main()
