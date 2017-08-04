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
import torch.nn as nn

import net.mynn as nn2
from myutils import utils
from myutils.vgg16 import Vgg16 
from myutils.StyleLoader import StyleLoader


def train(args):
	check_paths(args)
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

	style_model = Net()
	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		style_model.load_state_dict(torch.load(args.resume))

	print(style_model)
	optimizer = Adam(style_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()

	vgg = Vgg16()
	utils.init_vgg16(args.vgg_model_dir)
	vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

	if args.cuda:
		style_model.cuda()
		vgg.cuda()

	style_loader = StyleLoader(args.style_folder, args.style_size)

	for e in range(args.epochs):
		style_model.train()
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
			style_v = utils.subtract_imagenet_mean_batch(style_v)
			features_style = vgg(style_v)
			gram_style = [utils.gram_matrix(y) for y in features_style]
			style_model.setTarget(gram_style[2].data)

			y = style_model(x)
			xc = Variable(x.data.clone(), volatile=True)

			y = utils.subtract_imagenet_mean_batch(y)
			xc = utils.subtract_imagenet_mean_batch(xc)

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
				style_model.eval()
				style_model.cpu()
				save_model_filename = "Epoch_" + str(e) +  "iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
					args.content_weight) + "_" + str(args.style_weight) + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(style_model.state_dict(), save_model_path)
				style_model.train()
				style_model.cuda()
				print("\nCheckpoint, trained model saved at", save_model_path)

	# save model
	style_model.eval()
	style_model.cpu()
	save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
		args.content_weight) + "_" + str(args.style_weight) + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(style_model.state_dict(), save_model_path)

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

	style_model = Net()
	style_model.load_state_dict(torch.load(args.model))

	if args.cuda:
		style_model.cuda()
		vgg.cuda()
		content_image = content_image.cuda()
		style = style.cuda()

	style_v = Variable(style, volatile=True)
	style_v = utils.subtract_imagenet_mean_batch(style_v)
	features_style = vgg(style_v)
	gram_style = [utils.gram_matrix(y) for y in features_style]

	content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
	style_model.setTarget(gram_style[2].data)

	output = style_model(content_image)
	utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


class Net(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9, gpu_ids=[]):
		super(Net, self).__init__()
		self.gpu_ids = gpu_ids

		# make bottleneck as an option
		block = nn2.Bottleneck
		upblock = nn2.UpBottleneck
		expansion = 4

		model = []
		model += [nn2.ConvLayer(input_nc, 64, kernel_size=7, stride=1),
							norm_layer(64),
							nn.ReLU(inplace=True),
							block(64, 32, 2, 1, norm_layer),
							block(32*expansion, ngf, 2, 1, norm_layer)]

		self.ins = Inspiration(ngf*expansion)
		model += [self.ins]
		for i in range(n_blocks):
			model += [block(ngf*expansion, ngf, 1, None, norm_layer)]

		model += [upblock(ngf*expansion, 32, 2, norm_layer),
							upblock(32*expansion, 16, 2, norm_layer),
							norm_layer(64),
							nn.ReLU(inplace=True),
							nn2.ConvLayer(64, output_nc, kernel_size=7, stride=1)]
		model += [nn.Tanh(),
							nn2.MultConst()]
		self.model = nn.Sequential(*model)

	def setTarget(self, G):
		self.ins.setTarget(G)

	def forward(self, input):
		return self.model(input)


class Inspiration(nn.Module):
	""" Inspiration Layer (from MSG-Net paper)
	tuning the featuremap with target Gram Matrix
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, C, B=1):
		super(Inspiration, self).__init__()
		# B is equal to 1 or input mini_batch
		self.weight = nn.Parameter(torch.Tensor(1,C,C), requires_grad=True)
		# non-parameter buffer
		self.register_buffer('G', torch.Tensor(B,C,C))
		self.C = C
		self.reset_parameters()

	def reset_parameters(self):
		self.weight.data.uniform_(0.0, 0.02)

	def setTarget(self, target):
		self.G = target.expand_as(self.G)

	def forward(self, X):
		# input X is a 3D feature map
		self.P = torch.bmm(self.weight.expand_as(self.G), Variable(self.G))
		return torch.bmm(self.P.transpose(1,2).expand(X.size(0), self.C, self.C), X.view(X.size(0),X.size(1),-1)).view_as(X)

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ 'N x ' + str(self.C) + ')'
