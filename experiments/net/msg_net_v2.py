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

	transformer = Net(ngf=128)
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
			transformer.setTarget(style_v)

			style_v = utils.subtract_imagenet_mean_batch(style_v)
			features_style = vgg(style_v)
			gram_style = [utils.gram_matrix(y) for y in features_style]

			y = transformer(x)
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
				transformer.eval()
				transformer.cpu()
				save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
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
	content_image = utils.tensor_load_rgbimage(args.content_image)#, size=args.content_size, keep_asp=True)
	content_image = content_image.unsqueeze(0)
	style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
	style = style.unsqueeze(0)	
	style = utils.preprocess_batch(style)

	style_model = Net(ngf=128)
	style_model.load_state_dict(torch.load(args.model))

	if args.cuda:
		style_model.cuda()
		content_image = content_image.cuda()
		style = style.cuda()

	style_v = Variable(style, volatile=True)

	content_image = Variable(utils.preprocess_batch(content_image))
	style_model.setTarget(style_v)

	output = style_model(content_image)
	utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


class Net(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn2.InstanceNormalization, n_blocks=6, gpu_ids=[]):
		super(Net, self).__init__()
		self.gpu_ids = gpu_ids
		self.gram = nn2.GramMatrix()

		block = nn2.Bottleneck
		upblock = nn2.UpBottleneck
		expansion = 4

		model1 = []
		model1 += [nn2.ConvLayer(input_nc, 64, kernel_size=7, stride=1),
							norm_layer(64),
							nn.ReLU(inplace=True),
							block(64, 32, 2, 1, norm_layer),
							block(32*expansion, ngf, 2, 1, norm_layer)]
		self.model1 = nn.Sequential(*model1)

		model = []
		self.ins = Inspiration(ngf*expansion)
		model += [self.model1]
		model += [self.ins]	

		for i in range(n_blocks):
			model += [block(ngf*expansion, ngf, 1, None, norm_layer)]
		
		model += [upblock(ngf*expansion, 32, 2, norm_layer),
							upblock(32*expansion, 16, 2, norm_layer),
							norm_layer(16*expansion),
							nn.ReLU(inplace=True),
							nn2.ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1)]

		self.model = nn.Sequential(*model)

	def setTarget(self, Xs):
		F = self.model1(Xs)
		G = self.gram(F)
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
		self.G = Variable(torch.Tensor(B,C,C), requires_grad=True)
		self.C = C
		self.reset_parameters()

	def reset_parameters(self):
		self.weight.data.uniform_(0.0, 0.02)

	def setTarget(self, target):
		self.G = target
		#target.view_as(self.G).detach().data.clone()

	def forward(self, X):
		# input X is a 3D feature map
		self.P = torch.bmm(self.weight.expand_as(self.G),self.G)
		return torch.bmm(self.P.transpose(1,2).expand(X.size(0), self.C, self.C), X.view(X.size(0),X.size(1),-1)).view_as(X)

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ 'N x ' + str(self.C) + ')'
