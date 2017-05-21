##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

class MultConst(nn.Module):
	def forward(self, input):
		return 255*input


class GramMatrix(nn.Module):
	def forward(self, y):
		(b, ch, h, w) = y.size()
		features = y.view(b, ch, w * h)
		features_t = features.transpose(1, 2)
		gram = features.bmm(features_t) / (ch * h * w)
		return gram
	

class InstanceNormalization(nn.Module):
	"""InstanceNormalization
	Improves convergence of neural-style.
	ref: https://arxiv.org/pdf/1607.08022.pdf
	"""

	def __init__(self, dim, eps=1e-5):
		super(InstanceNormalization, self).__init__()
		self.weight = nn.Parameter(torch.FloatTensor(dim))
		self.bias = nn.Parameter(torch.FloatTensor(dim))
		self.eps = eps
		self._reset_parameters()

	def _reset_parameters(self):
		self.weight.data.uniform_()
		self.bias.data.zero_()

	def forward(self, x):
		n = x.size(2) * x.size(3)
		t = x.view(x.size(0), x.size(1), n)
		mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
		# Calculate the biased var. torch.var returns unbiased var
		var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
		scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		scale_broadcast = scale_broadcast.expand_as(x)
		shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		shift_broadcast = shift_broadcast.expand_as(x)
		out = (x - mean) / torch.sqrt(var + self.eps)
		out = out * scale_broadcast + shift_broadcast
		return out


class Basicblock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
		super(Basicblock, self).__init__()
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = nn.Conv2d(inplanes, planes,
														kernel_size=1, stride=stride)
		conv_block=[]
		conv_block+=[norm_layer(inplanes),
								nn.ReLU(inplace=True),
								ConvLayer(inplanes, planes, kernel_size=3, stride=stride),
								norm_layer(planes),
								nn.ReLU(inplace=True),
								ConvLayer(planes, planes, kernel_size=3, stride=1),
								norm_layer(planes)]
		self.conv_block = nn.Sequential(*conv_block)
	
	def forward(self, input):
		if self.downsample is not None:
			residual = self.residual_layer(input)
		else:
			residual = input
		return residual + self.conv_block(input)
			

class UpBasicblock(nn.Module):
	""" Up-sample residual block (from MSG-Net paper)
	Enables passing identity all the way through the generator
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
		super(UpBasicblock, self).__init__()
		self.residual_layer = UpsampleConvLayer(inplanes, planes,
 			 										kernel_size=1, stride=1, upsample=stride)
		conv_block=[]
		conv_block+=[norm_layer(inplanes),
								nn.ReLU(inplace=True),
								UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride),
								norm_layer(planes),
								nn.ReLU(inplace=True),
								ConvLayer(planes, planes, kernel_size=3, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)
	
	def forward(self, input):
		return self.residual_layer(input) + self.conv_block(input)


class Bottleneck(nn.Module):
	""" Pre-activation residual block
	Identity Mapping in Deep Residual Networks
	ref https://arxiv.org/abs/1603.05027
	"""
	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
														kernel_size=1, stride=stride)
		conv_block = []
		conv_block += [norm_layer(inplanes),
									nn.ReLU(inplace=True),
									nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									ConvLayer(planes, planes, kernel_size=3, stride=stride)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)
		
	def forward(self, x):
		if self.downsample is not None:
			residual = self.residual_layer(x)
		else:
			residual = x
		return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
	""" Up-sample residual block (from MSG-Net paper)
	Enables passing identity all the way through the generator
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
		super(UpBottleneck, self).__init__()
		self.expansion = 4
		self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
 			 										kernel_size=1, stride=1, upsample=stride)
		conv_block = []
		conv_block += [norm_layer(inplanes),
									nn.ReLU(inplace=True),
									nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, x):
		return  self.residual_layer(x) + self.conv_block(x)


class ConvLayer(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(ConvLayer, self).__init__()
		reflection_padding = int(np.floor(kernel_size / 2))
		self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		out = self.reflection_pad(x)
		out = self.conv2d(out)
		return out

class UpsampleConvLayer(torch.nn.Module):
	"""UpsampleConvLayer
	Upsamples the input and then does a convolution. This method gives better results
	compared to ConvTranspose2d.
	ref: http://distill.pub/2016/deconv-checkerboard/
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
		super(UpsampleConvLayer, self).__init__()
		self.upsample = upsample
		if upsample:
			self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
		self.reflection_padding = int(np.floor(kernel_size / 2))
		if self.reflection_padding != 0:
			self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		if self.upsample:
			x = self.upsample_layer(x)
		if self.reflection_padding != 0:
			x = self.reflection_pad(x)
		out = self.conv2d(x)
		return out
