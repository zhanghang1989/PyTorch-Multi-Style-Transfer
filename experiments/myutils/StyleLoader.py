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
from torch.autograd import Variable

from myutils import utils

class StyleLoader():
	def __init__(self, style_folder, style_size, cuda=True):
		self.folder = style_folder
		self.style_size = style_size
		self.files = os.listdir(style_folder)
		self.cuda = cuda
	
	def get(self, i):
		idx = i%len(self.files)
		filepath = os.path.join(self.folder, self.files[idx])
		style = utils.tensor_load_rgbimage(filepath, self.style_size)	
		style = style.unsqueeze(0)
		style = utils.preprocess_batch(style)
		if self.cuda:
			style = style.cuda()
		style_v = Variable(style, requires_grad=False)
		return style_v

	def size(self):
		return len(self.files)


