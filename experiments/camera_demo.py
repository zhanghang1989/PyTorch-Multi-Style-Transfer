import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
import utils
from utils import StyleLoader

def run_demo(args, mirror=False):
	style_model = Net(ngf=args.ngf)
	style_model.load_state_dict(torch.load(args.model))
	style_model.eval()
	if args.cuda:
		style_loader = StyleLoader(args.style_folder, args.style_size)
		style_model.cuda()
	else:
		style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
	height =  args.demo_size
	width = int(4.0/3*args.demo_size)
	swidth = int(width/4)
	sheight = int(height/4)
	if args.record:
		fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
	cam = cv2.VideoCapture(0)
	cam.set(3, width)
	cam.set(4, height)
	key = 0
	idx = 0
	while True:
		# read frame
		idx += 1
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, 1)
		cimg = img.copy()
		img = np.array(img).transpose(2, 0, 1)
		# changing style 
		if idx%20 == 1:
			style_v = style_loader.get(int(idx/20))
			style_v = Variable(style_v.data, volatile=True)
			style_model.setTarget(style_v)

		img=torch.from_numpy(img).unsqueeze(0).float()
		if args.cuda:
			img=img.cuda()

		img = Variable(img, volatile=True)
		img = style_model(img)

		if args.cuda:
			simg = style_v.cpu().data[0].numpy()
			img = img.cpu().clamp(0, 255).data[0].numpy()
		else:
			simg = style_v.data().numpy()
			img = img.clamp(0, 255).data[0].numpy()
		img = img.transpose(1, 2, 0).astype('uint8')
		simg = simg.transpose(1, 2, 0).astype('uint8')

		# display
		simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
		cimg[0:sheight,0:swidth,:]=simg
		img = np.concatenate((cimg,img),axis=1)
		cv2.imshow('MSG Demo', img)
		#cv2.imwrite('stylized/%i.jpg'%idx,img)
		key = cv2.waitKey(1)
		if args.record:
			out.write(img)
		if key == 27: 
			break
	cam.release()
	if args.record:
		out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)

if __name__ == '__main__':
	main()
