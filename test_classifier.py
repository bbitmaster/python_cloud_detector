import cloud_fcns as cloud
from cloud_fcns import writedot
import numpy as np
import cv2
#import shelve
import sys
import os
from nnet_toolkit import nnet

#sh = shelve.open('/local_scratch/cloudmasks/h19_net.out');

#old_net = sh['net']
#sample_mean = sh['sample_mean']
#sample_std = sh['sample_std']
#patchsize = sh['patchsize']
#sh.close()
imgdir = '/local_scratch/cloudmasks/images'
netfilename = 'net_python_cloud_detector.mat'
data = cloud.load_net(netfilename)

net = data['net']
sample_mean = data['sample_mean']
sample_std = data['sample_std']
patchsize = data['patchsize']

offset = (patchsize-1)/2;
inputsize = 7*patchsize**2

#layer = old_net.layer;
#layer.insert(0,nnet.layer(inputsize))
#net = nnet.net(layer);

d = cloud.load_all_data('/local_scratch/cloudmasks/mat_training_notime',amount=4)

for i in range(len(d[0])):
	sys.stdout.write('\ntesting image: '+ str(i));
	A = d[0][i]
	MASK = d[1][i]
	MASKF = d[2][i]

	blank = np.zeros((1000,1000,3),np.uint8);
	
	imsize_x = A.shape[1];
	imsize_y = A.shape[2];
	for x in range(offset,imsize_x-offset):
		if(x%20 == 0):
			writedot()
		for y in range(offset,imsize_y-offset):
			sample = A[:,x-offset:x+offset+1,y-offset:y+offset+1]
			sample = np.reshape(sample,(1,inputsize))
			
			sample = sample - np.transpose(sample_mean)
			sample = sample/np.transpose(sample_std)
			net.input = np.transpose(sample)
			net.feed_forward()
			guess = np.argmax(net.output,0)
			#clss = MASK[x,y]
			#if(clss&1):
			#	clss = 0
			#if(clss&8):
		#		clss = 1
		#	if(clss&16):
		#		clss = 2
		#	if(clss&128):
		#		clss = 3
		#	if(guess == 3
			if(guess == 0):
				c = (255,0,0)
			elif(guess == 1):
				c = (0,255,0)
			elif(guess == 2):
				c = (0,0,255)
			elif(guess == 3):
				c = (0,255,255)
			blank[x,y] = c
	cv2.imwrite(os.path.join(imgdir,'img_' + str(i) + '.png'),blank)

