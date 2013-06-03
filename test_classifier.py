import cloud_fcns as cloud
from cloud_fcns import writedot
import numpy as np
import cv2
#import shelve
import sys
import os
from nnet_toolkit import nnet

num_classes = 3

#sh = shelve.open('/local_scratch/cloudmasks/h19_net.out');

#old_net = sh['net']
#sample_mean = sh['sample_mean']
#sample_std = sh['sample_std']
#patchsize = sh['patchsize']
#sh.close()
imgdir = '/local_scratch/cloudmasks/images'
netfilename = 'net_python_cloud_detector4.mat'
data = cloud.load_net(netfilename)

net = data['net']
sample_mean = np.transpose(data['sample_mean'])
sample_std = np.transpose(data['sample_std'])
patchsize = data['patchsize'][0][0]

offset = (patchsize-1)/2;
inputsize = 7*patchsize**2

#layer = old_net.layer;
#layer.insert(0,nnet.layer(inputsize))
#net = nnet.net(layer);

d = cloud.load_all_data('/local_scratch/cloudmasks/mat_training_notime')

confusion_matrix = {}
total_confusion_matrix = np.zeros((num_classes,num_classes))
test_confusion_matrix = np.zeros((num_classes,num_classes))
f_out = open(os.path.join(imgdir,'classification_printout.txt'),'w')

np.set_printoptions(suppress=True)
print('test rate for this dataset: ' + str(data['test_rate']))

for i in range(len(d[0])):
	A = d[0][i]
	MASK = d[1][i]
	MASKF = d[2][i]
	filename = d[3][i]
	[fname, fext] = os.path.splitext(filename)
	sys.stdout.write('\ntesting image '+ str(i) + ' ' + str(fname) + '\n');
	blank = np.zeros((1000,1000,3),np.uint8);
	confusion_matrix[fname] = np.zeros((num_classes,num_classes))

	imsize_x = A.shape[1];
	imsize_y = A.shape[2];
	net.train = False

	for x in range(offset,imsize_x-offset):
		if(x%20 == 0):
			writedot()
		for y in range(offset,imsize_y-offset):
			sample = A[:,x-offset:x+offset+1,y-offset:y+offset+1]
			
			sample = np.reshape(sample,inputsize)
			
			sample = sample - sample_mean
			sample = sample/sample_std
			net.input = np.transpose(sample)
			net.feed_forward()
			guess = np.argmax(net.output,0)[0]
			mask = MASK[x,y]
			#import pdb; pdb.set_trace()
			#clss=0
			if(mask&1):
				clss = 0
			if(mask&8):
				clss = 1
			if(mask&16):
				clss = 1
			if(mask&128):
				clss = 2
			#if(clss == -1):
			#	print('ERROR! Invalid class in mask file: ' + str(mask) + ' ' + str(x) + ' ' + str(y))
			if(fname.startswith('p31r43')):
				test_confusion_matrix[clss,guess] += 1
			else:
				total_confusion_matrix[clss,guess] += 1
			confusion_matrix[fname][clss,guess] += 1
			#generate pixel for mask image
			if(guess == 0):
				c = (64,64,64)
			elif(guess == 1):
				c = (128,128,128)
			elif(guess == 2):
				c = (0,0,0)
			blank[y,x] = c
	class_percent = np.sum(np.diag(confusion_matrix[fname]))/np.sum(confusion_matrix[fname])
	confusion_percent = confusion_matrix[fname]/np.sum(confusion_matrix[fname])
	print('\n' + str(confusion_matrix[fname]) + '\n' + str(confusion_percent) + '\n' + str(class_percent))
	f_out.write(fname + '\n' + str(confusion_matrix[fname]) + '\n' + str(confusion_percent) + '\n' + str(class_percent) + '\n')
	cv2.imwrite(os.path.join(imgdir,str(fname) + '_mask.png'),blank)

class_percent = np.sum(np.diag(total_confusion_matrix))/np.sum(total_confusion_matrix)
confusion_percent = total_confusion_matrix/np.sum(total_confusion_matrix)
print(str(total_confusion_matrix) + '\n' + str(confusion_percent) + '\n' + str(class_percent))
f_out.write('\ntotal_confusion_matrix:\n' + str(total_confusion_matrix) + '\n' + str(confusion_percent) + '\n' + str(class_percent))

class_percent = np.sum(np.diag(test_confusion_matrix))/np.sum(test_confusion_matrix)
confusion_percent = test_confusion_matrix/np.sum(test_confusion_matrix)
print(str(test_confusion_matrix) + '\n' + str(confusion_percent) + '\n' + str(class_percent))
f_out.write('\ntest_confusion_matrix:\n' + str(test_confusion_matrix) + '\n' + str(confusion_percent) + '\n' + str(class_percent))
