import cloud_loader
import h5py
import sys
import numpy as np

from cloud_params import *
from nnet_toolkit import nnet

inputsize = 7*patchsize**2

offset = (patchsize-1)/2;

#d = cloud_loader.load_all_data('../cloudmasks/mat_training_notime')
d = cloud_loader.load_all_data(data_path)

#d is a tuple containing (A_list,MASK_list,MASKF_list)

sample_list = []
class_list = [];
for i in range(len(d[0])):
	sys.stdout.write('\nsampling image: '+ str(i));
	A = d[0][0]
	MASK = d[1][0]
	MASKF = d[2][0]
	#remove the first element from the list and from memory
	#to save memory as we go.
	del d[0][0]
	del d[1][0]
	del d[2][0]

	imsize_x = A.shape[1];
	imsize_y = A.shape[2];
	for x in range(offset,imsize_x-offset):
		if(x%20 == 0):
			sys.stdout.write('.');
			sys.stdout.flush();
		for y in range(offset,imsize_y-offset):
			if(np.random.rand() > load_percentage):
				continue;

			sample = A[:,x-offset:x+offset+1,y-offset:y+offset+1]
			sample = np.reshape(sample,inputsize)
			sample_list.append(sample)
			c = MASK[x,y];
			class_max = np.zeros(4);
			class_max[0] = c&1;
			class_max[1] = (c&8)>>3;
			class_max[2] = (c&16)>>4;
			class_max[3] = (c&128)>>7;
			class_list.append(class_max);


del A
del MASKF
del MASK

print('\nreshaping...')
sample_list = np.array(sample_list)
class_list = np.array(class_list)

print('calculating mean and std...')
sample_mean = np.mean(sample_list,0)
sample_std = np.std(sample_list,0)

print('normalizing data...')
#this works because of numpy's broadcasting rules
sample_list = sample_list - sample_mean
sample_list = sample_list/sample_std

print('shuffling data...')
#we need to shuffle both class and sample together in unison
#to do this we reset the random number generator state
rng_state = np.random.get_state()
np.random.shuffle(sample_list)
np.random.set_state(rng_state)
np.random.shuffle(class_list)

print('total number of samples: ' + str(sample_list.shape[0]))
print('creating testing and validation set')
data_size = sample_list.shape[0];

test_size = int(data_size*test_percent)
validation_size = int(data_size*validation_percent)

[sample_list_test,sample_list_validation,sample_list] = np.split(sample_list,[test_size,test_size+validation_size])
[class_list_test,class_list_validation,class_list] = np.split(class_list,[test_size,test_size+validation_size])

train_size = sample_list.shape[0]

print('training size: ' + str(sample_list.shape[0]))
print('test size: ' + str(sample_list_test.shape[0]))
print('validation size: ' + str(sample_list_validation.shape[0]))

print('initializing network...')
layers = [nnet.layer(inputsize)]

for i in range(len(hidden_sizes)):
	l = hidden_sizes[i]
	a = hidden_activations[i]
	layers.append(nnet.layer(l,a))

layers.append(nnet.layer(4,'squash'))

net = nnet.net(layers,step_size=step_size,dropout=dropout_percentage)

for i in range(training_epochs):
	minibatch_count = int(train_size/minibatch_size)
	#loop thru minibatches
	training_correct = 0;
	sys.stdout.write('.')
	sys.stdout.flush();
	rng_state = np.random.get_state()
	np.random.shuffle(sample_list)
	np.random.set_state(rng_state)
	np.random.shuffle(class_list)
	sys.stdout.write('.')
	sys.stdout.flush();

	for j in range(minibatch_count+1):
		#grab a minibatch
		net.input = np.transpose(sample_list[j*minibatch_size:(j+1)*minibatch_size])
		classification = class_list[j*minibatch_size:(j+1)*minibatch_size]
		#print(str(net.input.shape[0]) + ' ' + str(j*minibatch_size) + ' ' + str((j+1)*minibatch_size))
		#feed forward
		net.feed_forward();
		#calculate error & back propagate
		net.error = net.output - np.transpose(classification)
		#calculate # of correct classifications
		guess = np.argmax(net.output,0)
		c = np.argmax(classification,1);
		training_correct = training_correct + np.sum(c == guess)	
		net.back_propagate();
		#update weights
		net.update_weights();
	#calculate test error
	net.train = False
	net.input = np.transpose(sample_list_test)
	net.feed_forward()
	guess = np.argmax(net.output,0)
	c = np.argmax(class_list_test,1);
	test_correct = np.sum(c == guess)
	net.train = True
	#calculate (and print) test error
	print('epoch ' + str(i) + ': training rate : ' + str(float(training_correct)/float(train_size)) + \
			' test rate: ' + str(float(test_correct)/float(sample_list_test.shape[0])))

import pdb; pdb.set_trace()
