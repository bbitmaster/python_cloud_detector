import cloud_fcns as cloud
import h5py
import sys
import numpy as np

use_sample_storage = True  #off by default - below import may turn on
from cloud_params import *
from nnet_toolkit import nnet

np.random.seed = randomseed1;

inputsize = 7*patchsize**2

offset = (patchsize-1)/2;

d = cloud.load_all_data(data_path,2)
if(use_sample_storage):
	#d is a tuple containing (A_list,MASK_list,MASKF_list)
	print('creating h5py sample file: ' + cloud.get_sample_fname())
	cache_file = h5py.File(cloud.get_sample_fname(),'w')

	#40 images to load @ 1000x1000
	#cautiously allow a 3x margin of error
	estimated_sample_size = len(d[0])*1000.0*1000.0*load_percentage*3/batchsize
	sample_list_file = [];
	class_list_file = [];
	for i in range(num_batches):
		sample_list_file.append(cache_file.create_dataset('sample_list_' + str(i),(estimated_sample_size,inputsize),chunks=(10000,inputsize),dtype=np.float32))
		class_list_file.append(cache_file.create_dataset('class_list_' + str(i),(estimated_sample_size,3),chunks=(10000,3),dtype=np.uint8))

	sample_file_size = list(np.zeros(50));

sample_list = []
class_list = []
sample_list_validation = []
class_list_validation = []

for i in range(len(d[0])):
	sys.stdout.write('\nsampling image: '+ str(i));
	A = d[0][0]
	MASK = d[1][0]
	MASKF = d[2][0]
	fname = d[3][0]
	#remove the first element from the list and from memory
	#to save memory as we go.
	del d[0][0]
	del d[1][0]
	del d[2][0]
	del d[3][0]


	#validation set comes from p31r43
	is_validation = False
	if(fname.startswith('p31r43')):
		print('sampling validation')
		is_validation = True

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
			c = MASK[x,y];
			class_max = np.zeros(3);
			class_max[0] = c&1; #shadow
			class_max[1] = (c&8)>>3 | (c&16)>>4; #cloud (thick or thin)
			class_max[2] = (c&128)>>7; #clear sky
			if(not is_validation):
				sample_list.append(sample)
				class_list.append(class_max);
			else:
				sample_list_validation.append(sample)
				class_list_validation.append(class_max);
	#save sample_list for this image  h5py file
	if use_sample_storage:
		sample_list = np.array(sample_list)
		class_list = np.array(class_list)
		sample_size = sample_list.shape[0];

		#append peices of sample_list to all batches
		while sample_size > 0:
			chunk_size = 1000
			for b in range(num_batches):
				sample_list_file[b][sample_file_size[b]:new_sample_file_size,:] = sample_list
				class_list_file[sample_file_size:new_sample_file_size,:] = class_list
				sample_file_size = new_sample_file_size
				sample_list = []
				class_list = []
del A
del MASKF
del MASK

if use_sample_storage:
	print('\nreshaping h5py...')
	sample_list_file.resize((sample_file_size,inputsize))
	class_list_file.resize((sample_file_size,3))

	print('calculating mean and std from h5py dataset')
	sample_mean = np.mean(sample_list_file,0)
	sample_std = np.std(sample_list_file,0)
	
	print('normalizing data...')
	sample_list_file[:] = sample_list_file[:] - sample_mean
	sample_list_file[:] = sample_list_file[:]/sample_std
	
	print('shuffling data h5py... (could be slow)')
	#we need to shuffle both class and sample together in unison
	#to do this we reset the random number generator state
	rng_state = np.random.get_state()
	np.random.shuffle(sample_list_file)
	np.random.set_state(rng_state)
	np.random.shuffle(class_list_file)

	print('total number of samples: ' + str(sample_list_file.shape[0]))
	print('creating testing set')
	data_size = sample_list_file.shape[0];

	test_size = int(data_size*test_percent)

	#get test data off of end of sample list
	sample_list_test = sample_list_file[data_size - test_size:data_size]
	sample_list_file.resize((data_size - test_size,inputsize))
	
	class_list_test = class_list_file[data_size - test_size:data_size]
	class_list_file.resize((data_size - test_size,3))

	train_size = sample_list_file.shape[0]

	sample_list_validation = np.array(sample_list_validation);
	print('training size: ' + str(sample_list_file.shape[0]))
	print('test size: ' + str(sample_list_test.shape[0]))
	print('validation size: ' + str(sample_list_validation.shape[0]))
	#save more stuff to the h5py file

	print('saving stuff to h5py')
	f['sample_mean'] = sample_mean
	f['sample_std'] = sample_std
	f['sample_list_test'] = sample_list_test
	f['class_list_test'] = class_list_test
	f['sample_list_validation'] = sample_list_validation
	f['class_list_validation'] = class_list_validation

	print('getting initial batch')
	if(batch_size > train_size):
		batch_size = train_size
	sample_list = np.array(sample_list_file[0:batch_size])
	class_list = np.array(class_list_file[0:batch_size])
else:
	print('\nreshaping...')
	sample_list = np.array(sample_list)
	class_list = np.array(class_list)
	
	print('calculating mean and std...')
	sample_mean = np.mean(sample_list,0)
	sample_std = np.std(sample_list,0)

	print('normalizing data...')
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
	print('creating testing set')
	data_size = sample_list.shape[0];

	test_size = int(data_size*test_percent)

	[sample_list_test,sample_list] = np.split(sample_list,[test_size])
	[class_list_test,class_list] = np.split(class_list,[test_size])

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
