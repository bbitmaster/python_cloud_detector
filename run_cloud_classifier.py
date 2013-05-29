import cloud_fcns as cloud
from cloud_fcns import writedot
import h5py
import sys
import numpy as np
import time

use_sample_storage = True  #off by default - below import may turn on
from cloud_params import *
from nnet_toolkit import nnet

np.random.seed = randomseed1;

inputsize = 7*patchsize**2

offset = (patchsize-1)/2;

d = cloud.load_all_data(data_path)
load_cache = False
if(use_sample_storage):
	#d is a tuple containing (A_list,MASK_list,MASKF_list)
	print('creating h5py sample file: ' + cloud.get_sample_fname())
	cache_file = h5py.File(cloud.get_sample_fname(),'a')

	if('final' in cache_file):
		load_cache = True
	else:
		#we may have a file that wasn't finished saving... need to truncate
		cache_file.close()
		cache_file = h5py.File(cloud.get_sample_fname(),'w')

		#40 images to load @ 1000x1000
		#cautiously allow a 3x margin of error
		estimated_sample_size = int(len(d[0])*1000.0*1000.0*load_percentage*3/num_batches)
		sample_list_file = [];
		class_list_file = [];
		for i in range(num_batches):
			sample_list_file.append(cache_file.create_dataset('sample_list_' + str(i),(estimated_sample_size,inputsize),chunks=(chunk_size,inputsize),dtype=np.float32))
			class_list_file.append(cache_file.create_dataset('class_list_' + str(i),(estimated_sample_size,3),chunks=(chunk_size,3),dtype=np.uint8))

		sample_file_size = list(np.zeros(50,dtype=np.uint32));

if(load_cache):
	print('h5py file found. loading...')
	sample_list_file = [];
	class_list_file = [];
	for i in range(num_batches):
		sample_list_file.append(cache_file['sample_list_' + str(i)])
		class_list_file.append(cache_file['class_list_' + str(i)])
	sample_mean = np.array(cache_file['sample_mean'])
	sample_std = np.array(cache_file['sample_std'])
	sample_list_test = np.array(cache_file['sample_list_test'])
	class_list_test = np.array(cache_file['class_list_test'])
else:
	sample_list = []
	class_list = []
	sample_list_test = []
	class_list_test = []

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


		#test set comes from p31r43
		is_test = False
		if(fname.startswith('p31r43')):
			sys.stdout.write('test set')
			is_test = True

		if(not is_test):
			[sample_list,class_list] = cloud.sample_img(A,MASK,load_percentage)
		else:
			[sample_list_test_extend,class_list_test_extend] = cloud.sample_img(A,MASK,load_percentage)
			sample_list_test.extend(sample_list_test_extend)
			class_list_test.extend(class_list_test_extend)
			del sample_list_test_extend
			del class_list_test_extend

		#save sample_list for this image  h5py file
		if use_sample_storage:
			sample_list = sample_list
			class_list = class_list
			sample_size = len(sample_list);
			#sample_list = np.array(sample_list)
			#class_list = np.array(class_list)
			#sample_size = sample_list.shape[0];

			#append peices of sample_list to all batches
			chunk_size = chunk_append_size
			while sample_size > 0:
				if(chunk_size > sample_size):
					chunk_size = sample_size
				b = np.random.randint(num_batches)
				#if the below line crashes with "zero-length selections are not allowed"
				#it means the estimated sample size was too small
				sample_list_file[b][sample_file_size[b]:sample_file_size[b] + chunk_size,:] = sample_list[sample_size - chunk_size:sample_size]
				class_list_file[b][sample_file_size[b]:sample_file_size[b] + chunk_size,:] = class_list[sample_size - chunk_size:sample_size]
				sample_file_size[b] += chunk_size
				sample_size -= chunk_size
			del sample_list
			del class_list
			sample_list = []
			class_list = []
	del A
	del MASKF
	del MASK

	if use_sample_storage:
		sys.stdout.write('\nreshaping h5py')
		for i, s in enumerate(sample_file_size):
			sample_list_file[i].resize((sample_file_size[i],inputsize))
			class_list_file[i].resize((sample_file_size[i],3))
			writedot()

		sys.stdout.write('\ncalculating mean and std from h5py dataset')
		sample_mean = np.zeros(inputsize)
		sample_std = np.zeros(inputsize)
		for i in range(len(sample_file_size)):
			sample_list = np.array(sample_list_file[i])
			sample_mean += np.mean(sample_list,0)
			sample_std += np.std(sample_list,0)
			writedot()
		sample_mean /= len(sample_file_size)
		sample_std /= len(sample_file_size)

		sys.stdout.write('\nnormalizing data')
		for i in range(len(sample_file_size)):
			sample_list_file[i][:] = sample_list_file[i][:] - sample_mean
			sample_list_file[i][:] = sample_list_file[i][:]/sample_std
			writedot()
		
		sample_list_test = np.array(sample_list_test)
		class_list_test = np.array(class_list_test)

		sample_list_test = sample_list_test - sample_mean
		sample_list_test = sample_list_test/sample_std

		sys.stdout.write('\nshuffling data h5py (could be slow)')
		#we need to shuffle both class and sample together in unison
		#to do this we reset the random number generator state
		for i in range(len(sample_file_size)):
			sample_list = np.array(sample_list_file[i])
			class_list = np.array(class_list_file[i])
			rng_state = np.random.get_state()
			np.random.shuffle(sample_list)
			np.random.set_state(rng_state)
			np.random.shuffle(class_list)
			sample_list[:] = sample_list
			class_list[:] = class_list
			writedot()

		print('\ncreating test set')

		#print('training size: ' + str(sample_list_file.shape[0]))
		#print('test size: ' + str(sample_list_test.shape[0]))
		#print('validation size: ' + str(sample_list_validation.shape[0]))
		#save more stuff to the h5py file

		print('saving stuff to h5py')
		cache_file['sample_mean'] = sample_mean
		cache_file['sample_std'] = sample_std
		cache_file['sample_list_test'] = sample_list_test
		cache_file['class_list_test'] = class_list_test

		cache_file['final'] = np.array([1])
		print('getting initial batch')
		sample_list = np.array(sample_list_file[0])
		class_list = np.array(class_list_file[0])
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
		
		sample_list_test = sample_list_validation

		train_size = sample_list.shape[0]

		print('training size: ' + str(sample_list.shape[0]))
		print('validation size: ' + str(sample_list_validation.shape[0]))
import pdb; pdb.set_trace()

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
