import cloud_fcns as cloud
from cloud_fcns import writedot
import h5py
import sys
import numpy as np
import gc
import time

use_sample_storage = True  #off by default - below import may turn on
from cloud_params import *
from nnet_toolkit import nnet

np.random.seed = randomseed1;

inputsize = 7*patchsize**2

offset = (patchsize-1)/2;

d = cloud.load_all_data(data_path)

estimated_load_percentage = np.min(class_load_percentage)
estimated_size = int(1000.0*1000.0*44.0*2.4*estimated_load_percentage)
sample_list = np.zeros((estimated_size,inputsize),dtype=np.float32)
class_list = np.zeros((estimated_size,3),dtype=np.int8)
#class_list = []
sample_list_test = []
class_list_test = []

samples_stored = 0

print('estimated size: ' + str(estimated_size))
for i in range(len(d[0])):
	A = d[0][0]
	MASK = d[1][0]
	MASKF = d[2][0]
	fname = d[3][0]
	sys.stdout.write('\nsampling image '+ str(i) + ': ' + str(fname));
	#remove the first element from the list and from memory
	#to save memory as we go.
	del d[0][0]
	del d[1][0]
	del d[2][0]
	del d[3][0]


	#test set comes from p31r43
	is_test = False
	if(fname.startswith('p31r43')):
		sys.stdout.write(' test set ')
		is_test = True

	if(not is_test):
		[sample_list_extend,class_list_extend] = cloud.sample_img(A,MASK,class_load_percentage)
		sample_list[samples_stored:samples_stored + len(sample_list_extend)] = sample_list_extend
		class_list[samples_stored:samples_stored + len(sample_list_extend)] = class_list_extend
		samples_stored = samples_stored + len(sample_list_extend);
		#sample_list.extend(sample_list_extend)
		#class_list.extend(class_list_extend)
		del sample_list_extend
		del class_list_extend
	else:
		[sample_list_test_extend,class_list_test_extend] = cloud.sample_img(A,MASK,class_test_load_percentage)
		sample_list_test.extend(sample_list_test_extend)
		class_list_test.extend(class_list_test_extend)
		del sample_list_test_extend
		del class_list_test_extend

	if(samples_stored > estimated_size):
		print('estimated sample size mismatch')
		raise
del A
del MASKF
del MASK

print('\nreshaping via temporary file...')
gc.collect()

sample_list = sample_list[0:samples_stored,:]
class_list = class_list[0:samples_stored,:]

sample_list_test = np.array(sample_list_test,dtype=np.float32)
class_list_test = np.array(class_list_test,dtype=np.int8)

gc.collect()
if pixel_mean is None:
	print('calculating mean and std... ' + str(sample_list.shape))
	sample_mean = np.mean(sample_list,0)
	sample_std = np.std(sample_list,0)
	print('mean: ' + str(sample_mean))
	print('std: ' + str(sample_std))
else:
	print('using precomputed means')
	pixel_mean = np.array(pixel_mean,dtype=np.float32)
	pixel_mean = pixel_mean.reshape((7,1,1))
	sample_mean = np.zeros((7,patchsize,patchsize),dtype=np.float32)
	sample_mean[:] = pixel_mean[:]
	sample_mean = sample_mean.reshape(inputsize)
	pixel_std = np.array(pixel_std,dtype=np.float32)
	pixel_std = pixel_std.reshape((7,1,1))
	sample_std = np.zeros((7,patchsize,patchsize),dtype=np.float32)
	sample_std[:] = pixel_std[:]
	sample_std = sample_std.reshape(inputsize)
	

print('normalizing data...')
sample_list = sample_list - sample_mean
sample_list = sample_list/sample_std

sample_list_test = sample_list_test - sample_mean
sample_list_test = sample_list_test/sample_std

tmean = np.mean(sample_list,0)
tstd = np.std(sample_list,0)

print('test mean: ' + str(tmean))
print('test std: ' + str(tstd))

print('shuffling data...')
#we need to shuffle both class and sample together in unison
#to do this we reset the random number generator state
rng_state = np.random.get_state()
np.random.shuffle(sample_list)
np.random.set_state(rng_state)
np.random.shuffle(class_list)

print('total number of samples: ' + str(sample_list.shape[0]))
print('creating testing set')

train_size = sample_list.shape[0]

print('training size: ' + str(sample_list.shape[0]))
print('test size: ' + str(sample_list_test.shape[0]))

class_type = np.sum(class_list,axis=0,dtype=np.float64)
print('class type count: ' + str(class_type))

class_type_test = np.sum(class_list_test,axis=0,dtype=np.float64)
print('class test type count: ' + str(class_type_test))

print('initializing network...')
layers = [nnet.layer(inputsize)]

for i in range(len(hidden_sizes)):
	l = hidden_sizes[i]
	a = hidden_activations[i]
	layers.append(nnet.layer(l,a))

layers.append(nnet.layer(3,'squash'))

net = nnet.net(layers,step_size=step_size,dropout=dropout_percentage)

print('beginning training...')
save_time = time.time()
epoch_time = time.time()
for i in range(training_epochs):
	minibatch_count = int(train_size/minibatch_size)
	#loop thru minibatches
	training_correct = 0;

	if(i%shuffle_epochs == 0):
		writedot()
		rng_state = np.random.get_state()
		np.random.shuffle(sample_list)
		np.random.set_state(rng_state)
		np.random.shuffle(class_list)
		writedot()

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
	test_rate =  str(float(test_correct)/float(sample_list_test.shape[0])) 
	print('epoch ' + str(i) + ': training rate : ' + str(float(training_correct)/float(train_size)) + \
			' test rate: ' + test_rate + ' time: ' + \
			str(time.time() - epoch_time))
	epoch_time = time.time()
	if(time.time() - save_time > save_interval):
		print('saving net...');
		cloud.save_net(net,i,test_rate,sample_mean,sample_std);
		save_time = time.time()
import pdb; pdb.set_trace()
