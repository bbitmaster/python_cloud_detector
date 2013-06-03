import h5py
import numpy as np
import scipy.io
import os
import sys
from cloud_params import *
from nnet_toolkit import nnet

def writedot():
	sys.stdout.write('.')
	sys.stdout.flush()

#loads from a .mat file that must contain the variables
#'A' 'MASK' 'MASKF'
def load_data(filename):
	f = h5py.File(filename)
	A = np.array(f['A'])
	MASK = np.array(f['MASK'])
	MASKF = np.array(f['MASKF'])
	return (A,MASK,MASKF)

def load_all_data(dirname,amount = 1e99):
	A_list = []
	MASK_list = []
	MASKF_list = []
	fname_list = []
	i = amount
	for f in os.listdir(dirname):
		#skip p61r2 - it is polar region and seems to lack a mask
		if(f.startswith('p61r2')):
			print('skipping p61r2...');
			continue
		if(f.endswith('mat')):
			print('loading: ' + f)
			data = load_data(os.path.join(dirname,f))
			A_list.append(data[0])
			MASK_list.append(data[1])
			MASKF_list.append(data[2])
			fname_list.append(f)
		i = i - 1
		if(i == 0):
			break
	return(A_list,MASK_list,MASKF_list,fname_list);

def get_sample_fname():
	fname = 'cloud_sample_cache_' + str(patchsize) + '_' + str(load_percentage) + '_' + str(randomseed1) + '.hdf5'
	return os.path.join(sample_dir_name,fname)

def sample_img(A,MASK,class_percent):
	inputsize = 7*patchsize**2
	offset = (patchsize-1)/2;
	imsize_x = A.shape[1];
	imsize_y = A.shape[2];
	sample_list = []
	class_list = []
	classes = np.zeros(256,dtype=np.float32)
	classes[1] = 0
	classes[8] = 1
	classes[16] = 1
	classes[128] = 2
	for x in range(offset,imsize_x-offset):
		if(x%20 == 0):
			writedot()
		for y in range(offset,imsize_y-offset):
			#if(np.random.rand() > percent):
			#	continue;
			c = MASK[x,y];
			class_max = np.zeros(3,dtype=np.float32);
			
			class_max[0] = c&1; #shadow
			class_max[1] = (c&8)>>3 | (c&16)>>4; #cloud (thick or thin)
			class_max[2] = (c&128)>>7; #clear sky
			if(np.random.rand() > class_percent[int(classes[int(c)])]):
				continue;
			sample = A[:,x-offset:x+offset+1,y-offset:y+offset+1]
			sample = np.reshape(sample,inputsize)
			sample_list.append(sample)
			class_list.append(class_max);
	return (sample_list, class_list)

def save_net(net,epoch,test_rate,sample_mean,sample_std):
	matlabdict = {};
	#store layer info
	matlabdict['num_layers'] = len(net.layer)
	for i in range(len(net.layer)):
		matlabdict['layer_weights_' + str(i+1)] = net.layer[i].weights
		matlabdict['layer_activation_' + str(i+1)] = net.layer[i].activation
		matlabdict['layer_step_size_' + str(i+1)] = net.layer[i].step_size
		matlabdict['layer_dropout_' + str(i+1)] = str(net.layer[i].dropout)
		matlabdict['layer_node_count_input_' + str(i+1)] = net.layer[i].node_count_input
		matlabdict['layer_node_count_output_' + str(i+1)] = net.layer[i].node_count_output
	matlabdict['patchsize'] = patchsize
	matlabdict['epoch'] = epoch
	matlabdict['test_rate'] = test_rate
	matlabdict['sample_mean'] = sample_mean
	matlabdict['sample_std'] = sample_std
	scipy.io.savemat('net_' +  os.path.split(os.getcwd())[-1],matlabdict,oned_as='column');


def load_net(filename):
	matlabdict = {}
	scipy.io.loadmat(filename,matlabdict)
	net = nnet
	num_layers = matlabdict['num_layers'][0]
	layers = [nnet.layer(matlabdict['layer_node_count_input_1'][0])]
	for i in range(num_layers):
		l = matlabdict['layer_node_count_output_' + str(i+1)][0]
		a = matlabdict['layer_activation_' + str(i+1)]
		layers.append(nnet.layer(l,a))
	net = nnet.net(layers)

	for i in range(num_layers):
		net.layer[i].weights = matlabdict['layer_weights_' + str(i+1)]
		dropout = matlabdict['layer_dropout_' + str(i+1)][0]
		if(dropout == 'None'):
			#print('Layer ' + str(i) + ': Dropout is none')
			net.layer[i].dropout = None
		else:
			#print('Layer ' + str(i) + ': Dropout: ' + str(dropout))
			net.layer[i].dropout = float(dropout)
	data = {}
	data['net'] = net
	data['sample_mean'] = matlabdict['sample_mean']
	data['sample_std'] = matlabdict['sample_std']
	data['patchsize'] = matlabdict['patchsize']
	data['test_rate'] = matlabdict['test_rate']
	return data

