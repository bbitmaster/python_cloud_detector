import h5py
import numpy
import os

#possible todo - write function to convert to npz files for loading without h5py

#loads from a .mat file that must contain the variables
#'A' 'MASK' 'MASKF'
def load_data(filename):
	f = h5py.File(filename)
	A = numpy.array(f['A'])
	MASK = numpy.array(f['MASK'])
	MASKF = numpy.array(f['MASKF'])
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
	from cloud_params import *
	fname = 'cloud_sample_cache_' + str(patchsize) + '_' + str(load_percentage) + '_' + str(randomseed1) + '.hdf5'
	return os.path.join(sample_dir_name,fname)
