import h5py
import numpy
import os

#possible todo - write function to convert to npz files for loading without h5py

def load_data(filename):
	f = h5py.File(filename)
	A = numpy.array(f['A'])
	MASK = numpy.array(f['MASK'])
	MASKF = numpy.array(f['MASKF'])
	return (A,MASK,MASKF)

def load_all_data(dirname):
	A_list = []
	MASK_list = []
	MASKF_list = []
	for f in os.listdir(dirname):
		#skip p61r2 - it is polar region and seems to lack a mask
		if(f.startswith('p61r2')):
			continue
		if(f.endswith('mat')):
			print('loading: ' + f)
			data = load_data(os.path.join(dirname,f))
			A_list.append(data[0])
			MASK_list.append(data[1])
			MASKF_list.append(data[2])
	return(A_list,MASK_list,MASKF_list);

