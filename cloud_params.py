#load_percentage = 0.12
#load_percentage = 1.0

#tells how much of each class to load
#This comes from the percentages of each class in the image
#2020868,   6077934,  31901198
f = 1.0 # actual load percentage here
class_load_percentage = [f*1,f*(1.0/3.0076),f*(1.0/15.7859)]

#280822   834253  2884925
f = 0.15
class_test_load_percentage = [1.0*f,(1.0/2.9708)*f,(1.0/10.2731)*f]

pixel_mean = [0.1450149,0.14385642,0.12862228,0.25409362,0.18768486,7.74838924,0.12216207]
pixel_std = [0.11748467,0.13816862,0.13534597,0.20941748,0.1624822 ,1.1833992,0.12372045]

patchsize = 5
data_path = '/local_scratch/cloudmasks/mat_training_notime'
hidden_sizes = [160,128,96]
hidden_activations = ['linear_rectifier','linear_rectifier','linear_rectifier']
output_activation = 'sigmoid'
step_size = .015
dropout_percentage = 0.5
training_epochs = 5000
minibatch_size = 2500


#test_load_percentage = .020

shuffle_epochs = 10 #how many epochs to train before shuffling

save_interval = 20*60 # save network every n minutes

# -- everything below this line is not used by the in memory trainer
num_batches = 50
#if this is true, we use h5py to store all of our samples in a file as we load them
use_sample_storage = True
sample_dir_name = '/local_scratch/cloudmasks/samples'
randomseed1 = 12345;

#should never have to modify these...
chunk_size = 1000
chunk_append_size = 500
