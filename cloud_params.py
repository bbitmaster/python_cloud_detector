load_percentage = 0.12
patchsize = 5
data_path = '/local_scratch/cloudmasks/mat_training_notime'
hidden_sizes = [128,128,128]
hidden_activations = ['linear_rectifier','linear_rectifier','linear_rectifier']
output_activation = 'sigmoid'
step_size = .025
dropout_percentage = None
training_epochs = 5000
minibatch_size = 5000


test_load_percentage = .020

shuffle_epochs = 5 #how many epochs to train before shuffling

save_interval = 5*60 # save network every n minutes

# -- everything below this line is not used by the in memory trainer
num_batches = 50
#if this is true, we use h5py to store all of our samples in a file as we load them
use_sample_storage = True
sample_dir_name = '/local_scratch/cloudmasks/samples'
randomseed1 = 12345;

#should never have to modify these...
chunk_size = 1000
chunk_append_size = 500
