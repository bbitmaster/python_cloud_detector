load_percentage = 0.10
patchsize = 3
data_path = '/local_scratch/cloudmasks/mat_training_notime'
hidden_sizes = [128,128,128]
hidden_activations = ['linear_rectifier','linear_rectifier','linear_rectifier']
output_activation = 'sigmoid'
step_size = .01
dropout_percentage = 0.5
training_epochs = 5000
minibatch_size = 10000

test_percent = .05
validation_percent = .05
