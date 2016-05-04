#=================================================================================================#

import numpy as np
import pylab as plt
import pylae
import pylae.utils as u
import copy
import sklearn.decomposition as dec
import os 
import tools

#=================================================================================================#

network_name = 'output/ae/test'
path_in_dataset = 'output/psf_nonoise/moffat.pkl'
scale = 0.0012

ids_train = range(5000)
ids_validation = range(5000, 7000)
ids_test = range(7500, 10000)

#=================================================================================================#

print 'Datasets...',

dataset = u.readpickle(path_in_dataset)
dataset, _, _ = u.standardize(dataset)
nonoise = copy.copy(dataset)

dataset += np.random.normal(scale=scale, size=dataset.shape)

dataset_nn = copy.copy(dataset)
dataset_low, dataset_high = 0., 1.
#dataset, dataset_low, dataset_high = u.normalise(dataset)


train_dataset = dataset[ids_train]

test_dataset = dataset[ids_test]
test_nonoise = nonoise[ids_test]

print 'loaded.'


plt.figure()
plt.imshow(dataset[3].reshape(24,24), interpolation='None')
plt.show()

#=================================================================================================#
ae = pylae.autoencoder.AutoEncoder(network_name)

n_pca=100
architecture = [1024, 512, n_pca]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]

n_pca=100
architecture = [n_pca]
layers_type = ["SIGMOID"] * (len(architecture)) + ['LINEAR']

redo = False
if redo :
	pre_train = True
	train = True
	pca_train = True
	do_meas_params = True
else:
	pre_train = False
	train = False
	pca_train = False
	do_meas_params = False
	
do_meas_params = True

cost_fct = 'cross-entropy'
cost_fct = 'L2'
print 'Auto-encoder...',

if pre_train:
	
	ae.pre_train(train_dataset, architecture, layers_type, learn_rate={'SIGMOID':0.05, 'LINEAR':0.05/10.}, 
				initialmomentum=0.5,finalmomentum=0.9, iterations=2000, mini_batch=50, regularisation=0.)
	u.writepickle(ae.rbms, "%s/gd/rbms.pkl" % network_name)
	print 'AE pre-trained.'
else:
	rbms = u.readpickle("%s/gd/rbms.pkl" % network_name)
	ae.set_autoencoder(rbms)
	ae.is_pretrained = True

if train:
	ae.fine_tune(train_dataset, iterations=5000, regularisation=0., sparsity=0.0, beta=0., cost_fct=cost_fct)#regularisation=0.000, sparsity=0.05, beta=3.)
	ae.save()
	print 'AE trained.'
	
else:
	
	ae = u.readpickle("%s/gd/ae.pkl" % network_name)
	print 'loaded.'
	
print 'PCA decomposition...',
path_out_pcabasis = os.path.join(network_name, 'pcabasis.pkl')
if pca_train:

	pca = dec.PCA(n_pca)
	pca.fit(dataset[ids_train])
	u.writepickle(pca, path_out_pcabasis)
	
	print 'Trained.'
	
	ae.display_train_history()
	
else:
	pca = u.readpickle(path_out_pcabasis)
	print 'loaded.'
	
#=================================================================================================#

test_tilde = ae.decode(ae.encode(test_dataset))
test_tilde = u.unnormalise(test_tilde, dataset_low, dataset_high)

test_pca_tilde = pca.inverse_transform(pca.transform(test_dataset))
test_pca_tilde = u.unnormalise(test_pca_tilde, dataset_low, dataset_high)
test_dataset = u.unnormalise(test_dataset, dataset_low, dataset_high)

#=================================================================================================#


datasets = [test_nonoise, test_dataset, test_tilde, test_pca_tilde]
tools.analysis_plots(datasets, outdir=network_name, do_meas_params=do_meas_params)

tools.reconstruction_plots(test_nonoise, test_dataset, test_tilde, test_pca_tilde)
