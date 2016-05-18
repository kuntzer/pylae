#=================================================================================================#

import numpy as np
import pylab as plt
import pylae
import pylae.utils as u
import copy
import tools
import sklearn.decomposition as dec
import os 

#=================================================================================================#

network_name = 'output/dA/test25'
path_in_dataset = 'output/psf_nonoise/moffat.pkl'
min_scale = 0.0005
max_scale = 0.0012

ids_train = range(5000)
ids_validation = range(5000, 7000)
ids_test = range(7500, 10000)

#=================================================================================================#

print 'Datasets...',

noisy = u.readpickle(path_in_dataset)
print np.amin(noisy), np.amax(noisy), '<<<'


nonoise = copy.copy(noisy)
print np.amin(noisy), np.amax(noisy), '<<<'
little_noisy = copy.copy(noisy)
very_noisy = copy.copy(noisy)

noisy = [n + np.random.normal(scale=np.random.uniform(min_scale, max_scale), size=n.shape) for n in noisy]
little_noisy = [n + np.random.normal(scale=np.random.uniform(min_scale/2., max_scale/2.), size=n.shape) for n in little_noisy]
very_noisy = [n + np.random.normal(scale=np.random.uniform(max_scale*1.0, max_scale*1.100), size=n.shape) for n in very_noisy]

#very_noisy, noisy_mean, noisy_std = u.standardize(very_noisy)
#noisy, _, _ = u.standardize(noisy, noisy_mean, noisy_std)
#nonoise, _, _ = u.standardize(nonoise, noisy_mean, noisy_std)
#little_noisy, _, _ = u.standardize(little_noisy, noisy_mean, noisy_std)

very_noisy, noisy_low, noisy_high = u.normalise(very_noisy)
noisy, noisy_low, noisy_high = u.normalise(noisy, noisy_low, noisy_high)
nonoise, nonoise_low, nonoise_high = u.normalise(nonoise, noisy_low, noisy_high)
little_noisy, _, _ = u.normalise(little_noisy, noisy_low, noisy_high)

data_std = [0.,1.]#[noisy_mean, noisy_std]
data_norm = [noisy_low, noisy_high]
'''
noisy_low = 0. 
noisy_high = 1.
nonoise_low = 0.
nonoise_high = 1.
'''
train_noisy = noisy[ids_train]
train_nonoise = nonoise[ids_train]
train_lnoise = little_noisy[ids_train]

test_noisy = noisy[ids_test]
test_nonoise = nonoise[ids_test]

print 'loaded.'

print '------------------'
print np.amin(test_nonoise), np.amax(test_nonoise)
print np.amin(test_noisy), np.amax(test_noisy), '<<<'
print np.amin(train_nonoise), np.amax(train_nonoise)
print np.amin(train_noisy), np.amax(train_noisy), '<<<'
print np.amin(very_noisy), np.amax(very_noisy)

"""
plt.figure()
plt.imshow(test_nonoise[3].reshape(24,24), interpolation='None')
plt.show()

exit()
"""
#=================================================================================================#
layer_type='rmse'
dA = pylae.dA.AutoEncoder(network_name, layer_type=layer_type)

n_pca=25
architecture = [n_pca]
layers_type = ["SIGMOID"] * (len(architecture) + 1)

#n_pca=200
#architecture = [n_pca]
#layers_type = ["SIGMOID", "SIGMOID"]

redo = True
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

#cost_fct = 'cross-entropy'
cost_fct = 'L2'
corruption = [0,max_scale * 1.08]# train_noisy#[0.000, 0.0012]#[0., 0.001]#[0.0, 0.001]#0.2# 0.0#[0., 0.3]

if pre_train:
	dA.pre_train(train_nonoise, architecture, layers_type, iterations=2000, mini_batch=0,\
				 corruption=corruption, data_std=data_std, data_norm=data_norm, regularisation=0.00003)
	u.writepickle(dA.rbms, "%s/%s/layers.pkl" % (network_name, layer_type))
else:
	rbms = u.readpickle("%s/%s/layers.pkl" % (network_name, layer_type))
	dA.set_autoencoder(rbms)
	dA.is_pretrained = True

if train:
	#dA = u.readpickle("%s/dA/ae.pkl" % network_name)
	dA.fine_tune(train_nonoise, iterations=4000, regularisation=0.00003, sparsity=0.0, beta=0., corruption=corruption, cost_fct=cost_fct)#regularisation=0.000, sparsity=0.05, beta=3.)

	dA.save()
	dA.display_train_history()
	
else:
	dA = u.readpickle("%s/%s/ae.pkl" % (network_name, layer_type))
	print 'AE loaded!'
	
print 'PCA decomposition...',
path_out_pcabasis = os.path.join(network_name, 'pcabasis.pkl')
if pca_train:

	pca = dec.PCA(n_pca)
	pca.fit(train_nonoise)
	u.writepickle(pca, path_out_pcabasis)
	
	print 'Trained.'
else:
	pca = u.readpickle(path_out_pcabasis)
	print 'loaded.'
	

#=================================================================================================#

test_tilde = dA.decode(dA.encode(test_noisy))
test_pca_tilde = pca.inverse_transform(pca.transform(test_noisy))
print np.amin(test_nonoise), np.amax(test_nonoise)
print np.amin(test_noisy), np.amax(test_noisy)
print np.amin(test_pca_tilde), np.amax(test_pca_tilde)
print np.amin(test_tilde), np.amax(test_tilde)

datasets = [test_nonoise, test_noisy, test_tilde, test_pca_tilde]
for ds in datasets:
	ds = u.unnormalise(test_pca_tilde, noisy_low, noisy_high)
	#ds = u.unstandardize(ds, noisy_mean, noisy_std)
print '------------------'
print np.amin(test_nonoise), np.amax(test_nonoise)
print np.amin(test_noisy), np.amax(test_noisy)
print np.amin(test_pca_tilde), np.amax(test_pca_tilde)
print np.amin(test_tilde), np.amax(test_tilde)

#=================================================================================================#

tools.analysis_plots(datasets, outdir=network_name, do_meas_params=do_meas_params)

tools.reconstruction_plots(test_nonoise, test_noisy, test_tilde, test_pca_tilde)
