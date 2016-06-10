
#=================================================================================================#

import numpy as np
import pylab as plt
import pylae
import pylae.utils as u
import copy
import tools
import sklearn.decomposition as dec
import os 
from matplotlib import gridspec

#=================================================================================================#

network_name = 'output/dA/test25'
path_in_dataset = 'output/psf_nonoise/13spectra_PSF.pkl'
path_in_dataset = 'output/psf_nonoise/moffat_rel.pkl'
path_in_dataset = 'output/psf_nonoise/moffat.pkl'
#path_in_dataset = 'output/psf_nonoise/13spectra_PSF_smallpx_smallbatch.pkl'
min_scale = 0.0005
max_scale = 0.0012

ids_train = range(4000)
ids_validation = range(4000, 4005)
ids_test = range(4005, 5200)

ids_train = range(5000)
ids_validation = range(5000, 5002)
ids_test = range(5002, 7005)

#ids_train = range(600)
#ids_validation = range(600, 6002)
#ids_test = range(600, 845)


layer_type='rmse'

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

if not train:
	dA = u.readpickle("%s/%s/ae.pkl" % (network_name, layer_type))
	noisy_low, noisy_high = dA.layers[0].data_norm
else:
	noisy_low, noisy_high = None, None
	
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

very_noisy, noisy_low, noisy_high = u.normalise(very_noisy, noisy_low, noisy_high)
noisy, noisy_low, noisy_high = u.normalise(noisy, noisy_low, noisy_high)
nonoise, nonoise_low, nonoise_high = u.normalise(nonoise, noisy_low, noisy_high)
little_noisy, _, _ = u.normalise(little_noisy, noisy_low, noisy_high)

data_std = [0.,1.]#[noisy_mean, noisy_std]
data_norm = [noisy_low, noisy_high]
print 'data norm', data_norm
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

#=================================================================================================#
dA = pylae.dA.AutoEncoder(network_name, layer_type=layer_type)

n_pca=25
architecture = [n_pca]
layers_type = ["RELU"] * (len(architecture) + 1)

cost_fct = 'cross-entropy'
cost_fct = 'L2'
corruption =[0., max_scale * 1.05]# train_noisy#[0.000, 0.0012]#[0., 0.001]#[0.0, 0.001]#0.2# 0.0#[0., 0.3]
regularisation = 0*2.e-5

#train_data_ae = np.tile(train_nonoise.T, 10).T
train_data_ae = train_nonoise

print train_data_ae.shape

if pre_train:
	dA.pre_train(train_data_ae, architecture, layers_type, iterations=2000, mini_batch=0,\
				 corruption=corruption, data_std=data_std, data_norm=data_norm, regularisation=regularisation)
	u.writepickle(dA.rbms, "%s/%s/layers.pkl" % (network_name, layer_type))
else:
	rbms = u.readpickle("%s/%s/layers.pkl" % (network_name, layer_type))
	dA.set_autoencoder(rbms)
	dA.is_pretrained = True

if train:
	#dA = u.readpickle("%s/dA/ae.pkl" % network_name)
	dA.fine_tune(train_data_ae, iterations=4000, regularisation=regularisation, sparsity=0., beta=0., corruption=corruption, cost_fct=cost_fct)#regularisation=0.000, sparsity=0.05, beta=3.)

	dA.save()
	dA.display_train_history()
	
else:
	dA = u.readpickle("%s/%s/ae.pkl" % (network_name, layer_type))
	print 'AE loaded!'
	
dA.display_network()

print [x/y for x,y in zip(dA.layers[0].data_norm, data_norm)]
	
print 'PCA decomposition...',
path_out_pcabasis = os.path.join(network_name, 'pcabasis.pkl')

if pca_train:
	pca = dec.PCA(n_pca)
	pca.fit(train_noisy)
	u.writepickle(pca, path_out_pcabasis)
	print 'trained.'
	
else:
	pca = u.readpickle(path_out_pcabasis)
	print 'loaded.'
	
figC = copy.deepcopy(pca.components_)

fig=plt.figure(figsize=(7,8.5))
xxx = np.int(np.ceil(np.sqrt(np.shape(figC)[0])))
gs = gridspec.GridSpec(xxx, xxx)
gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
ix = 0
iy = 0
for ii in range(np.shape(figC)[0]):
	#if ii > 15: break
	ax = fig.add_subplot(gs[iy,ix])
	#plt.subplot('44%d' % ii)
	sfigc = np.sqrt(np.size(figC[ii]))
	img_c = figC[ii].reshape(sfigc, sfigc)
	#img_c -= np.amin(img_c)
	img_c -= np.median(img_c)
	#img_c -= np.amin(img_c)
	if np.amax(img_c) > -np.amin(img_c):
		img_c /= np.amax(img_c)
	else:
		img_c /= -np.amin(img_c)
	ax.imshow(img_c, interpolation="nearest", cmap=plt.get_cmap("gray"), vmin=-1.,vmax=1.)
	ax.text(23,23,r'$%d$' % (ii))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_aspect('equal')
	
	
	ix += 1
	if ix > xxx-1:
		ix = 0
		iy += 1

#=================================================================================================#

test_tilde = dA.decode(dA.encode(test_noisy))
test_pca_tilde = pca.inverse_transform(pca.transform(test_noisy))

datasets = [test_nonoise, test_noisy, test_tilde, test_pca_tilde]
for ds in range(len(datasets)):
	pass#datasets[ds] = u.unnormalise(datasets[ds], noisy_low, noisy_high)
	#ds = u.unstandardize(ds, noisy_mean, noisy_std)

#=================================================================================================#

tools.analysis_plots(datasets, outdir=network_name, do_meas_params=do_meas_params)

tools.reconstruction_plots(test_nonoise, test_noisy, test_tilde, test_pca_tilde)
