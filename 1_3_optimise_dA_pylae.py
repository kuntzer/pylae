
#=================================================================================================#

import numpy as np
import pylae
import pylae.utils as u
import copy
import tools
import sklearn.decomposition as dec
import os 
import itertools 
from datetime import datetime

#=================================================================================================#

workdir_global = 'output/optimisation/dA_pylae/'
path_in_dataset_train = 'output/psf_nonoise/moffat_rel.pkl'
path_in_dataset_test = 'output/psf_nonoise/moffat.pkl'

min_scale = 0.0005
max_scale = 0.0012

ids_train = range(5000)
ids_validation = range(5000, 5002)
ids_test = range(5002, 8002)

#=================================================================================================#

nb_comps = [8, 16, 25, 100]
#layer_types = [['rmse', 'L2'], ['dA', 'L2'], ['rmse', 'cross-entropy'], ['dA', 'cross-entropy']]
costs = [['rmse', 'L2'], ['dA', 'cross-entropy']]
activation_fcts = ['RELU', 'SIGMOID', 'LEAKY_RELU']
#final_activations = [None, 'LINEAR']
regularisations = [0., -1e-6, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
corruptions = [None, [0., max_scale * 1.1], [0.5*min_scale, max_scale * 1.1], [0.5*min_scale, max_scale * 0.8], 0.05, 0.3]
sparsities = [[0.,0.], [3., 0.02], [3., 0.05]]
a = [nb_comps, costs, activation_fcts, regularisations, corruptions, sparsities]
iteration_pre_train = 20#000
iteration_train = 40#000

b = list(itertools.product(*a))

#=================================================================================================#
print 'Preparing for %d configurations' % len(b)

# Getting the data --------------------------------------------------------------------------------
noisy_train = u.readpickle(path_in_dataset_train)
noisy_test = u.readpickle(path_in_dataset_test)

nonoise_train = copy.copy(noisy_train)
nonoise_test = copy.copy(noisy_test)

# TODO: Do this later, when preparing the configu
noisy_train = [n + np.random.normal(scale=np.random.uniform(min_scale, max_scale), size=n.shape) for n in nonoise_train]
noisy_test = [n + np.random.normal(scale=np.random.uniform(min_scale, max_scale), size=n.shape) for n in nonoise_test]

noisy_low = None
noisy_high = None

noisy_train, noisy_low, noisy_high = u.normalise(noisy_train, noisy_low, noisy_high)
nonoise_train, nonoise_low, nonoise_high = u.normalise(nonoise_train, noisy_low, noisy_high)

noisy_test, noisy_low, noisy_high = u.normalise(noisy_test, noisy_low, noisy_high)
nonoise_test, nonoise_low, nonoise_high = u.normalise(nonoise_test, noisy_low, noisy_high)

data_std = [0.,1.]#[noisy_mean, noisy_std]
data_norm = [noisy_low, noisy_high]
# Data ready --------------------------------------------------------------------------------------

# Go for main engine start ------------------------------------------------------------------------
for id_test, (nb_comp, cost, activation_fct, regularisation, corruption, sparsity) in enumerate(b):
	start_time = datetime.now()
	
	print 'Starting evaluating configuration %d/%d at %s' % (id_test, len(b), start_time)
	
	# Naming the realisation and trying making a new directory
	name_cfg = 'id_%d' % (id_test)
	workdir = os.path.join(workdir_global, name_cfg)
	
	# Are we restarting some work from an existing workdir ?
	if os.path.isdir(workdir):
		print "Detected workdir %s" % (workdir)
		print 'NEED TO LOAD NOW OR AT LEAST TELL TO LOAD'
		exit()
	else:
		u.mkdir(workdir)
		
		# Don't want to forget how we configured the AE, also machine readable form.
		config = open(os.path.join(workdir,'config.cfg'),'w')
		print >> config, 'nb_comp', nb_comp
		print >> config, 'cost_pre_train', cost[0]
		print >> config, 'cost_train', cost[1]
		print >> config, 'activation_fct', activation_fct
		print >> config, 'regularisation', regularisation
		print >> config, 'corruption', corruption
		print >> config, 'sparsity_beta', sparsity[0]
		print >> config, 'sparsity_rho', sparsity[1]
		print >> config, 'iteration_pre_train', iteration_pre_train
		print >> config, 'iteration_train', iteration_train
		config.close()
		
		# TODO: replace here by multiprocessing
		
		# Starting an instance of dA
		dA = pylae.dA.AutoEncoder(directory=workdir, layer_type=cost[0], mkdir=False)
		
		# Defining the architecture
		architecture = [nb_comp]
		layers_type = [activation_fct] * (len(architecture) + 1)
		
		print 'Pre-training for config %d at %s' % (id_test, datetime.now())
		dA.pre_train(nonoise_train, architecture, layers_type, iterations=iteration_pre_train, mini_batch=0,\
					 corruption=corruption, data_std=data_std, data_norm=data_norm, regularisation=regularisation)
		print 'Pre-training terminated for config %d at %s' % (id_test, datetime.now())
		
		print 'Fine-tuning starting for config %d at %s' % (id_test, datetime.now())
		dA.fine_tune(nonoise_train, iterations=iteration_train, regularisation=regularisation, \
				 sparsity=sparsity[1], beta=sparsity[0], corruption=corruption, cost_fct=cost[1])
		print 'Fine-tuning terminated for config %d at %s' % (id_test, datetime.now())
	
		dA.save(workdir)
		
		fig = dA.display_network(layer_number=0, show=False)
		pylae.figures.savefig(os.path.join(workdir, 'weights'), fig)
		
		test_tilde = dA.decode(dA.encode(noisy_test))
		
		# The AE has been trained. Now let's have a look at its performance.
		datasets = [nonoise_test, noisy_test, test_tilde]
		for ds in range(len(datasets)):
			datasets[ds] = u.unnormalise(datasets[ds], noisy_low, noisy_high)
		results = tools.analysis(datasets, outdir=workdir)
		np.savetxt(os.path.join(workdir, 'rslt.dat'), results)

		
		
	print 'Evaluating configuration %d/%d terminated at %s' % (id_test, len(b), datetime.now())
	print '\n'




# TODO : PCA decompositons for the different n_comp as well (and on noisy + nonoise).
