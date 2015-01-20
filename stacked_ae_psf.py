import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os

network_name = "gauss_psf_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
#dataset = np.loadtxt("data/psfs-gs_rnd.dat", delimiter=",")
dataset = np.loadtxt("data/psfs-smalldev.dat", delimiter=",")

dataset, low, high = utils.normalise(dataset)
size = np.sqrt(np.shape(dataset)[1])

# Can we skip some part of the training ?
train = True

# Separate into training and testing data
datasize = np.shape(dataset)[0]
datasize = 2000
trainper = 0.7
ind = np.int(trainper * datasize)

train_data = dataset[0:ind]
test_data = dataset[ind:datasize]
#test_data[0] = np.zeros_like(test_data[0])

print 'Shape of the training set: ', np.shape(train_data)
print 'Shape of the testing set: ', np.shape(test_data)

architectures = [[784, 500],
				 [256, 15]]

ae = []
for ii, architecture in enumerate(architectures):
	layers_type = ["SIGMOID"] * len(architecture) + ["LINEAR"]
	network_namei = '%s%d' % (network_name, ii)
	ae.append(pylae.autoencoder.AutoEncoder(network_namei))
	if ii > 0:
		tdata = ae[ii-1].feedforward(train_data)
	ae[ii].pre_train(tdata, architecture, layers_type, learn_rate={'SIGMOID':0.0034, 'LINEAR':0.0034/10.}, 
		initialmomentum=0.53,finalmomentum=0.93, iterations=2000, mini_batch=100, regularisation=0.001)
	ae[ii].backpropagation(train_data, iterations=1000, learn_rate=0.13, momentum_rate=0.83)
utils.writepickle(ae, '%sstacked_ae' % network_name)
	
for ii in len(architectures):
	reconstruc = ae[ii].feedforward(train_data)
