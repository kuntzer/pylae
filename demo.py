import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os	
import copy	
		
network_name = "demo_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
data = np.loadtxt("data/digit0.dat", delimiter=",")
data, _ , _ = utils.normalise(data)
data = data.T
size = np.sqrt(np.shape(data)[1])

# Can we skip some part of the training ?
pre_train = False
train = False

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [256, 64, 16]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]

# Let's go
ae = pylae.autoencoder.AutoEncoder('demo')
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	ae.pre_train(data, architecture, layers_type, learn_rate={'SIGMOID':0.1, 'LINEAR':1e-3}, 
				iterations=2000, mini_batch=100)
	
	# Save the resulting layers
	utils.writepickle(ae.rbms, "%srbms.pkl" % network_name)
	
elif not pre_train and train :
	rbms = utils.readpickle("%srbms.pkl" % network_name)
	# An autoencoder instance was created some lines earlier, preparing the other half of the 
	# network based on layers loaded from the pickle file. 
	ae.set_autencoder(rbms)
	
if train:
	print 'Starting backpropagation'
	ae.backpropagation(data, iterations=500, learn_rate=0.001, momentum_rate=0.9)

	ae.save("%sautoencoder.pkl" % network_name)
	
	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

else :
	ae = utils.readpickle("%sautoencoder.pkl" % network_name)

# Use the training data as if it were a training set
data, _ , _ = utils.normalise(data)
test = copy.deepcopy(data)
np.random.shuffle(test)

reconstruc = ae.feedforward(test)

# Compute the RMSD error for the training set
rmsd = []
for ii in range(np.shape(test)[0]) :
	img = test[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size)
	rmsd.append(np.sqrt(np.mean((img - recon)*(img - recon))))

# Show the figures for the distribution of the RMSD and the learning curves
plt.figure()
plt.hist(rmsd)

ae.plot_rmsd_history()

# Show the original image, the reconstruction and the residues for the first 10 cases
for ii in range(10):
	img = test[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size)

	plt.figure()
	plt.subplot(1, 3, 1)
	plt.imshow((img), interpolation="nearest")
	plt.subplot(1, 3, 2)
	plt.imshow((recon), interpolation="nearest")
	plt.subplot(1, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
plt.show()