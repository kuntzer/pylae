import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os		
		
network_name = "gauss_psf_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
dataset = np.loadtxt("data/psfs-gs_rnd.dat", delimiter=",")
dataset, low, high = utils.normalise(dataset)
size = np.sqrt(np.shape(dataset)[0])

# Can we skip some part of the training ?
pre_train = True
train = True

# Separate into training and testing data
train_data = dataset[:,0:1000]
test_data = dataset[:,1000:2000]

print 'Shape of the training data: ', np.shape(train_data)
print 'Shape of the training data: ', np.shape(test_data)

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [784, 392, 196, 98, 8]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]

# Let's go
ae = pylae.autoencoder.AutoEncoder(network_name)
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':3e-3, 'LINEAR':3e-4}, 
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
	ae.backpropagation(train_data, iterations=500, learn_rate=0.001, momentum_rate=0.9)

	ae.save("%sautoencoder.pkl" % network_name)
	
	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

else :
	ae = utils.readpickle("%sautoencoder.pkl" % network_name)

# Use the training data as if it were a training set
reconstruc = ae.feedforward(test_data)

# Compute the RMSD error for the training set
rmsd_train = []
for ii in range(np.shape(train_data)[1]) :
	img = train_data[:,ii].reshape(size,size)
	recon = reconstruc[:,ii].reshape(size,size)
	rmsd_train.append(np.sqrt(np.mean((img - recon)*(img - recon))))

# Compute the RMSD error for the test set
rmsd_test = []
for ii in range(np.shape(test_data)[1]) :
	img = test_data[:,ii].reshape(size,size)
	recon = reconstruc[:,ii].reshape(size,size)
	rmsd_test.append(np.sqrt(np.mean((img - recon)*(img - recon))))

# Show the figures for the distribution of the RMSD and the learning curves
plt.figure()
plt.hist(rmsd_train, label="training")
plt.hist(rmsd_test, label="test")
plt.legend(loc="best")

ae.plot_rmsd_history()

# Show the original test image, the reconstruction and the residues for the first 10 cases
for ii in range(20):
	img = train_data[:,ii].reshape(size,size)
	recon = reconstruc[:,ii].reshape(size,size)

	plt.figure()
	plt.subplot(1, 3, 1)
	plt.imshow((img), interpolation="nearest")
	plt.subplot(1, 3, 2)
	plt.imshow((recon), interpolation="nearest")
	plt.subplot(1, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
plt.show()